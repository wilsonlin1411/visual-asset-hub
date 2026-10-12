import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.services.library import AssetLibrary
from backend.services.search import SearchService
from backend.services.vocab import load_vocabulary
from fakes import SPACE, FakeAnalyzer, FakeTextEncoder
from test_library import FakePipeline, scene


@pytest.fixture
def client(tmp_path):
    lib = AssetLibrary(tmp_path / "data", pipeline_factory=FakePipeline, analyzer_factory=FakeAnalyzer)
    app = create_app(lib, search_factory=lambda l: SearchService(l, FakeTextEncoder(), load_vocabulary(), SPACE),
                     serve_frontend=False)
    with TestClient(app) as c:
        c.lib = lib
        yield c
    lib.close()


def upload(client, name="桌面.png"):
    data = cv2.imencode(".png", scene())[1].tobytes()
    return client.post("/api/photos", files={"file": (name, data, "image/png")})


def wait_ready(client, photo_id, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        p = client.get(f"/api/photos/{photo_id}").json()
        if p["status"] != "processing":
            return p
        time.sleep(0.05)
    raise AssertionError("photo still processing")


def decode(resp):
    return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_UNCHANGED)


def ready_photo(client):
    pid = upload(client).json()["photo"]["id"]
    wait_ready(client, pid)
    return pid


def test_upload_process_confirm_search_delete(client):
    r = upload(client)
    assert r.status_code == 202 and r.json()["created"]
    pid = r.json()["photo"]["id"]
    photo = wait_ready(client, pid)
    assert photo["status"] == "ready"
    assert [c["value"] for c in photo["contexts"]] == ["咖啡廳"]
    assert photo["counts"] == {"pending": 2, "confirmed": 0, "discarded": 0, "assets": 0}

    again = upload(client, "copy.png").json()
    assert not again["created"] and again["photo"]["id"] == pid

    cands = client.get(f"/api/photos/{pid}/candidates").json()
    assert cands["hidden_count"] == 1                       # unlabeled, objectness 0.2 < threshold
    assert [c["idx"] for c in cands["candidates"]] == [0]
    everything = client.get(f"/api/photos/{pid}/candidates", params={"include_hidden": True}).json()
    assert [(c["idx"], c["hidden"]) for c in everything["candidates"]] == [(0, False), (1, True)]
    assert client.get(everything["candidates"][0]["mask_url"]).headers["content-type"] == "image/png"

    assets = client.post("/api/assets/confirm", json={"photo_id": pid, "idxs": [0]}).json()
    assert len(assets) == 1 and assets[0]["category"] == "咖啡杯"
    assert {(t["kind"], t["value"]) for t in assets[0]["tags"]} == {("attribute", "紅"), ("context", "咖啡廳")}
    aid = assets[0]["id"]
    png = client.get(f"/api/assets/{aid}/png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"

    assert [a["id"] for a in client.get("/api/assets", params={"tag": "咖啡廳"}).json()] == [aid]
    res = client.get("/api/search", params={"q": "咖啡廳的咖啡杯"}).json()
    assert res["categories"] == ["咖啡杯"] and res["contexts"] == ["咖啡廳"]
    top = res["hits"][0]
    assert top["asset"]["id"] == aid and any("情境 咖啡廳" in r for r in top["reasons"])

    assert client.delete(f"/api/assets/{aid}").json() == {"deleted": aid}
    assert client.get(f"/api/assets/{aid}").status_code == 404
    assert client.get("/api/stats").json()["assets"] == 0


def test_thumbnails_are_small_and_cached(client):
    pid = ready_photo(client)
    display = client.get(f"/api/photos/{pid}/display")
    assert display.headers["content-type"] == "image/jpeg" and decode(display).shape[:2] == (300, 400)

    cand = client.get(f"/api/photos/{pid}/candidates").json()["candidates"][0]
    preview = decode(client.get(cand["preview_url"]))
    assert preview.shape[2] == 4 and max(preview.shape[:2]) <= 256
    overlay = decode(client.get(cand["overlay_url"]))
    x0, y0, x1, y1 = cand["bbox"]
    assert overlay.shape[:2] == (y1 - y0, x1 - x0) and overlay[..., 3].max() == 255

    [asset] = client.post("/api/assets/confirm", json={"photo_id": pid, "idxs": [0]}).json()
    thumb = client.get(asset["thumb_url"])
    assert thumb.headers["content-type"] == "image/webp" and decode(thumb).shape[2] == 4
    assert client.lib.thumbs.asset_path(asset["id"]).is_file()      # written at confirm time

    client.delete(f"/api/assets/{asset['id']}")
    assert not client.lib.thumbs.asset_path(asset["id"]).exists()


def test_user_category_and_tag_edits_survive_recomputation(client):
    pid = ready_photo(client)
    [a] = client.post("/api/assets/confirm", json={"photo_id": pid, "idxs": [0]}).json()
    aid = a["id"]

    edited = client.patch(f"/api/assets/{aid}", json={"category": "公仔"}).json()
    assert (edited["category"], edited["category_source"]) == ("公仔", "user")
    assert client.patch(f"/api/assets/{aid}", json={"category": "不存在"}).status_code == 400

    r = client.post(f"/api/assets/{aid}/tags", json={"kind": "context", "value": "收藏"}).json()
    assert ("context", "收藏", "user") in {(t["kind"], t["value"], t["source"]) for t in r["tags"]}
    r = client.delete(f"/api/assets/{aid}/tags", params={"kind": "context", "value": "咖啡廳"}).json()
    assert "咖啡廳" not in {t["value"] for t in r["tags"]}
    assert client.post(f"/api/assets/{aid}/tags", json={"kind": "context", "value": "亂寫"}).status_code == 400

    # confirming another candidate recomputes the photo's contexts: user edits must stick
    client.post("/api/assets/confirm", json={"photo_id": pid, "idxs": [1]})
    tags = {(t["kind"], t["value"]) for t in client.get(f"/api/assets/{aid}").json()["tags"]}
    assert ("context", "收藏") in tags and ("context", "咖啡廳") not in tags
    assert client.get(f"/api/assets/{aid}").json()["category"] == "公仔"


def test_delete_photo_and_vocabulary(client):
    pid = ready_photo(client)
    client.post("/api/assets/confirm", json={"photo_id": pid})
    client.get(f"/api/photos/{pid}/display")
    assert client.delete(f"/api/photos/{pid}").json() == {"deleted": pid}
    assert client.get(f"/api/photos/{pid}").status_code == 404
    assert not client.lib.thumbs.photo_path(pid).exists()
    assert client.get("/api/stats").json()["assets"] == 0

    v = client.get("/api/vocabulary").json()
    assert len(v["categories"]) == 44 and "咖啡廳" in v["contexts"] and "藍" in v["attributes"]["color"]
    assert v["excluded_categories"] == ["人物"]


def test_errors(client):
    assert client.get("/api/photos/999").status_code == 404
    assert client.post("/api/photos", files={"file": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 400
    pid = ready_photo(client)
    assert client.post("/api/assets/confirm", json={"photo_id": pid, "idxs": [7]}).status_code == 404
    assert client.get("/api/search", params={"q": ""}).status_code == 422
    assert client.get("/photos").status_code == 404   # API lives under /api only
