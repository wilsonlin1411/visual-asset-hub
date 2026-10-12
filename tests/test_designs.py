import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.db import Database
from backend.services.designs import DesignStore, Scene
from backend.services.library import AssetLibrary
from fakes import FakeAnalyzer
from test_api import ready_photo
from test_library import FakePipeline


@pytest.fixture
def client(tmp_path):
    lib = AssetLibrary(tmp_path / "data", pipeline_factory=FakePipeline, analyzer_factory=FakeAnalyzer)
    with TestClient(create_app(lib, serve_frontend=False)) as c:
        c.lib = lib
        yield c
    lib.close()


def png(w, h, alpha=False):
    img = np.full((h, w, 4 if alpha else 3), 200, np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def make_asset(client):
    pid = ready_photo(client)
    return client.post("/api/assets/confirm", json={"photo_id": pid, "idxs": [0]}).json()[0]


SCENE = {
    "background": {"type": "gradient", "color": "#ffffff", "color2": "#eeeeee", "angle": 90},
    "layers": [
        {"id": "t1", "type": "text", "text": "新品上市", "x": 400, "y": 700, "width": 600, "font_size": 56},
    ],
}


def test_presets_and_design_crud(client):
    presets = {p["key"]: (p["width"], p["height"]) for p in client.get("/api/presets").json()}
    assert presets == {"product": (800, 800), "ig": (1080, 1350), "banner": (1640, 856)}

    d = client.post("/api/designs", json={"preset": "product"}).json()
    assert (d["width"], d["height"], d["layer_count"], d["thumb_url"]) == (800, 800, 0, None)
    assert d["scene"]["background"]["type"] == "color"

    asset = make_asset(client)
    scene = {**SCENE, "layers": SCENE["layers"] + [
        {"id": "a1", "type": "asset", "asset_id": asset["id"], "x": 400, "y": 380, "width": 500, "shadow": True}]}
    u = client.put(f"/api/designs/{d['id']}", json={"scene": scene, "name": "春季主圖"}).json()
    assert u["name"] == "春季主圖" and u["layer_count"] == 2
    assert u["scene"]["layers"][0]["font_family"] == "Microsoft JhengHei"     # defaults filled in
    assert set(u["assets"]) == {str(asset["id"])} and u["missing_assets"] == []

    # switching preset changes the size; scene coordinates are the client's job
    b = client.put(f"/api/designs/{d['id']}", json={"scene": scene, "preset": "banner"}).json()
    assert (b["width"], b["height"]) == (1640, 856)

    # a deleted asset shows up as missing instead of breaking the design
    client.delete(f"/api/assets/{asset['id']}")
    assert client.get(f"/api/designs/{d['id']}").json()["missing_assets"] == [asset["id"]]

    assert [x["id"] for x in client.get("/api/designs").json()] == [d["id"]]
    assert client.delete(f"/api/designs/{d['id']}").json() == {"deleted": d["id"]}
    assert client.get(f"/api/designs/{d['id']}").status_code == 404


@pytest.mark.parametrize("bad", [
    {"layers": [{"id": "x", "type": "sticker", "x": 0, "y": 0}]},                         # unknown layer type
    {"layers": [{"id": "x", "type": "asset", "asset_id": 1, "x": 0, "y": 0, "width": -5}]},
    {"layers": [{"id": "x", "type": "text", "text": "a", "x": 0, "y": 0, "width": 10, "font_size": 10,
                 "fill": "red"}]},                                                       # not #rrggbb
    {"background": {"type": "image"}},
])
def test_invalid_scenes_are_rejected(client, bad):
    d = client.post("/api/designs", json={"preset": "ig"}).json()
    assert client.put(f"/api/designs/{d['id']}", json={"scene": bad}).status_code == 422


def test_unknown_preset(client):
    assert client.post("/api/designs", json={"preset": "a4"}).status_code == 400


def test_export_must_match_design_size(client):
    d = client.post("/api/designs", json={"preset": "banner"}).json()
    wrong = client.post(f"/api/designs/{d['id']}/export", files={"file": ("x.png", png(1640, 855), "image/png")},
                        data={"format": "png"})
    assert wrong.status_code == 400 and "1640x855" in wrong.json()["detail"]

    ok = client.post(f"/api/designs/{d['id']}/export", files={"file": ("x.png", png(1640, 856, alpha=True), "image/png")},
                     data={"format": "png"}).json()
    assert (ok["width"], ok["height"], ok["format"]) == (1640, 856, "png")
    f = client.get(ok["url"])
    assert f.status_code == 200 and "banner-1640x856" in f.headers["content-disposition"]
    assert cv2.imdecode(np.frombuffer(f.content, np.uint8), cv2.IMREAD_UNCHANGED).shape == (856, 1640, 4)

    jpg = cv2.imencode(".jpg", np.zeros((856, 1640, 3), np.uint8))[1].tobytes()
    client.post(f"/api/designs/{d['id']}/export", files={"file": ("x.jpg", jpg, "image/jpeg")}, data={"format": "jpeg"})
    assert [e["format"] for e in client.get("/api/exports", params={"design_id": d["id"]}).json()] == ["jpeg", "png"]

    # the design list now has a preview; deleting the design removes its export files
    assert client.get("/api/designs").json()[0]["thumb_url"]
    assert client.get(f"/api/designs/{d['id']}/thumb").headers["content-type"] == "image/jpeg"
    client.delete(f"/api/designs/{d['id']}")
    assert not list((client.lib.data_dir / "exports").iterdir())
    assert client.get("/api/exports").json() == []


def test_medium_asset_image(client):
    asset = make_asset(client)
    r = client.get(f"/api/assets/{asset['id']}/medium")
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert r.headers["content-type"] == "image/png" and img.shape[2] == 4


def test_store_roundtrip_without_http(tmp_path):
    db = Database(":memory:")
    store = DesignStore(db, tmp_path)
    d = store.create("ig", scene=Scene.model_validate(SCENE))
    assert store.get(d.id).scene.layers[0].text == "新品上市"
    with pytest.raises(ValueError):
        store.save_export(d.id, b"junk", "png")
    with pytest.raises(ValueError):
        store.save_export(d.id, png(1080, 1350), "gif")
    db.close()
