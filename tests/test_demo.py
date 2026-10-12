import dataclasses
import time

import cv2
import pytest
from fastapi.testclient import TestClient

from backend import config
from backend.api.app import create_app
from backend.api.demo import SessionManager, SharedModels, create_demo_app
from backend.services.library import AssetLibrary
from fakes import SPACE, FakeAnalyzer, FakeTextEncoder
from test_library import FakePipeline, scene

A, B, C, D = (ch * 32 for ch in "abcd")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, minutes):
        self.t += minutes * 60


@pytest.fixture
def seed(tmp_path):
    """Demo library: one photo, one confirmed asset."""
    lib = AssetLibrary(tmp_path / "seed", pipeline_factory=FakePipeline, analyzer_factory=FakeAnalyzer)
    src = tmp_path / "seed.png"
    cv2.imwrite(str(src), scene())
    pid = lib.add_photo(src).photo.id
    lib.confirm(pid, [0])
    lib.close()
    return tmp_path / "seed"


def demo_client(tmp_path, seed, **cfg):
    clock = Clock()
    cfg = dataclasses.replace(config.DEMO, sessions_dir=tmp_path / "sessions", **cfg)
    models = SharedModels(FakePipeline, FakeAnalyzer, FakeTextEncoder, SPACE)
    manager = SessionManager(seed, models, cfg, clock)
    client = TestClient(create_demo_app(manager, serve_frontend=False, warm_up=False))
    client.manager, client.clock = manager, clock
    return client


@pytest.fixture
def demo(tmp_path, seed):
    with demo_client(tmp_path, seed, max_uploads=1, max_upload_mb=1, max_sessions=2) as c:
        yield c


def get(client, sid, url, **kw):
    return client.get(url, headers={"X-VAH-Session": sid}, **kw)


def upload(client, sid, image, name="new.png"):
    data = image if isinstance(image, bytes) else cv2.imencode(".png", image)[1].tobytes()
    return client.post("/api/photos", headers={"X-VAH-Session": sid}, files={"file": (name, data, "image/png")})


def wait_ready(client, sid, photo_id, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        p = get(client, sid, f"/api/photos/{photo_id}").json()
        if p["status"] != "processing":
            return p
        time.sleep(0.05)
    raise AssertionError("photo still processing")


def test_each_session_has_its_own_copy_of_the_demo_library(demo, tmp_path):
    assert [p["original_name"] for p in get(demo, A, "/api/photos").json()] == ["seed.png"]
    [seed_asset] = get(demo, B, "/api/assets").json()

    r = upload(demo, A, cv2.flip(scene(), 1))
    assert r.status_code == 202
    pid = r.json()["photo"]["id"]
    assert wait_ready(demo, A, pid)["status"] == "ready"
    assert len(get(demo, A, "/api/photos").json()) == 2
    assert len(get(demo, B, "/api/photos").json()) == 1

    assert demo.delete(f"/api/assets/{seed_asset['id']}", headers={"X-VAH-Session": A}).status_code == 200
    assert get(demo, A, "/api/assets").json() == []
    assert [a["id"] for a in get(demo, B, "/api/assets").json()] == [seed_asset["id"]]

    # URLs the browser loads by itself carry the id as ?s=
    assert demo.get(f"{seed_asset['thumb_url']}?s={B}").status_code == 200
    assert demo.get(f"/api/photos/{pid}/display?s={A}").status_code == 200
    assert demo.get(f"/api/photos/{pid}/display?s={B}").status_code == 404
    assert (tmp_path / "sessions" / A / config.DB_FILENAME).is_file()


def test_requests_need_a_valid_session_id(demo):
    assert demo.get("/api/photos").status_code == 400
    assert demo.get("/api/photos", headers={"X-VAH-Session": "../../etc"}).status_code == 400
    assert demo.get("/api/photos?s=ABC").status_code == 400
    info = demo.get("/api/demo").json()
    assert info["enabled"] and info["seed_photos"] == 1 and info["max_uploads"] == 1
    assert info["source_url"].startswith("https://")
    assert len(demo.manager) == 0


def test_upload_limits(demo):
    too_big = b"\xff\xd8" + b"0" * 2**20
    assert upload(demo, A, too_big, "big.jpg").status_code == 413
    pid = upload(demo, A, cv2.flip(scene(), 1)).json()["photo"]["id"]
    wait_ready(demo, A, pid)
    r = upload(demo, A, cv2.flip(scene(), 0))
    assert r.status_code == 429 and "上限" in r.json()["detail"]
    assert upload(demo, B, cv2.flip(scene(), 0)).status_code == 202   # the limit is per session


def test_idle_sessions_are_deleted(demo, tmp_path):
    get(demo, A, "/api/photos")
    demo.clock.advance(30)
    get(demo, B, "/api/photos")
    get(demo, C, "/api/photos")                     # full (2): A, idle for 30 min, makes room
    assert not (tmp_path / "sessions" / A).exists()
    r = get(demo, D, "/api/photos")                 # B and C are both in use
    assert r.status_code == 503 and "已滿" in r.json()["detail"]

    demo.clock.advance(59)
    assert demo.manager.sweep() == 0
    demo.clock.advance(1)
    assert demo.manager.sweep() == 2 and len(demo.manager) == 0
    assert not any((tmp_path / "sessions").iterdir())
    assert len(get(demo, A, "/api/photos").json()) == 1   # a returning visitor starts over


def test_new_sessions_are_rate_limited_per_client(tmp_path, seed):
    with demo_client(tmp_path, seed, new_sessions_per_ip=2) as c:
        assert get(c, A, "/api/photos").status_code == 200
        assert get(c, B, "/api/photos").status_code == 200
        assert get(c, C, "/api/photos").status_code == 429
        assert get(c, A, "/api/photos").status_code == 200       # existing sessions keep working
        other = {"X-VAH-Session": C, "X-Forwarded-For": "203.0.113.9, 10.0.0.1"}
        assert c.get("/api/photos", headers=other).status_code == 200


def test_local_app_reports_demo_disabled(tmp_path):
    lib = AssetLibrary(tmp_path / "data", pipeline_factory=FakePipeline, analyzer_factory=FakeAnalyzer)
    with TestClient(create_app(lib, serve_frontend=False)) as c:
        assert c.get("/api/demo").json() == {"enabled": False, "seed_photos": None, "idle_minutes": None,
                                             "max_uploads": None, "source_url": None}
    lib.close()
