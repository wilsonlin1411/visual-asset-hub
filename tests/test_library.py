import cv2
import numpy as np
import pytest

from backend.config import EMBEDDING_DIM
from backend.db.vector_index import EmbeddingSpace
from backend.imageio import imread
from backend.inference import RegionMask
from backend.services.capture import Proposal
from backend.services.library import AssetLibrary, decode_mask, encode_mask
from backend.services.merge import Candidate


def scene():
    img = np.full((300, 400, 3), (0, 150, 0), np.uint8)
    cv2.circle(img, (120, 150), 60, (0, 0, 220), -1)
    cv2.rectangle(img, (250, 80), (350, 220), (200, 50, 0), -1)
    return img


class FakePipeline:
    """Stands in for the ONNX models: two fixed candidates."""

    def __init__(self):
        self.calls = 0

    def propose(self, image):
        self.calls += 1
        h, w = image.shape[:2]
        disc = np.zeros((h, w), np.uint8)
        cv2.circle(disc, (120, 150), 58, 1, -1)
        box = np.zeros((h, w), bool)
        box[82:218, 252:348] = True
        cands = [
            Candidate(0, RegionMask.from_full(disc.astype(bool)), 0.9, 41, "cup", "yolov8s-seg", "fastsam-s",
                      int(disc.sum())),
            Candidate(1, RegionMask.from_full(box), 0.6, None, None, None, "fastsam-s", int(box.sum())),
        ]
        return Proposal(cands, [], (h, w), "fake")


def no_pipeline():
    raise AssertionError("segmentation models must not be loaded for this operation")


@pytest.fixture
def photo_file(tmp_path):
    p = tmp_path / "照片.png"  # non-ASCII on purpose (Windows path handling)
    cv2.imencode(".png", scene())[1].tofile(str(p))
    return p


@pytest.fixture
def lib(tmp_path):
    fake = FakePipeline()
    lib = AssetLibrary(tmp_path / "data", pipeline_factory=lambda: fake, analyzer_factory=lambda: None)
    lib.fake = fake
    yield lib
    lib.close()


def test_mask_png_roundtrip():
    m = np.zeros((37, 53), bool)
    m[5:30, 7:40] = True
    m[10:12, 10:12] = False
    rm = RegionMask(m, 11, 22)
    back = decode_mask(encode_mask(rm), rm.box)
    assert back.box == rm.box and np.array_equal(back.data, rm.data)


def test_add_photo_stores_photo_and_candidates_and_dedupes(lib, photo_file):
    res = lib.add_photo(photo_file)
    assert res.created and res.photo.original_name == "照片.png"
    assert (res.photo.width, res.photo.height) == (400, 300)
    assert lib.path(res.photo.path).read_bytes() == photo_file.read_bytes()

    cands = lib.candidates(res.photo.id)
    assert [(c.idx, c.class_name, c.status) for c in cands] == [(0, "cup", "pending"), (1, None, "pending")]
    original = res.proposal.candidates[0].mask
    stored = lib.load_candidate_mask(cands[0])
    assert stored.box == original.box and np.array_equal(stored.data, original.data)

    again = lib.add_photo(photo_file)
    assert not again.created and again.photo.id == res.photo.id and lib.fake.calls == 1


def test_confirm_creates_assets_idempotently(lib, photo_file):
    pid = lib.add_photo(photo_file).photo.id
    [asset] = lib.confirm(pid, [0])
    assert asset.category == "cup" and asset.label_model == "yolov8s-seg" and asset.mask_model == "fastsam-s"
    rgba = imread(lib.path(asset.png_path), cv2.IMREAD_UNCHANGED)
    assert rgba.shape == (asset.height, asset.width, 4)
    x0, y0, x1, y1 = asset.bbox
    assert abs(x0 - 60) <= 3 and abs(x1 - 181) <= 3  # refined to the drawn disc (r=60), not the r=58 mask
    assert [c.status for c in lib.candidates(pid)] == ["confirmed", "pending"]

    assert [a.id for a in lib.confirm(pid, [0])] == [asset.id]  # no duplicate
    assert len(list(lib.path("assets").iterdir())) == 1

    both = lib.confirm(pid)  # default: remaining pending (idx 1) plus nothing else
    assert [a.id for a in both] == [asset.id + 1]
    assert lib.stats()["assets"] == 2


def test_confirm_in_new_process_without_models(tmp_path, photo_file):
    fake = FakePipeline()
    first = AssetLibrary(tmp_path / "data", pipeline_factory=lambda: fake, analyzer_factory=lambda: None)
    pid = first.add_photo(photo_file).photo.id
    first.close()

    second = AssetLibrary(tmp_path / "data", pipeline_factory=no_pipeline, analyzer_factory=lambda: None)
    assets = second.confirm(pid, [0, 1])
    assert len(assets) == 2 and all(second.path(a.png_path).is_file() for a in assets)
    second.close()


def test_discard_and_unknown_candidates(lib, photo_file):
    pid = lib.add_photo(photo_file).photo.id
    assert lib.discard(pid, [1]) == 1
    assert [a.category for a in lib.confirm(pid)] == ["cup"]  # discarded one skipped
    with pytest.raises(KeyError):
        lib.confirm(pid, [5])
    # explicitly selecting a discarded candidate brings it back
    [revived] = lib.confirm(pid, [1])
    assert revived.category is None and lib.stats()["assets"] == 2


def test_delete_asset_frees_candidate_and_vectors(lib, photo_file):
    pid = lib.add_photo(photo_file).photo.id
    [asset] = lib.confirm(pid, [0])
    space = EmbeddingSpace("clip-vit-b32", "onnx-cpu")
    v = np.random.default_rng(0).normal(size=EMBEDDING_DIM)
    lib.index.upsert(asset.id, space, v)
    assert lib.index.search(v, space, k=1)[0].asset_id == asset.id

    assert lib.delete_asset(asset.id)
    assert not lib.path(asset.png_path).exists()
    assert lib.candidates(pid)[0].status == "pending"
    assert lib.index.search(v, space, k=1) == [] and lib.index.count() == 0


def test_delete_photo_removes_rows_and_files(lib, photo_file):
    pid = lib.add_photo(photo_file).photo.id
    lib.confirm(pid)
    assert lib.delete_photo(pid)
    assert not any(p.is_file() for p in lib.data_dir.rglob("*") if p.suffix in (".png", ".jpg"))
    s = lib.stats()
    assert (s["photos"], s["assets"], s["candidates_pending"], s["candidates_confirmed"]) == (0, 0, 0, 0)
    assert not lib.delete_photo(pid)
