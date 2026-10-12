import numpy as np

from backend.inference.masks import RegionMask, intersection, mask_iou


def _full(shape, y0, y1, x0, x1):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_region_mask_roundtrip_and_tight():
    full = _full((50, 60), 10, 20, 5, 25)
    rm = RegionMask.from_full(full)
    assert rm.box == (5, 10, 25, 20)
    assert rm.area == 200
    np.testing.assert_array_equal(rm.to_full((50, 60)), full)


def test_iou_matches_full_frame_computation():
    rng = np.random.default_rng(1)
    shape = (80, 90)
    for _ in range(20):
        a = rng.random(shape) > 0.5
        b = rng.random(shape) > 0.5
        a[:, :rng.integers(0, 60)] = False
        b[rng.integers(30, 80):, :] = False
        ra, rb = RegionMask.from_full(a), RegionMask.from_full(b)
        expected = (a & b).sum() / max((a | b).sum(), 1)
        assert abs(mask_iou(ra, rb) - expected) < 1e-9
        assert intersection(ra, rb) == (a & b).sum()


def test_disjoint_and_empty():
    a = RegionMask.from_full(_full((40, 40), 0, 10, 0, 10))
    b = RegionMask.from_full(_full((40, 40), 20, 30, 20, 30))
    assert mask_iou(a, b) == 0.0
    empty = RegionMask(np.zeros((0, 0), bool), 0, 0)
    assert mask_iou(a, empty) == 0.0
    assert RegionMask(np.zeros((5, 5), bool), 3, 3).tight().area == 0
