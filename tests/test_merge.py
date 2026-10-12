import numpy as np

from backend.config import MergeConfig
from backend.inference import Detection, RegionMask
from backend.services.merge import clean_mask, merge_detections

SHAPE = (200, 200)
CFG = MergeConfig(fuse_iou=0.7, dedup_iou=0.7, min_area_ratio=0.002, max_area_ratio=0.9,
                  min_component_ratio=0.02)


def rect(y0, y1, x0, x1, shape=SHAPE):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def det(full, score, cls=None, model="yolov8s-seg"):
    rm = RegionMask.from_full(full)
    x0, y0, x1, y1 = rm.box
    return Detection(bbox=np.float32([x0, y0, x1, y1]), class_id=None if cls is None else 0,
                     class_name=cls, score=score, mask=rm, source_model=model)


def fs(full, score=0.9):
    return det(full, score, None, "fastsam-s")


def test_fuse_takes_yolo_label_and_fastsam_mask():
    y = det(rect(50, 100, 50, 100), 0.8, "cup")
    f = fs(rect(52, 100, 50, 102))  # IoU ~0.92 with y
    res = merge_detections([y], [f], SHAPE, CFG)
    assert len(res.candidates) == 1
    c = res.candidates[0]
    assert c.class_name == "cup" and c.label_model == "yolov8s-seg" and c.mask_model == "fastsam-s"
    assert c.bbox == (50, 52, 102, 100)
    assert [(r.reason, r.other_id) for r in res.rejected] == [("fused", 0)]


def test_low_overlap_keeps_both_and_part_inside_object_is_not_a_duplicate():
    y = det(rect(20, 180, 20, 180), 0.8, "cake")
    part = fs(rect(60, 90, 60, 90))  # fully inside, IoU ~0.035
    res = merge_detections([y], [part], SHAPE, CFG)
    assert [c.class_name for c in res.candidates] == ["cake", None]
    assert res.candidates[0].mask_model == "yolov8s-seg"
    assert res.rejected == []


def test_best_fastsam_match_is_fused_and_runner_up_deduplicated():
    y = det(rect(50, 100, 50, 100), 0.8, "bowl")
    best = fs(rect(50, 100, 50, 101), 0.5)
    other = fs(rect(50, 100, 48, 100), 0.95)
    res = merge_detections([y], [best, other], SHAPE, CFG)
    assert len(res.candidates) == 1
    reasons = sorted(r.reason for r in res.rejected)
    assert reasons == ["duplicate", "fused"]


def test_yolo_duplicates_of_other_classes_are_dropped():
    a = det(rect(50, 100, 50, 100), 0.9, "cup")
    b = det(rect(50, 100, 50, 98), 0.6, "bowl")  # same object, other class
    res = merge_detections([b, a], [], SHAPE, CFG)
    assert [c.class_name for c in res.candidates] == ["cup"]
    assert res.rejected[0].reason == "duplicate" and res.rejected[0].other_id == 0


def test_area_filters_and_ordering():
    tiny = fs(rect(0, 5, 0, 5))           # 25 px < 0.2% of 40000
    huge = fs(rect(0, 200, 0, 195))       # 97.5% > 90%
    small = fs(rect(150, 170, 150, 170))  # 400 px
    big = fs(rect(10, 60, 10, 60))        # 2500 px
    labeled = det(rect(100, 140, 10, 40), 0.3, "book")
    res = merge_detections([labeled], [tiny, huge, small, big], SHAPE, CFG)
    # labeled first, then unlabeled by area (largest first)
    assert [(c.class_name, c.area) for c in res.candidates] == [("book", 1200), (None, 2500), (None, 400)]
    assert sorted(r.reason for r in res.rejected) == ["too_large", "too_small"]
    assert [c.id for c in res.candidates] == [0, 1, 2]


def test_clean_mask_removes_specks_keeps_holes():
    m = rect(20, 120, 20, 120)
    m[60:80, 60:80] = False   # hole: must stay
    m[150:152, 150:152] = True  # 4 px speck: removed
    cleaned = clean_mask(RegionMask.from_full(m), 0.02)
    full = cleaned.to_full(SHAPE)
    assert not full[150:152, 150:152].any()
    assert not full[60:80, 60:80].any()
    assert cleaned.box == (20, 20, 120, 120)
