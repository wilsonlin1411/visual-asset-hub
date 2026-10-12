"""Combine YOLO-seg detections and FastSAM masks into one candidate list.

1. Clean every mask (drop specks) and filter by area.
2. YOLO detections, best score first: drop duplicates of an already kept YOLO candidate;
   otherwise look for the FastSAM mask overlapping it best. If that IoU >= fuse_iou the
   candidate keeps the YOLO label but takes the FastSAM mask (better edges).
3. Unused FastSAM masks, best score first: drop duplicates of anything kept, else keep
   them unlabeled (CLIP zero-shot labels them in stage 4).
Containment is deliberately not a duplicate: a cat printed on a phone case is a useful
asset of its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import MergeConfig
from ..inference import Detection, RegionMask
from ..inference.masks import mask_iou


@dataclass
class Candidate:
    id: int
    mask: RegionMask
    score: float
    class_id: Optional[int]
    class_name: Optional[str]
    label_model: Optional[str]  # model that supplied the class; None = unlabeled
    mask_model: str             # model whose mask is used
    area: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.mask.box


@dataclass
class Rejected:
    source_model: str
    class_name: Optional[str]
    score: float
    bbox: Tuple[int, int, int, int]
    reason: str  # too_small | too_large | empty | duplicate | fused
    other_id: Optional[int] = None  # candidate it duplicated / was fused into


@dataclass
class MergeResult:
    candidates: List[Candidate]
    rejected: List[Rejected]


class _Item:
    __slots__ = ("det", "mask", "area", "mask_model", "final_id")

    def __init__(self, det: Detection, mask: RegionMask):
        self.det = det
        self.mask = mask
        self.area = mask.area
        self.mask_model = det.source_model
        self.final_id: Optional[int] = None


def clean_mask(mask: RegionMask, min_component_ratio: float) -> RegionMask:
    """Remove connected components smaller than `min_component_ratio` of the mask. Holes are kept."""
    if mask.data.size == 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.data.astype(np.uint8), connectivity=8)
    if n <= 2:  # background + at most one component
        return mask.tight()
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= min_component_ratio * areas.sum()) + 1
    return RegionMask(np.isin(labels, keep), mask.x0, mask.y0).tight()


def _iou(a: _Item, b: _Item, threshold: float) -> float:
    # IoU <= min(area)/max(area); skip the pixel work when it cannot reach the threshold.
    if min(a.area, b.area) < threshold * max(a.area, b.area):
        return 0.0
    return mask_iou(a.mask, b.mask, a.area, b.area)


def merge_detections(yolo: Sequence[Detection], fastsam: Sequence[Detection],
                     image_shape: Tuple[int, int], cfg: MergeConfig) -> MergeResult:
    image_area = image_shape[0] * image_shape[1]
    rejected: List[Tuple[Detection, str, Optional[_Item]]] = []

    def prefilter(dets: Sequence[Detection]) -> List[_Item]:
        items = []
        for det in sorted(dets, key=lambda d: -d.score):
            item = _Item(det, clean_mask(det.mask, cfg.min_component_ratio))
            ratio = item.area / image_area
            if item.area == 0:
                rejected.append((det, "empty", None))
            elif ratio < cfg.min_area_ratio:
                rejected.append((det, "too_small", None))
            elif ratio > cfg.max_area_ratio:
                rejected.append((det, "too_large", None))
            else:
                items.append(item)
        return items

    yolo_items = prefilter(yolo)
    fs_items = prefilter(fastsam)
    labeled: List[_Item] = []
    fs_used = set()

    for y in yolo_items:
        dup = next((k for k in labeled if _iou(y, k, cfg.dedup_iou) >= cfg.dedup_iou), None)
        if dup is not None:
            rejected.append((y.det, "duplicate", dup))
            continue
        best, best_iou = None, cfg.fuse_iou
        for j, f in enumerate(fs_items):
            if j in fs_used:
                continue
            iou = _iou(y, f, cfg.fuse_iou)
            if iou >= best_iou:
                best, best_iou = j, iou
        if best is not None:
            f = fs_items[best]
            fs_used.add(best)
            y.mask, y.area, y.mask_model = f.mask, f.area, f.mask_model
            rejected.append((f.det, "fused", y))
        labeled.append(y)

    unlabeled: List[_Item] = []
    for j, f in enumerate(fs_items):
        if j in fs_used:
            continue
        dup = next((k for k in (*labeled, *unlabeled) if _iou(f, k, cfg.dedup_iou) >= cfg.dedup_iou), None)
        if dup is not None:
            rejected.append((f.det, "duplicate", dup))
            continue
        unlabeled.append(f)

    # Labeled first by score (already sorted), then unlabeled by size: biggest objects first.
    unlabeled.sort(key=lambda it: -it.area)
    candidates = []
    for i, it in enumerate((*labeled, *unlabeled)):
        it.final_id = i
        candidates.append(Candidate(
            id=i,
            mask=it.mask,
            score=it.det.score,
            class_id=it.det.class_id,
            class_name=it.det.class_name,
            label_model=it.det.source_model if it.det.class_name is not None else None,
            mask_model=it.mask_model,
            area=it.area,
        ))

    rejects = [
        Rejected(
            source_model=det.source_model,
            class_name=det.class_name,
            score=det.score,
            bbox=tuple(int(v) for v in np.round(det.bbox)),
            reason=reason,
            other_id=other.final_id if other is not None else None,
        )
        for det, reason, other in rejected
    ]
    return MergeResult(candidates, rejects)
