"""Capture pipeline: photo -> candidates (propose) -> user picks -> transparent cut-outs (extract).

The two steps map to the API: GET /photos/{id}/candidates and POST /assets/confirm.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .. import config
from ..inference import InferenceBackend
from ..inference.labels import COCO80
from ..inference.segmenter import YoloSegmenter
from .export_png import make_cutout
from .merge import Candidate, Rejected, merge_detections
from .refine import GuidedFilterRefiner, Refiner, edge_band_px


@dataclass
class Proposal:
    candidates: List[Candidate]
    rejected: List[Rejected]
    image_shape: Tuple[int, int]
    backend: str
    timings_ms: Dict[str, float] = field(default_factory=dict)
    raw_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ExtractedAsset:
    candidate_id: int
    class_id: Optional[int]
    class_name: Optional[str]
    score: float
    label_model: Optional[str]
    mask_model: str
    rgba: np.ndarray                 # BGRA uint8
    bbox: Tuple[int, int, int, int]  # tight object box in the original photo
    origin: Tuple[int, int]          # PNG top-left in original coords
    elapsed_ms: float


class CapturePipeline:
    def __init__(self, backend: InferenceBackend,
                 yolo_cfg: config.SegmentationConfig = config.YOLO_SEG,
                 fastsam_cfg: config.SegmentationConfig = config.FASTSAM,
                 merge_cfg: config.MergeConfig = config.MERGE,
                 refine_cfg: config.RefineConfig = config.REFINE,
                 refiner: Optional[Refiner] = None):
        self.backend = backend
        self.yolo = YoloSegmenter(backend, yolo_cfg, COCO80)
        self.fastsam = YoloSegmenter(backend, fastsam_cfg, None)
        self.merge_cfg = merge_cfg
        self.refine_cfg = refine_cfg
        self.refiner = refiner or GuidedFilterRefiner(refine_cfg)

    def warmup(self) -> None:
        self.yolo.warmup()
        self.fastsam.warmup()

    def propose(self, image_bgr: np.ndarray) -> Proposal:
        t0 = time.perf_counter()
        yolo = self.yolo(image_bgr)
        fastsam = self.fastsam(image_bgr)
        t1 = time.perf_counter()
        merged = merge_detections(yolo.detections, fastsam.detections, image_bgr.shape[:2], self.merge_cfg)
        t2 = time.perf_counter()

        timings = {f"yolo_{k}": v for k, v in yolo.timings_ms.items()}
        timings.update({f"fastsam_{k}": v for k, v in fastsam.timings_ms.items()})
        timings["merge"] = (t2 - t1) * 1000
        timings["total"] = (t2 - t0) * 1000
        return Proposal(
            candidates=merged.candidates,
            rejected=merged.rejected,
            image_shape=image_bgr.shape[:2],
            backend=self.backend.name,
            timings_ms=timings,
            raw_counts={"yolo": len(yolo.detections), "fastsam": len(fastsam.detections)},
        )

    def extract(self, image_bgr: np.ndarray, candidates: Iterable[Candidate],
                selected_ids: Optional[Iterable[int]] = None) -> List[ExtractedAsset]:
        """Refine and cut out the selected candidates (all when `selected_ids` is None)."""
        return extract_assets(image_bgr, candidates, selected_ids, self.refiner, self.refine_cfg)


def extract_assets(image_bgr: np.ndarray, candidates: Iterable[Candidate],
                   selected_ids: Optional[Iterable[int]] = None,
                   refiner: Optional[Refiner] = None,
                   refine_cfg: config.RefineConfig = config.REFINE) -> List[ExtractedAsset]:
    """Extraction needs no segmentation model, so confirming stored candidates stays cheap."""
    refiner = refiner or GuidedFilterRefiner(refine_cfg)
    by_id = {c.id: c for c in candidates}
    ids = list(by_id) if selected_ids is None else list(selected_ids)
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise KeyError(f"unknown candidate ids: {unknown}")

    # Edge colours are re-estimated from solid pixels within about one edge band.
    sigma = float(edge_band_px(image_bgr.shape[:2], refine_cfg))
    assets = []
    for cid in ids:
        c = by_id[cid]
        t0 = time.perf_counter()
        matte = refiner.refine(image_bgr, c.mask)
        cut = make_cutout(image_bgr, matte, refine_cfg.png_padding, decontaminate_sigma=sigma)
        assets.append(ExtractedAsset(
            candidate_id=c.id, class_id=c.class_id, class_name=c.class_name, score=c.score,
            label_model=c.label_model, mask_model=c.mask_model,
            rgba=cut.rgba, bbox=cut.bbox, origin=cut.origin,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        ))
    return assets
