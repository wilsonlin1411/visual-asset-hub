"""Instance segmentation on top of any InferenceBackend (YOLOv8-seg now, FastSAM later)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..config import SegmentationConfig
from . import postprocess as pp
from .base import Detection, InferenceBackend, ModelSpec


@dataclass
class SegmentationResult:
    detections: List[Detection]
    backend: str
    timings_ms: Dict[str, float] = field(default_factory=dict)


class YoloSegmenter:
    def __init__(self, backend: InferenceBackend, cfg: SegmentationConfig,
                 class_names: Optional[Sequence[str]]):
        """`class_names=None` for class-agnostic models (FastSAM): detections get no class."""
        self.backend = backend
        self.cfg = cfg
        self.class_names = list(class_names) if class_names is not None else None
        self.input_shape = (cfg.input_size, cfg.input_size)
        backend.load_model(ModelSpec(cfg.model_id, cfg.model_path, self.input_shape))

    def warmup(self) -> None:
        """First run includes lazy allocation (and CUDA kernel setup); keep it out of timings."""
        self.backend.infer(self.cfg.model_id, np.zeros((*self.input_shape, 3), dtype=np.uint8))

    def __call__(self, image_bgr: np.ndarray) -> SegmentationResult:
        cfg = self.cfg
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        model_input, info = pp.letterbox(rgb, self.input_shape)

        t1 = time.perf_counter()
        outputs = self.backend.infer(cfg.model_id, model_input)

        t2 = time.perf_counter()
        head = pp.decode_yolov8_seg(outputs)
        names = self.class_names
        if names is not None and head.scores.shape[1] != len(names):
            raise ValueError(f"model has {head.scores.shape[1]} classes, got {len(names)} names")
        sel = pp.select_detections(head, cfg.conf_threshold, cfg.iou_threshold, cfg.max_det)
        boxes = pp.scale_boxes(sel.boxes, info)
        masks = pp.process_masks(head.proto, sel.coeffs, boxes, info, cfg.mask_threshold)
        t3 = time.perf_counter()

        detections = [
            Detection(
                bbox=boxes[i],
                class_id=int(sel.class_ids[i]) if names is not None else None,
                class_name=names[int(sel.class_ids[i])] if names is not None else None,
                score=float(sel.scores[i]),
                mask=masks[i],
                source_model=cfg.model_id,
            )
            for i in range(len(boxes))
        ]
        timings = {
            "preprocess": (t1 - t0) * 1000,
            "inference": (t2 - t1) * 1000,
            "postprocess": (t3 - t2) * 1000,
        }
        return SegmentationResult(detections, self.backend.name, timings)
