"""Run one segmentation model on photos and write visualizations.

    python -m backend.cli.segment tests/images/pudding.jpg --out outputs/
    python -m backend.cli.segment tests/images/pudding.jpg --seg fastsam
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

from .. import config
from ..inference import create_backend
from ..inference.labels import COCO80
from ..inference.segmenter import YoloSegmenter
from .viz import collect_images, draw_overlay, fit, imread, imwrite

SEG_MODELS = {"yolo": (config.YOLO_SEG, COCO80), "fastsam": (config.FASTSAM, None)}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Instance segmentation via InferenceBackend")
    parser.add_argument("inputs", nargs="+", help="image files or directories")
    parser.add_argument("--out", type=Path, default=config.OUTPUTS_DIR)
    parser.add_argument("--seg", default="yolo", choices=list(SEG_MODELS))
    parser.add_argument("--backend", default=config.BACKEND, choices=["onnx", "hailo"])
    parser.add_argument("--device", default=config.DEVICE, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model", type=Path, help="override the ONNX file")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--no-masks", action="store_true", help="skip per-instance mask PNGs")
    parser.add_argument("--max-side", type=int, default=2048, help="downscale the overlay image")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    base_cfg, names = SEG_MODELS[args.seg]
    overrides = {k: v for k, v in (("model_path", args.model), ("conf_threshold", args.conf),
                                   ("iou_threshold", args.iou)) if v is not None}
    cfg = dataclasses.replace(base_cfg, **overrides)
    segmenter = YoloSegmenter(create_backend(args.backend, args.device), cfg, names)
    segmenter.warmup()
    args.out.mkdir(parents=True, exist_ok=True)

    for path in collect_images(args.inputs):
        image = imread(path)
        result = segmenter(image)
        dets = result.detections
        stem = f"{path.stem}_{args.seg}"

        items = [(d.mask, d.bbox, f"{i}:{d.class_name or '?'} {d.score:.2f}") for i, d in enumerate(dets)]
        imwrite(args.out / f"{stem}_overlay.jpg", fit(draw_overlay(image, items), args.max_side))
        if not args.no_masks:
            mask_dir = args.out / f"{stem}_masks"
            mask_dir.mkdir(exist_ok=True)
            for old in mask_dir.glob("*.png"):
                old.unlink()
            for i, d in enumerate(dets):
                name = f"{i:02d}_{(d.class_name or 'object').replace(' ', '-')}_{d.score:.2f}.png"
                imwrite(mask_dir / name, d.mask.to_full(image.shape).astype(np.uint8) * 255)

        report = {
            "image": str(path),
            "image_size": list(image.shape[1::-1]),
            "backend": result.backend,
            "model": cfg.model_id,
            "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
            "detections": [
                {
                    "index": i,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "score": round(d.score, 4),
                    "bbox_xyxy": [round(float(v), 1) for v in d.bbox],
                    "mask_area": d.mask_area,
                }
                for i, d in enumerate(dets)
            ],
        }
        (args.out / f"{stem}_detections.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        t = result.timings_ms
        print(f"\n{path.name}  {image.shape[1]}x{image.shape[0]}  {cfg.model_id}  backend={result.backend}  "
              f"pre={t['preprocess']:.0f}ms infer={t['inference']:.0f}ms post={t['postprocess']:.0f}ms")
        if not dets:
            print("  (no detections)")
        for i, d in enumerate(dets):
            print(f"  [{i:2d}] {d.class_name or '?':<14} {d.score:.2f}  "
                  f"bbox={np.round(d.bbox).astype(int).tolist()}  area={d.mask_area}")
    print(f"\noutputs -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
