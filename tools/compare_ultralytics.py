"""Check our host-side post-processing against Ultralytics on the same ONNX models.

Both sides run the identical ONNX graph, so any difference comes from decode / NMS /
mask assembly. Run in the export env (needs ultralytics):

    conda run -n vah-export python tools/compare_ultralytics.py tests/images/pudding.jpg
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.cli.viz import collect_images, imread  # noqa: E402
from backend.inference import create_backend  # noqa: E402
from backend.inference.labels import COCO80  # noqa: E402
from backend.inference.postprocess import box_iou  # noqa: E402
from backend.inference.segmenter import YoloSegmenter  # noqa: E402


def full_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def compare_model(cfg, names, image_paths) -> bool:
    from ultralytics import YOLO

    ref_model = YOLO(str(cfg.model_path), task="segment")
    ours = YoloSegmenter(create_backend("onnx", "cpu"), cfg, names)
    label = (lambda c: names[c]) if names else (lambda c: "object")

    all_ok = True
    for path in image_paths:
        image = imread(path)
        ref = ref_model.predict(image, imgsz=cfg.input_size, conf=cfg.conf_threshold,
                                iou=cfg.iou_threshold, max_det=cfg.max_det,
                                retina_masks=True, verbose=False)[0]
        ref_boxes = ref.boxes.xyxy.cpu().numpy()
        ref_cls = ref.boxes.cls.cpu().numpy().astype(int)
        ref_conf = ref.boxes.conf.cpu().numpy()
        ref_masks = ref.masks.data.cpu().numpy() > 0.5 if ref.masks is not None else None

        dets = ours(image).detections
        our_boxes = np.array([d.bbox for d in dets]).reshape(-1, 4)
        our_cls = np.array([d.class_id or 0 for d in dets])

        print(f"\n[{cfg.model_id}] {path.name}: ultralytics={len(ref_boxes)}  ours={len(dets)}")
        box_ious, mask_ious, score_diffs = [], [], []
        for i in range(len(ref_boxes)):
            same = np.flatnonzero(our_cls == ref_cls[i]) if len(dets) else np.zeros(0, int)
            if same.size == 0:
                print(f"  MISSING {label(ref_cls[i])} {ref_conf[i]:.2f}")
                all_ok = False
                continue
            ious = box_iou(ref_boxes[i], our_boxes[same])
            j = same[ious.argmax()]
            box_ious.append(ious.max())
            score_diffs.append(abs(dets[j].score - ref_conf[i]))
            mask_ious.append(full_iou(ref_masks[i], dets[j].mask.to_full(image.shape)))
            if names:
                print(f"  {label(ref_cls[i]):<14} score ref={ref_conf[i]:.3f} ours={dets[j].score:.3f}  "
                      f"boxIoU={box_ious[-1]:.4f}  maskIoU={mask_ious[-1]:.4f}")
        if box_ious:
            ok = (len(ref_boxes) == len(dets) and min(box_ious) > 0.95 and min(mask_ious) > 0.9)
            all_ok &= ok
            print(f"  -> min boxIoU={min(box_ious):.4f}  min maskIoU={min(mask_ious):.4f}  "
                  f"mean maskIoU={np.mean(mask_ious):.4f}  max |score diff|={max(score_diffs):.4f}  "
                  f"{'OK' if ok else 'MISMATCH'}")
        elif len(dets):
            print("  -> ultralytics found nothing but we did: MISMATCH")
            all_ok = False
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--models", nargs="+", default=["yolo", "fastsam"], choices=["yolo", "fastsam"])
    # Ultralytics retina masks are full-frame float tensors; cap FastSAM's count to bound memory.
    parser.add_argument("--fastsam-max-det", type=int, default=30)
    args = parser.parse_args()

    paths = collect_images(args.inputs)
    ok = True
    if "yolo" in args.models:
        ok &= compare_model(config.YOLO_SEG, COCO80, paths)
    if "fastsam" in args.models:
        ok &= compare_model(dataclasses.replace(config.FASTSAM, max_det=args.fastsam_max_det), None, paths)
    print("\nALL MATCH" if ok else "\nDIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
