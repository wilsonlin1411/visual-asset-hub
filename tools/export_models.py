"""Export ONNX models into models/. Run in the export-only env (has torch + ultralytics):

    conda run -n vah-export python tools/export_models.py

The runtime env never imports torch or ultralytics.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


# output name -> (ultralytics model class, weights file)
MODELS = {
    "yolov8s-seg.onnx": ("YOLO", "yolov8s-seg.pt"),
    "fastsam-s.onnx": ("FastSAM", "FastSAM-s.pt"),
}


def export_seg(target_name: str, imgsz: int, opset: int) -> Path:
    import ultralytics

    cls_name, weights = MODELS[target_name]
    MODELS_DIR.mkdir(exist_ok=True)
    # Downloads the .pt into models/ if missing (slow links: fetch it manually from
    # github.com/ultralytics/assets/releases and drop it in models/).
    model = getattr(ultralytics, cls_name)(str(MODELS_DIR / weights))
    # Static 1x3xSxS, no NMS in graph: post-processing stays on the host (Hailo parity).
    exported = Path(model.export(format="onnx", imgsz=imgsz, opset=opset,
                                 simplify=True, dynamic=False, nms=False))
    target = MODELS_DIR / target_name
    if exported.resolve() != target.resolve():
        shutil.move(str(exported), target)
    return target


def describe(path: Path) -> None:
    import onnx

    model = onnx.load(str(path))

    def dims(t):
        return [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]

    print(f"{path.name}  ({path.stat().st_size / 1e6:.1f} MB, opset {model.opset_import[0].version})")
    for t in model.graph.input:
        print(f"  input  {t.name}: {dims(t)}")
    for t in model.graph.output:
        print(f"  output {t.name}: {dims(t)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()
    for name in args.models:
        describe(export_seg(name, args.imgsz, args.opset))


if __name__ == "__main__":
    main()
