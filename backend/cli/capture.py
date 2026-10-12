"""Capture pipeline CLI: photo -> candidates -> (selected) transparent PNG cut-outs.

    python -m backend.cli.capture tests/images/pudding.jpg
    python -m backend.cli.capture tests/images/pudding.jpg --select 0,2,5
    python -m backend.cli.capture tests/images --out outputs/capture

Per photo, in <out>/<stem>/:
    candidates.jpg      numbered candidates (YOLO label, or '?' for FastSAM-only)
    candidates.json     candidates, rejected masks with reasons, timings
    assets/NN_<label>.png   transparent cut-outs
    assets_preview.jpg  all cut-outs on a checkerboard
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional

from .. import config
from ..inference import create_backend
from ..services.capture import CapturePipeline, Proposal
from .viz import collect_images, contact_sheet, draw_overlay, fit, imread, imwrite


def candidate_label(c) -> str:
    return f"{c.id}:{c.class_name} {c.score:.2f}" if c.class_name else f"{c.id}:?"


def proposal_json(path: Path, p: Proposal) -> dict:
    image_area = p.image_shape[0] * p.image_shape[1]
    return {
        "image": str(path),
        "image_size": [p.image_shape[1], p.image_shape[0]],
        "backend": p.backend,
        "raw_counts": p.raw_counts,
        "timings_ms": {k: round(v, 1) for k, v in p.timings_ms.items()},
        "candidates": [
            {
                "id": c.id,
                "class_name": c.class_name,
                "score": round(c.score, 4),
                "label_model": c.label_model,
                "mask_model": c.mask_model,
                "bbox_xyxy": list(c.bbox),
                "area": c.area,
                "area_ratio": round(c.area / image_area, 4),
            }
            for c in p.candidates
        ],
        "rejected": [
            {
                "source_model": r.source_model,
                "class_name": r.class_name,
                "score": round(r.score, 4),
                "bbox_xyxy": list(r.bbox),
                "reason": r.reason,
                "other_id": r.other_id,
            }
            for r in p.rejected
        ],
    }


def parse_ids(text: Optional[str]) -> Optional[List[int]]:
    if text is None or text.strip().lower() == "all":
        return None
    return [int(t) for t in text.replace(" ", "").split(",") if t]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture pipeline: candidates + transparent PNGs")
    parser.add_argument("inputs", nargs="+", help="image files or directories")
    parser.add_argument("--out", type=Path, default=config.OUTPUTS_DIR / "capture")
    parser.add_argument("--select", help="candidate ids to extract, e.g. 0,2,5 (default: all)")
    parser.add_argument("--no-assets", action="store_true", help="only propose candidates")
    parser.add_argument("--backend", default=config.BACKEND, choices=["onnx", "hailo"])
    parser.add_argument("--device", default=config.DEVICE, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-side", type=int, default=2048, help="downscale the candidates image")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    selected = parse_ids(args.select)
    pipeline = CapturePipeline(create_backend(args.backend, args.device))
    pipeline.warmup()

    summary = []
    for path in collect_images(args.inputs):
        out_dir = args.out / path.stem
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        t0 = time.perf_counter()
        image = imread(path)
        decode_ms = (time.perf_counter() - t0) * 1000
        proposal = pipeline.propose(image)
        cands = proposal.candidates

        items = [(c.mask, c.bbox, candidate_label(c)) for c in cands]
        imwrite(out_dir / "candidates.jpg", fit(draw_overlay(image, items), args.max_side))
        report = proposal_json(path, proposal)
        report["timings_ms"]["decode"] = round(decode_ms, 1)

        t = proposal.timings_ms
        print(f"\n{path.name}  {image.shape[1]}x{image.shape[0]}  backend={proposal.backend}")
        print(f"  propose {t['total']:.0f}ms (yolo {t['yolo_inference']:.0f}+{t['yolo_postprocess']:.0f}, "
              f"fastsam {t['fastsam_inference']:.0f}+{t['fastsam_postprocess']:.0f}, merge {t['merge']:.0f})"
              f"  decode {decode_ms:.0f}ms")
        print(f"  raw: yolo={proposal.raw_counts['yolo']} fastsam={proposal.raw_counts['fastsam']}"
              f"  -> {len(cands)} candidates "
              f"({sum(c.class_name is not None for c in cands)} labeled, "
              f"{sum(c.mask_model != c.label_model and c.label_model is not None for c in cands)} fused)")

        asset_ms: List[float] = []
        if not args.no_assets and cands:
            ids = selected if selected is not None else [c.id for c in cands]
            ids = [i for i in ids if i < len(cands)]
            assets = pipeline.extract(image, cands, ids)
            asset_dir = out_dir / "assets"
            asset_dir.mkdir()
            report["assets"] = []
            for a in assets:
                name = f"{a.candidate_id:02d}_{(a.class_name or 'object').replace(' ', '-')}.png"
                imwrite(asset_dir / name, a.rgba)
                asset_ms.append(a.elapsed_ms)
                report["assets"].append({"candidate_id": a.candidate_id, "file": f"assets/{name}",
                                         "bbox_xyxy": list(a.bbox), "origin": list(a.origin),
                                         "size": [a.rgba.shape[1], a.rgba.shape[0]],
                                         "elapsed_ms": round(a.elapsed_ms, 1)})
            labels = [f"{a.candidate_id}:{a.class_name or '?'}" for a in assets]
            imwrite(out_dir / "assets_preview.jpg", contact_sheet([a.rgba for a in assets], labels))
            print(f"  extract {len(assets)} assets: mean {statistics.mean(asset_ms):.0f}ms "
                  f"max {max(asset_ms):.0f}ms")

        (out_dir / "candidates.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
        summary.append({"image": path.name, "propose_ms": round(t["total"], 1),
                        "candidates": len(cands),
                        "asset_ms_max": round(max(asset_ms), 1) if asset_ms else None,
                        "asset_ms_mean": round(statistics.mean(asset_ms), 1) if asset_ms else None})

    if summary:
        propose = [s["propose_ms"] for s in summary]
        asset_max = [s["asset_ms_max"] for s in summary if s["asset_ms_max"] is not None]
        ok_p = max(propose) <= config.PROPOSE_BUDGET_MS
        ok_a = not asset_max or max(asset_max) <= config.EXTRACT_PER_ASSET_BUDGET_MS
        print(f"\n== {len(summary)} photos  backend={pipeline.backend.name}")
        print(f"   propose  mean {statistics.mean(propose):.0f}ms  max {max(propose):.0f}ms"
              f"  (budget {config.PROPOSE_BUDGET_MS}ms) {'OK' if ok_p else 'OVER'}")
        if asset_max:
            print(f"   per-asset max {max(asset_max):.0f}ms"
                  f"  (budget {config.EXTRACT_PER_ASSET_BUDGET_MS}ms) {'OK' if ok_a else 'OVER'}")
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"\noutputs -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
