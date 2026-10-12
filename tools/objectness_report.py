"""Objectness calibration report: distribution, hidden counts per threshold, and a contact
sheet of candidates near the threshold (to pick the threshold by eye).

    conda run -n vah python tools/objectness_report.py --out outputs/objectness
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.cli.viz import contact_sheet, imwrite  # noqa: E402
from backend.db import repo  # noqa: E402
from backend.imageio import imread  # noqa: E402
from backend.services.library import AssetLibrary  # noqa: E402


def preview(image, mask, size=320):
    """Candidate crop with off-mask pixels dimmed, as BGRA for the contact sheet."""
    crop = image[mask.y0:mask.y1, mask.x0:mask.x1].copy()
    alpha = np.where(mask.data, 255, 60).astype(np.uint8)
    bgra = np.dstack([crop, alpha])
    s = size / max(bgra.shape[:2])
    return cv2.resize(bgra, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/objectness"))
    parser.add_argument("--low", type=float, default=0.25)
    parser.add_argument("--high", type=float, default=0.45)
    parser.add_argument("--include-labeled", action="store_true", help="also YOLO-labelled candidates")
    parser.add_argument("--columns", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lib = AssetLibrary()
    threshold = lib.objectness_threshold
    rows = []
    for pid in repo.photo_ids_with_status(lib.db, "ready"):
        rows += [r for r in repo.list_candidates(lib.db, pid)
                 if (args.include_labeled or r.class_name is None) and r.objectness is not None]
    obj = np.array([r.objectness for r in rows])
    print(f"{len(rows)} unlabeled candidates (YOLO-labelled ones are never hidden)")
    hist, edges = np.histogram(obj, bins=10, range=(0, 1))
    for n, lo in zip(hist, edges):
        print(f"  {lo:.1f}-{lo + 0.1:.1f} {'#' * int(n // 2):<60} {n}")
    for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
        print(f"  threshold {t:.2f}: hides {np.sum(obj < t):3d} ({np.mean(obj < t):.0%})")

    near = sorted([r for r in rows if args.low <= r.objectness < args.high], key=lambda r: r.objectness)
    images = {}
    tiles, labels = [], []
    for r in near:
        if r.photo_id not in images:
            images[r.photo_id] = imread(lib.path(lib.get_photo(r.photo_id).path))
        tiles.append(preview(images[r.photo_id], lib.load_candidate_mask(r)))
        state = "HIDE" if r.hidden(threshold) else "show"
        labels.append(f"#{r.photo_id}/{r.idx} {r.objectness:.2f} {state}" if args.low < 0.5 else
                      f"#{r.photo_id}/{r.idx} {r.objectness:.2f} {r.category or '?'}")
    sheet = contact_sheet(tiles, labels, cell=220, columns=args.columns)
    path = args.out / f"near_threshold_{args.low:.2f}-{args.high:.2f}.jpg"
    imwrite(path, sheet)
    print(f"{len(near)} candidates with {args.low} <= objectness < {args.high} -> {path}")
    lib.close()


if __name__ == "__main__":
    main()
