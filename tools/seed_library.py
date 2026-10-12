"""Simulate a user confirming the obvious objects of every photo (demo / evaluation setup).

Confirms visible candidates that look like real objects with a confident category:
  - not hidden, objectness >= --min-objectness
  - category not 其他 / 人物, and YOLO+CLIP agree or CLIP prob >= --min-prob
  - largest first; a candidate mostly inside an already chosen one of the same category
    (a fragment of the same object) is skipped

    conda run -n vah python tools/seed_library.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.db import repo  # noqa: E402
from backend.inference.masks import intersection  # noqa: E402
from backend.services.library import AssetLibrary  # noqa: E402
from backend.services.vocab import load_vocabulary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-prob", type=float, default=0.6)
    parser.add_argument("--min-objectness", type=float, default=0.6)
    parser.add_argument("--containment", type=float, default=0.8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    vocab = load_vocabulary()
    skip = {vocab.fallback_category.name, *vocab.excluded_categories}
    lib = AssetLibrary()
    total = 0
    t0 = time.perf_counter()
    for pid in repo.photo_ids_with_status(lib.db, "ready"):
        records = lib.candidates(pid)
        chosen = []
        taken_by_cat = {}
        for r in sorted(records, key=lambda r: -r.area):
            if r.category in skip or r.category is None or (r.objectness or 0) < args.min_objectness:
                continue
            yolo_agrees = r.class_name is not None and vocab.coco_to_category(r.class_name) == r.category
            if not yolo_agrees and (r.category_score or 0) < args.min_prob:
                continue
            mask = lib.load_candidate_mask(r)
            area = mask.area
            if any(intersection(mask, other) >= args.containment * area for other in taken_by_cat.get(r.category, [])):
                continue
            taken_by_cat.setdefault(r.category, []).append(mask)
            if r.status != "confirmed":
                chosen.append(r)
        names = [f"{r.idx}:{r.category}" for r in chosen]
        print(f"photo {pid}: confirm {names}")
        if chosen and not args.dry_run:
            lib.confirm(pid, [r.idx for r in chosen])
        total += len(chosen)
    print(f"{total} candidates confirmed in {time.perf_counter() - t0:.0f}s; library: {lib.stats()['assets']} assets")
    lib.close()


if __name__ == "__main__":
    main()
