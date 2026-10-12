"""Evaluate search against tests/search_cases.toml (hit@k per query, overall pass rate, MRR).

    conda run -n vah python tools/eval_search.py [--k 5] [--no-bonus]

Cases name photos by file name; they are matched to the library by the name they were
imported under, so build the library from tests/images first (see search_cases.toml).
--no-bonus disables the vocabulary bonuses to show what pure CLIP similarity achieves.
"""
from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.db import repo  # noqa: E402
from backend.inference import create_backend  # noqa: E402
from backend.services.embedding import query_encoder  # noqa: E402
from backend.services.library import AssetLibrary, embedding_space  # noqa: E402
from backend.services.search import SearchService  # noqa: E402
from backend.services.vocab import load_vocabulary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "tests" / "search_cases.toml")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-bonus", action="store_true")
    parser.add_argument("--target", type=float, default=0.8)
    args = parser.parse_args()

    cases = tomllib.loads(args.cases.read_text(encoding="utf-8"))["case"]
    cfg = config.SEARCH
    if args.no_bonus:
        cfg = dataclasses.replace(cfg, category_bonus=0, context_bonus=0, attribute_bonus=0)
    lib = AssetLibrary(analyzer_factory=lambda: None)
    svc = SearchService(lib, query_encoder(create_backend("onnx", "cpu")), load_vocabulary(), embedding_space(), cfg)
    by_name = repo.photo_ids_by_name(lib.db)
    missing = sorted({name for case in cases for name in case["photos"]} - by_name.keys())
    if missing:
        print(f"not in library (import tests/images first): {', '.join(missing)}\n")

    passed, rr, latencies = 0, [], []
    for case in cases:
        t0 = time.perf_counter()
        res = svc.search(case["query"], k=20)
        latencies.append((time.perf_counter() - t0) * 1000)
        photos = {pid for name in case["photos"] for pid in by_name.get(name, [])}
        cats = set(case.get("categories", []))

        def relevant(h):
            return h.asset.photo_id in photos and (not cats or h.asset.category in cats)

        rank = next((i + 1 for i, h in enumerate(res.hits) if relevant(h)), None)
        ok = rank is not None and rank <= args.k
        passed += ok
        rr.append(1 / rank if rank else 0.0)
        top = ", ".join(f"#{h.asset.photo_id}{h.asset.category or '?'}" for h in res.hits[:args.k])
        print(f"{'PASS' if ok else 'FAIL'}  rank={rank or '-':<3} 「{case['query']}」  top{args.k}: {top}")

    rate = passed / len(cases)
    print(f"\nhit@{args.k}: {passed}/{len(cases)} = {rate:.0%}  (target {args.target:.0%})  MRR={statistics.mean(rr):.3f}"
          f"  latency median {statistics.median(latencies):.0f}ms max {max(latencies):.0f}ms")
    lib.close()
    return 0 if rate >= args.target else 1


if __name__ == "__main__":
    sys.exit(main())
