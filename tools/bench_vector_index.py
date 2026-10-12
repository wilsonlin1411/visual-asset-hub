"""Benchmark vector search: N random 512-d vectors, top-k latency for each index.

    conda run -n vah python tools/bench_vector_index.py --n 10000
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import EMBEDDING_DIM  # noqa: E402
from backend.db import Database, repo  # noqa: E402
from backend.db.vector_index import EmbeddingSpace, NumpyIndex, SqliteVecIndex  # noqa: E402

SPACE = EmbeddingSpace("clip-vit-b32", "onnx-cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--queries", type=int, default=50)
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    with tempfile.TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "bench.db")
        vec = SqliteVecIndex(db, EMBEDDING_DIM)
        photo = repo.insert_photo(db, "p", "a.jpg", "s", 1, 1)
        t0 = time.perf_counter()
        with db.transaction():
            for _ in range(args.n):
                a = repo.insert_asset(db, photo.id, None, "x", (0, 0, 1, 1), (0, 0), 1, 1, None, "m", None, "m", None)
                vec.upsert(a.id, SPACE, rng.normal(size=EMBEDDING_DIM))
        print(f"insert {args.n} vectors: {time.perf_counter() - t0:.1f}s")

        queries = rng.normal(size=(args.queries, EMBEDDING_DIM))
        for index in (vec, NumpyIndex(db, EMBEDDING_DIM)):
            index.search(queries[0], SPACE, k=args.k)  # warm caches
            times = []
            for q in queries:
                t = time.perf_counter()
                index.search(q, SPACE, k=args.k)
                times.append((time.perf_counter() - t) * 1000)
            print(f"{type(index).__name__:<15} top-{args.k}: median {statistics.median(times):.1f}ms"
                  f"  p95 {sorted(times)[int(0.95 * len(times)) - 1]:.1f}ms")
        a = vec.search(queries[1], SPACE, k=args.k)
        b = NumpyIndex(db, EMBEDDING_DIM).search(queries[1], SPACE, k=args.k)
        print("results identical:", [h.asset_id for h in a] == [h.asset_id for h in b])
        db.close()


if __name__ == "__main__":
    main()
