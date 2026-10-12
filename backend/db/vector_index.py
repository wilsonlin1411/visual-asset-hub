"""Vector index over asset embeddings.

Every vector lives in the `embeddings` table (BLOB, L2-normalized) keyed by
(asset, embedding_model, backend): the "embedding space". Vectors from different
spaces are never compared, so ONNX FP32 and Hailo INT8 CLIP vectors can coexist while
the library is being re-embedded.

  SqliteVecIndex  sqlite-vec vec0 table as a KNN index over those rows (rebuildable).
  NumpyIndex      brute force over the BLOBs; fallback when sqlite-vec cannot load,
                  and the reference implementation in tests.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .models import Database, utcnow

log = logging.getLogger(__name__)

VEC_TABLE = "vec_embeddings"
MAX_VEC_K = 4096  # sqlite-vec's KNN limit


@dataclass(frozen=True)
class EmbeddingSpace:
    embedding_model: str  # e.g. "clip-vit-b32"
    backend: str          # e.g. "onnx-cpu", "hailo"

    @property
    def key(self) -> str:
        return f"{self.embedding_model}@{self.backend}"


class Hit(NamedTuple):
    asset_id: int
    score: float  # cosine similarity, higher is closer


def normalize(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n == 0 or not np.isfinite(n):
        raise ValueError("cannot index a zero or non-finite vector")
    return v / n


class VectorIndex(ABC):
    def __init__(self, db: Database, dim: int):
        self.db = db
        self.dim = dim

    # -- writes (shared): the embeddings table is the source of truth

    def upsert(self, asset_id: int, space: EmbeddingSpace, vector: np.ndarray) -> None:
        v = normalize(vector)
        if v.size != self.dim:
            raise ValueError(f"expected {self.dim}-d vector, got {v.size}")
        with self.db.transaction():
            row = self.db.conn.execute(
                "INSERT INTO embeddings (asset_id, embedding_model, backend, dim, vector, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (asset_id, embedding_model, backend)"
                " DO UPDATE SET vector = excluded.vector, dim = excluded.dim, created_at = excluded.created_at"
                " RETURNING id",
                (asset_id, space.embedding_model, space.backend, self.dim, v.tobytes(), utcnow())).fetchone()
            self._on_upsert(row[0], space, v)

    def delete(self, asset_id: int, space: Optional[EmbeddingSpace] = None) -> int:
        sql, params = "DELETE FROM embeddings WHERE asset_id = ?", [asset_id]
        if space is not None:
            sql += " AND embedding_model = ? AND backend = ?"
            params += [space.embedding_model, space.backend]
        with self.db.transaction():
            n = self.db.conn.execute(sql, params).rowcount
        self._on_delete()
        return n

    def get(self, asset_id: int, space: EmbeddingSpace) -> Optional[np.ndarray]:
        row = self.db.conn.execute(
            "SELECT vector FROM embeddings WHERE asset_id = ? AND embedding_model = ? AND backend = ?",
            (asset_id, space.embedding_model, space.backend)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32).copy() if row else None

    def count(self, space: Optional[EmbeddingSpace] = None) -> int:
        if space is None:
            return self.db.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return self.db.conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE embedding_model = ? AND backend = ?",
            (space.embedding_model, space.backend)).fetchone()[0]

    def missing(self, space: EmbeddingSpace) -> List[int]:
        """Assets without a vector in `space`: the work list for (re-)embedding."""
        rows = self.db.conn.execute(
            "SELECT a.id FROM assets a WHERE NOT EXISTS (SELECT 1 FROM embeddings e"
            " WHERE e.asset_id = a.id AND e.embedding_model = ? AND e.backend = ?) ORDER BY a.id",
            (space.embedding_model, space.backend))
        return [r[0] for r in rows]

    def spaces(self) -> List[Tuple[EmbeddingSpace, int]]:
        rows = self.db.conn.execute(
            "SELECT embedding_model, backend, COUNT(*) FROM embeddings GROUP BY embedding_model, backend")
        return [(EmbeddingSpace(m, b), n) for m, b, n in rows]

    # -- search (implementation specific)

    @abstractmethod
    def search(self, query: np.ndarray, space: EmbeddingSpace, k: int = 20,
               asset_ids: Optional[Collection[int]] = None) -> List[Hit]:
        """Top-k assets by cosine similarity within `space`, optionally restricted to `asset_ids`."""

    def _on_upsert(self, embedding_id: int, space: EmbeddingSpace, v: np.ndarray) -> None:
        pass

    def _on_delete(self) -> None:
        pass

    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        q = normalize(query)
        if q.size != self.dim:
            raise ValueError(f"expected {self.dim}-d query, got {q.size}")
        return q


class NumpyIndex(VectorIndex):
    """Brute-force cosine search over the stored BLOBs, cached per space."""

    def __init__(self, db: Database, dim: int):
        super().__init__(db, dim)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _on_upsert(self, embedding_id, space, v):
        self._cache.pop(space.key, None)

    def _on_delete(self):
        self._cache.clear()

    def invalidate(self) -> None:
        """Call after deleting assets/photos (cascade deletes bypass this object)."""
        self._cache.clear()

    def _matrix(self, space: EmbeddingSpace) -> Tuple[np.ndarray, np.ndarray]:
        if space.key not in self._cache:
            rows = self.db.conn.execute(
                "SELECT asset_id, vector FROM embeddings WHERE embedding_model = ? AND backend = ?"
                " ORDER BY asset_id", (space.embedding_model, space.backend)).fetchall()
            ids = np.array([r[0] for r in rows], dtype=np.int64)
            mat = (np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(-1, self.dim)
                   if rows else np.zeros((0, self.dim), np.float32))
            self._cache[space.key] = (ids, mat)
        return self._cache[space.key]

    def search(self, query, space, k=20, asset_ids=None):
        q = self._prepare_query(query)
        ids, mat = self._matrix(space)
        if asset_ids is not None:
            keep = np.isin(ids, np.fromiter(asset_ids, dtype=np.int64))
            ids, mat = ids[keep], mat[keep]
        if ids.size == 0 or k <= 0:
            return []
        scores = mat @ q
        k = min(k, ids.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [Hit(int(ids[i]), float(scores[i])) for i in top]


class SqliteVecIndex(VectorIndex):
    """KNN through a sqlite-vec vec0 table whose rowid is embeddings.id.

    A trigger keeps it in sync on delete, including FK cascades from assets/photos.
    """

    def __init__(self, db: Database, dim: int):
        if not db.has_vec:
            raise RuntimeError("sqlite-vec is not loaded on this connection")
        super().__init__(db, dim)
        self._ensure()

    def _ensure(self) -> None:
        with self.db.transaction():
            self.db.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE} USING vec0("
                f" embedding_id INTEGER PRIMARY KEY, space TEXT,"
                f" embedding float[{self.dim}] distance_metric=cosine)")
            self.db.conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS embeddings_vec_delete AFTER DELETE ON embeddings"
                f" BEGIN DELETE FROM {VEC_TABLE} WHERE embedding_id = old.id; END")
        n_vec = self.db.conn.execute(f"SELECT COUNT(*) FROM {VEC_TABLE}").fetchone()[0]
        if n_vec != self.count():
            log.info("vector index out of sync (%d vs %d rows); rebuilding", n_vec, self.count())
            self.rebuild()

    def rebuild(self) -> None:
        """Recreate the vec0 rows from the embeddings table."""
        with self.db.transaction():
            self.db.conn.execute(f"DELETE FROM {VEC_TABLE}")
            rows = self.db.conn.execute(
                "SELECT id, embedding_model, backend, vector FROM embeddings WHERE dim = ?", (self.dim,))
            self.db.conn.executemany(
                f"INSERT INTO {VEC_TABLE} (embedding_id, space, embedding) VALUES (?, ?, ?)",
                [(r[0], EmbeddingSpace(r[1], r[2]).key, r[3]) for r in rows])

    def _on_upsert(self, embedding_id, space, v):
        # vec0 has no upsert: replace the row.
        self.db.conn.execute(f"DELETE FROM {VEC_TABLE} WHERE embedding_id = ?", (embedding_id,))
        self.db.conn.execute(f"INSERT INTO {VEC_TABLE} (embedding_id, space, embedding) VALUES (?, ?, ?)",
                             (embedding_id, space.key, v.tobytes()))

    def search(self, query, space, k=20, asset_ids=None):
        q = self._prepare_query(query)
        if k <= 0:
            return []
        # KNN runs alone on the vec0 table (CTE), the join to asset ids happens afterwards.
        knn = f"SELECT embedding_id, distance FROM {VEC_TABLE} WHERE embedding MATCH ? AND k = ? AND space = ?"
        params: list = [q.tobytes(), min(k, MAX_VEC_K), space.key]
        if asset_ids is not None:
            asset_ids = list(asset_ids)
            if not asset_ids:
                return []
            emb_ids = [r[0] for r in self.db.conn.execute(
                f"SELECT id FROM embeddings WHERE embedding_model = ? AND backend = ?"
                f" AND asset_id IN ({','.join('?' * len(asset_ids))})",
                [space.embedding_model, space.backend, *asset_ids])]
            if not emb_ids:
                return []
            knn += f" AND embedding_id IN ({','.join('?' * len(emb_ids))})"
            params += emb_ids
        rows = self.db.conn.execute(
            f"WITH knn AS ({knn}) SELECT e.asset_id, knn.distance FROM knn"
            f" JOIN embeddings e ON e.id = knn.embedding_id ORDER BY knn.distance", params).fetchall()
        return [Hit(r[0], 1.0 - float(r[1])) for r in rows]


def create_index(db: Database, dim: int) -> VectorIndex:
    if db.has_vec:
        return SqliteVecIndex(db, dim)
    log.warning("using NumpyIndex (brute force) because sqlite-vec is unavailable")
    return NumpyIndex(db, dim)
