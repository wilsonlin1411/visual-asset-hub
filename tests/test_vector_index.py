import numpy as np
import pytest

from backend.config import EMBEDDING_DIM
from backend.db import Database, repo
from backend.db.vector_index import VEC_TABLE, EmbeddingSpace, NumpyIndex, SqliteVecIndex

A = EmbeddingSpace("clip-vit-b32", "onnx-cpu")
B = EmbeddingSpace("clip-vit-b32", "hailo")
DIM = EMBEDDING_DIM


def brute_force(vectors: dict, query, k, allowed=None):
    q = query / np.linalg.norm(query)
    scored = [(aid, float(v @ q / np.linalg.norm(v))) for aid, v in vectors.items()
              if allowed is None or aid in allowed]
    return [aid for aid, _ in sorted(scored, key=lambda t: -t[1])[:k]]


@pytest.fixture(params=["numpy", "sqlite-vec"])
def setup(request):
    db = Database(":memory:")
    index = NumpyIndex(db, DIM) if request.param == "numpy" else SqliteVecIndex(db, DIM)
    photo = repo.insert_photo(db, "p", "a.jpg", "s", 10, 10)
    rng = np.random.default_rng(42)
    vecs_a, vecs_b = {}, {}
    for i in range(300):
        a = repo.insert_asset(db, photo.id, None, f"assets/{i}.png", (0, 0, 1, 1), (0, 0), 1, 1,
                              None, "fastsam-s", None, "fastsam-s", None)
        vecs_a[a.id] = rng.normal(size=DIM).astype(np.float32)
        index.upsert(a.id, A, vecs_a[a.id])
        if i % 2 == 0:
            vecs_b[a.id] = rng.normal(size=DIM).astype(np.float32)
            index.upsert(a.id, B, vecs_b[a.id])
    yield db, index, vecs_a, vecs_b, rng
    db.close()


def test_search_matches_brute_force_per_space(setup):
    db, index, vecs_a, vecs_b, rng = setup
    for _ in range(5):
        q = rng.normal(size=DIM)
        assert [h.asset_id for h in index.search(q, A, k=10)] == brute_force(vecs_a, q, 10)
        assert [h.asset_id for h in index.search(q, B, k=10)] == brute_force(vecs_b, q, 10)
    hits = index.search(q, A, k=3)
    assert all(-1.0 <= h.score <= 1.0 for h in hits) and hits[0].score >= hits[-1].score


def test_exact_match_scores_one(setup):
    db, index, vecs_a, *_ = setup
    aid = next(iter(vecs_a))
    top = index.search(vecs_a[aid] * 3.0, A, k=1)[0]  # scale must not matter
    assert top.asset_id == aid and top.score == pytest.approx(1.0, abs=1e-5)


def test_asset_filter(setup):
    db, index, vecs_a, _, rng = setup
    allowed = set(list(vecs_a)[:40])
    q = rng.normal(size=DIM)
    assert [h.asset_id for h in index.search(q, A, k=5, asset_ids=allowed)] == brute_force(vecs_a, q, 5, allowed)
    assert index.search(q, A, k=5, asset_ids=[]) == []


def test_upsert_replaces_and_delete_removes(setup):
    db, index, vecs_a, _, rng = setup
    aid = next(iter(vecs_a))
    target = rng.normal(size=DIM)
    index.upsert(aid, A, target)
    assert index.count(A) == 300
    assert index.search(target, A, k=1)[0].asset_id == aid
    np.testing.assert_allclose(index.get(aid, A), target / np.linalg.norm(target), atol=1e-6)

    assert index.delete(aid, A) == 1
    assert aid not in [h.asset_id for h in index.search(target, A, k=300)]
    assert index.get(aid, A) is None


def test_asset_cascade_removes_vectors(setup):
    db, index, vecs_a, *_ = setup
    aid = next(iter(vecs_a))
    repo.delete_asset(db, aid)  # FK cascade on embeddings, trigger on vec0
    if isinstance(index, NumpyIndex):
        index.invalidate()
    assert aid not in [h.asset_id for h in index.search(vecs_a[aid], A, k=300)]
    assert index.count() == 299 + 149


def test_missing_lists_assets_to_reembed(setup):
    db, index, vecs_a, vecs_b, _ = setup
    assert index.missing(A) == []
    assert index.missing(B) == sorted(set(vecs_a) - set(vecs_b))
    assert dict((s.key, n) for s, n in index.spaces()) == {A.key: 300, B.key: 150}


def test_rejects_bad_vectors(setup):
    _, index, vecs_a, *_ = setup
    aid = next(iter(vecs_a))
    with pytest.raises(ValueError):
        index.upsert(aid, A, np.zeros(DIM))
    with pytest.raises(ValueError):
        index.upsert(aid, A, np.ones(DIM - 1))


def test_sqlite_vec_index_rebuilds_when_out_of_sync():
    db = Database(":memory:")
    index = SqliteVecIndex(db, DIM)
    photo = repo.insert_photo(db, "p", "a.jpg", "s", 10, 10)
    rng = np.random.default_rng(0)
    vecs = {}
    for i in range(20):
        a = repo.insert_asset(db, photo.id, None, "x", (0, 0, 1, 1), (0, 0), 1, 1, None, "m", None, "m", None)
        vecs[a.id] = rng.normal(size=DIM)
        index.upsert(a.id, A, vecs[a.id])
    db.conn.execute(f"DELETE FROM {VEC_TABLE}")  # simulate a lost / stale index
    rebuilt = SqliteVecIndex(db, DIM)
    q = rng.normal(size=DIM)
    assert [h.asset_id for h in rebuilt.search(q, A, k=5)] == brute_force(vecs, q, 5)
    db.close()
