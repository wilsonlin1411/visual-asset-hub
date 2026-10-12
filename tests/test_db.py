import sqlite3

import pytest

from backend.db import Database, Tag, repo
from backend.db.models import SCHEMA_VERSION


def make_asset(db, photo_id, category="cup", candidate_id=None):
    return repo.insert_asset(db, photo_id, candidate_id, "assets/x.png", (1, 2, 3, 4), (-7, -6), 10, 10,
                             category, "fastsam-s", "yolov8s-seg", "fastsam-s", 0.9)


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "t.db"
    Database(path).close()
    db = Database(path)  # second open must not re-run the schema
    assert [tuple(r) for r in db.conn.execute("SELECT version FROM schema_version")] == [(SCHEMA_VERSION,)]
    db.close()


def test_photo_sha_is_unique(db):
    repo.insert_photo(db, "photos/1.jpg", "a.jpg", "abc", 10, 20)
    assert repo.get_photo_by_sha256(db, "abc").original_name == "a.jpg"
    with pytest.raises(sqlite3.IntegrityError):
        repo.insert_photo(db, "photos/2.jpg", "b.jpg", "abc", 10, 20)


def test_candidates_roundtrip_and_status(db):
    p = repo.insert_photo(db, "p", "a.jpg", "s", 100, 100)
    cid = repo.insert_candidate(db, p.id, 0, None, None, 0.7, None, "fastsam-s", (1, 2, 30, 40), 500, "m.png")
    repo.insert_candidate(db, p.id, 1, 41, "cup", 0.9, "yolov8s-seg", "fastsam-s", (5, 5, 9, 9), 16, "n.png")
    c = repo.list_candidates(db, p.id, [0])[0]
    assert (c.bbox, c.class_name, c.status, c.objectness) == ((1, 2, 30, 40), None, "pending", None)
    repo.set_candidate_status(db, [cid], "confirmed")
    assert [c.idx for c in repo.list_candidates(db, p.id, statuses=["pending"])] == [1]
    with pytest.raises(sqlite3.IntegrityError):
        repo.set_candidate_status(db, [cid], "bogus")


def test_delete_photo_cascades(db):
    p = repo.insert_photo(db, "p", "a.jpg", "s", 100, 100)
    cid = repo.insert_candidate(db, p.id, 0, None, None, 0.7, None, "fastsam-s", (1, 2, 30, 40), 500, "m.png")
    a = make_asset(db, p.id, candidate_id=cid)
    repo.set_tags(db, a.id, [Tag(a.id, "context", "remote work", 0.8, "rule")])
    db.conn.execute("INSERT INTO embeddings (asset_id, embedding_model, backend, dim, vector, created_at)"
                    " VALUES (?, 'm', 'b', 1, x'00000000', 'now')", (a.id,))
    repo.delete_photo(db, p.id)
    assert repo.counts(db) == {"photos": 0, "candidates_pending": 0, "candidates_confirmed": 0,
                               "candidates_discarded": 0, "assets": 0, "tags": 0, "embeddings": 0}


def test_asset_fields_and_tag_filters(db):
    p = repo.insert_photo(db, "p", "a.jpg", "s", 100, 100)
    a1 = make_asset(db, p.id, "laptop")
    a2 = make_asset(db, p.id, "cup")
    got = repo.get_asset(db, a1.id)
    assert got.bbox == (1, 2, 3, 4) and got.origin == (-7, -6) and got.category == "laptop"

    repo.set_tags(db, a1.id, [Tag(a1.id, "context", "remote work", 0.9, "rule"),
                              Tag(a1.id, "attribute", "silver", 0.7, "clip")])
    repo.set_tags(db, a2.id, [Tag(a2.id, "context", "remote work", 0.6, "rule")])
    assert {a.id for a in repo.list_assets(db, tag="remote work")} == {a1.id, a2.id}
    assert [a.id for a in repo.list_assets(db, tag="silver", tag_kind="attribute")] == [a1.id]
    assert [a.id for a in repo.list_assets(db, category="cup")] == [a2.id]

    # replacing one kind keeps the other
    repo.set_tags(db, a1.id, [Tag(a1.id, "attribute", "black", 0.5, "clip")], kind="attribute")
    assert {(t.kind, t.value) for t in repo.get_tags(db, a1.id)} == {("context", "remote work"),
                                                                      ("attribute", "black")}


def test_transaction_rolls_back(db):
    with pytest.raises(RuntimeError):
        with db.transaction():
            repo.insert_photo(db, "p", "a.jpg", "s", 1, 1)
            raise RuntimeError("boom")
    assert repo.counts(db)["photos"] == 0
