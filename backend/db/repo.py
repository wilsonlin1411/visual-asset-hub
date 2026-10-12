"""Thin data-access functions. Callers group multi-step writes with `db.transaction()`."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from .models import Asset, BBox, CandidateRecord, Database, Photo, Tag, utcnow


# ------------------------------------------------------------------------------ photos

def insert_photo(db: Database, path: str, original_name: str, sha256: str, width: int, height: int,
                 status: str = "ready") -> Photo:
    created = utcnow()
    cur = db.conn.execute(
        "INSERT INTO photos (path, original_name, sha256, width, height, created_at, status)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (path, original_name, sha256, width, height, created, status))
    return Photo(cur.lastrowid, path, original_name, sha256, width, height, created, status)


def set_photo_path(db: Database, photo_id: int, path: str) -> None:
    db.conn.execute("UPDATE photos SET path = ? WHERE id = ?", (path, photo_id))


def set_photo_status(db: Database, photo_id: int, status: str, error: Optional[str] = None) -> None:
    db.conn.execute("UPDATE photos SET status = ?, error = ? WHERE id = ?", (status, error, photo_id))


def photo_ids_with_status(db: Database, status: str) -> List[int]:
    return [r[0] for r in db.conn.execute("SELECT id FROM photos WHERE status = ? ORDER BY id", (status,))]


def photo_ids_by_name(db: Database) -> Dict[str, List[int]]:
    """original_name -> photo ids (a name can repeat when different files share it)."""
    out: Dict[str, List[int]] = {}
    for pid, name in db.conn.execute("SELECT id, original_name FROM photos ORDER BY id"):
        out.setdefault(name, []).append(pid)
    return out


def set_photo_tags(db: Database, photo_id: int, tags: Iterable[tuple]) -> None:
    """Replace photo-level tags; each tag is (kind, value, score, source)."""
    db.conn.execute("DELETE FROM photo_tags WHERE photo_id = ?", (photo_id,))
    db.conn.executemany(
        "INSERT INTO photo_tags (photo_id, kind, value, score, source) VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT (photo_id, kind, value) DO UPDATE SET score = MAX(score, excluded.score)",
        [(photo_id, *t) for t in tags])


def get_photo_tags(db: Database, photo_id: int) -> List[tuple]:
    rows = db.conn.execute("SELECT kind, value, score, source FROM photo_tags WHERE photo_id = ?"
                           " ORDER BY score DESC", (photo_id,))
    return [tuple(r) for r in rows]


def get_photo(db: Database, photo_id: int) -> Optional[Photo]:
    row = db.conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
    return Photo.from_row(row) if row else None


def get_photo_by_sha256(db: Database, sha256: str) -> Optional[Photo]:
    row = db.conn.execute("SELECT * FROM photos WHERE sha256 = ?", (sha256,)).fetchone()
    return Photo.from_row(row) if row else None


def list_photos(db: Database, limit: int = 100, offset: int = 0) -> List[Photo]:
    rows = db.conn.execute("SELECT * FROM photos ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
    return [Photo.from_row(r) for r in rows]


def delete_photo(db: Database, photo_id: int) -> bool:
    """Cascades to candidates, assets, tags and embeddings."""
    return db.conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,)).rowcount > 0


# -------------------------------------------------------------------------- candidates

def insert_candidate(db: Database, photo_id: int, idx: int, class_id: Optional[int],
                     class_name: Optional[str], score: float, label_model: Optional[str],
                     mask_model: str, bbox: BBox, area: int, mask_path: str,
                     objectness: Optional[float] = None) -> int:
    cur = db.conn.execute(
        "INSERT INTO candidates (photo_id, idx, class_id, class_name, score, label_model, mask_model,"
        " x0, y0, x1, y1, area, mask_path, objectness, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (photo_id, idx, class_id, class_name, score, label_model, mask_model, *bbox, area, mask_path,
         objectness, utcnow()))
    return cur.lastrowid


def list_candidates(db: Database, photo_id: int, idxs: Optional[Iterable[int]] = None,
                    statuses: Optional[Sequence[str]] = None) -> List[CandidateRecord]:
    sql = "SELECT * FROM candidates WHERE photo_id = ?"
    params: list = [photo_id]
    if idxs is not None:
        idxs = list(idxs)
        sql += f" AND idx IN ({','.join('?' * len(idxs))})"
        params += idxs
    if statuses:
        sql += f" AND status IN ({','.join('?' * len(statuses))})"
        params += list(statuses)
    rows = db.conn.execute(sql + " ORDER BY idx", params)
    return [CandidateRecord.from_row(r) for r in rows]


def set_candidate_status(db: Database, candidate_ids: Iterable[int], status: str) -> None:
    db.conn.executemany("UPDATE candidates SET status = ? WHERE id = ?",
                        [(status, cid) for cid in candidate_ids])


def set_candidate_analysis(db: Database, candidate_id: int, objectness: Optional[float],
                           category: Optional[str], category_score: Optional[float]) -> None:
    db.conn.execute("UPDATE candidates SET objectness = ?, category = ?, category_score = ? WHERE id = ?",
                    (objectness, category, category_score, candidate_id))


# ------------------------------------------------------------------------------ assets

def insert_asset(db: Database, photo_id: int, candidate_id: Optional[int], png_path: str, bbox: BBox,
                 origin: tuple, width: int, height: int, category: Optional[str], source_model: str,
                 label_model: Optional[str], mask_model: str, score: Optional[float],
                 category_score: Optional[float] = None, category_source: Optional[str] = None) -> Asset:
    created = utcnow()
    cur = db.conn.execute(
        "INSERT INTO assets (photo_id, candidate_id, png_path, x0, y0, x1, y1, origin_x, origin_y,"
        " width, height, category, source_model, label_model, mask_model, score, created_at,"
        " category_score, category_source)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (photo_id, candidate_id, png_path, *bbox, *origin, width, height, category, source_model,
         label_model, mask_model, score, created, category_score, category_source))
    return Asset(cur.lastrowid, photo_id, candidate_id, png_path, tuple(bbox), tuple(origin), width,
                 height, category, source_model, label_model, mask_model, score, created,
                 category_score, category_source)


def set_asset_png_path(db: Database, asset_id: int, png_path: str) -> None:
    db.conn.execute("UPDATE assets SET png_path = ? WHERE id = ?", (png_path, asset_id))


def set_asset_category(db: Database, asset_id: int, category: Optional[str],
                       score: Optional[float] = None, source: Optional[str] = None) -> None:
    db.conn.execute("UPDATE assets SET category = ?, category_score = ?, category_source = ? WHERE id = ?",
                    (category, score, source, asset_id))


def all_asset_ids(db: Database) -> List[int]:
    return [r[0] for r in db.conn.execute("SELECT id FROM assets ORDER BY id")]


def get_assets(db: Database, asset_ids: Sequence[int]) -> Dict[int, Asset]:
    if not asset_ids:
        return {}
    rows = db.conn.execute(f"SELECT * FROM assets WHERE id IN ({','.join('?' * len(asset_ids))})",
                           list(asset_ids))
    return {r["id"]: Asset.from_row(r) for r in rows}


def asset_ids_matching(db: Database, categories: Sequence[str] = (), tag_values: Sequence[str] = ()) -> List[int]:
    """Assets whose category is in `categories` or that carry any tag value in `tag_values`."""
    parts, params = [], []
    if categories:
        parts.append(f"SELECT id FROM assets WHERE category IN ({','.join('?' * len(categories))})")
        params += list(categories)
    if tag_values:
        parts.append(f"SELECT asset_id FROM asset_tags WHERE value IN ({','.join('?' * len(tag_values))})")
        params += list(tag_values)
    if not parts:
        return []
    return [r[0] for r in db.conn.execute(" UNION ".join(parts), params)]


def tags_for_assets(db: Database, asset_ids: Sequence[int]) -> Dict[int, List[Tag]]:
    out: Dict[int, List[Tag]] = {a: [] for a in asset_ids}
    if not asset_ids:
        return out
    rows = db.conn.execute(
        f"SELECT * FROM asset_tags WHERE asset_id IN ({','.join('?' * len(asset_ids))})"
        f" ORDER BY kind, score DESC", list(asset_ids))
    for r in rows:
        out[r["asset_id"]].append(Tag(r["asset_id"], r["kind"], r["value"], r["score"], r["source"]))
    return out


def get_asset(db: Database, asset_id: int) -> Optional[Asset]:
    row = db.conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
    return Asset.from_row(row) if row else None


def get_asset_by_candidate(db: Database, candidate_id: int) -> Optional[Asset]:
    row = db.conn.execute("SELECT * FROM assets WHERE candidate_id = ?", (candidate_id,)).fetchone()
    return Asset.from_row(row) if row else None


def list_assets(db: Database, category: Optional[str] = None, photo_id: Optional[int] = None,
                tag: Optional[str] = None, tag_kind: Optional[str] = None,
                limit: int = 100, offset: int = 0) -> List[Asset]:
    sql = "SELECT a.* FROM assets a"
    where, params = [], []
    if tag is not None:
        sql += " JOIN asset_tags t ON t.asset_id = a.id"
        where.append("t.value = ?")
        params.append(tag)
        if tag_kind is not None:
            where.append("t.kind = ?")
            params.append(tag_kind)
    if category is not None:
        where.append("a.category = ?")
        params.append(category)
    if photo_id is not None:
        where.append("a.photo_id = ?")
        params.append(photo_id)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY a.id DESC LIMIT ? OFFSET ?"
    return [Asset.from_row(r) for r in db.conn.execute(sql, params + [limit, offset])]


def delete_asset(db: Database, asset_id: int) -> bool:
    """Cascades to tags and embeddings."""
    return db.conn.execute("DELETE FROM assets WHERE id = ?", (asset_id,)).rowcount > 0


# -------------------------------------------------------------------------------- tags

def set_tags(db: Database, asset_id: int, tags: Iterable[Tag], kind: Optional[str] = None,
             source: Optional[str] = None) -> None:
    """Replace the asset's tags (optionally only those of one kind / source)."""
    sql, params = "DELETE FROM asset_tags WHERE asset_id = ?", [asset_id]
    if kind is not None:
        sql += " AND kind = ?"
        params.append(kind)
    if source is not None:
        sql += " AND source = ?"
        params.append(source)
    db.conn.execute(sql, params)
    db.conn.executemany(
        "INSERT INTO asset_tags (asset_id, kind, value, score, source) VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT (asset_id, kind, value) DO UPDATE SET score = excluded.score, source = excluded.source",
        [(asset_id, t.kind, t.value, t.score, t.source) for t in tags])


def set_tag_override(db: Database, asset_id: int, kind: str, value: str, action: str) -> None:
    db.conn.execute(
        "INSERT INTO asset_tag_overrides (asset_id, kind, value, action) VALUES (?, ?, ?, ?)"
        " ON CONFLICT (asset_id, kind, value) DO UPDATE SET action = excluded.action",
        (asset_id, kind, value, action))


def get_tag_overrides(db: Database, asset_id: int, kind: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """{kind: {value: 'add' | 'remove'}}"""
    sql, params = "SELECT kind, value, action FROM asset_tag_overrides WHERE asset_id = ?", [asset_id]
    if kind is not None:
        sql += " AND kind = ?"
        params.append(kind)
    out: Dict[str, Dict[str, str]] = {}
    for k, v, a in db.conn.execute(sql, params):
        out.setdefault(k, {})[v] = a
    return out


def get_tags(db: Database, asset_id: int) -> List[Tag]:
    rows = db.conn.execute(
        "SELECT * FROM asset_tags WHERE asset_id = ? ORDER BY kind, score DESC, value", (asset_id,))
    return [Tag(r["asset_id"], r["kind"], r["value"], r["score"], r["source"]) for r in rows]


# ------------------------------------------------------------------------------- stats

def photo_counts(db: Database, photo_ids: Sequence[int]) -> Dict[int, Dict[str, int]]:
    """Per photo: candidates by status and number of assets (for photo lists)."""
    out = {pid: {"pending": 0, "confirmed": 0, "discarded": 0, "assets": 0} for pid in photo_ids}
    if not photo_ids:
        return out
    marks = ",".join("?" * len(photo_ids))
    for pid, status, n in db.conn.execute(
            f"SELECT photo_id, status, COUNT(*) FROM candidates WHERE photo_id IN ({marks}) GROUP BY photo_id, status",
            list(photo_ids)):
        out[pid][status] = n
    for pid, n in db.conn.execute(
            f"SELECT photo_id, COUNT(*) FROM assets WHERE photo_id IN ({marks}) GROUP BY photo_id", list(photo_ids)):
        out[pid]["assets"] = n
    return out


def counts(db: Database) -> dict:
    q = lambda sql: db.conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "photos": q("SELECT COUNT(*) FROM photos"),
        "candidates_pending": q("SELECT COUNT(*) FROM candidates WHERE status = 'pending'"),
        "candidates_confirmed": q("SELECT COUNT(*) FROM candidates WHERE status = 'confirmed'"),
        "candidates_discarded": q("SELECT COUNT(*) FROM candidates WHERE status = 'discarded'"),
        "assets": q("SELECT COUNT(*) FROM assets"),
        "tags": q("SELECT COUNT(*) FROM asset_tags"),
        "embeddings": q("SELECT COUNT(*) FROM embeddings"),
    }
