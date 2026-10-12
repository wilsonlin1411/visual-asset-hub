"""SQLite schema, connection handling and row types.

Plain sqlite3; SQL stays close to what PostgreSQL accepts so a later move to
PostgreSQL + pgvector only touches this package. File paths are stored relative to
the data directory so the whole data/ folder can be moved.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

log = logging.getLogger(__name__)

SCHEMA_VERSION = 4

SCHEMA_V1 = """
CREATE TABLE photos (
    id            INTEGER PRIMARY KEY,
    path          TEXT    NOT NULL,
    original_name TEXT    NOT NULL,
    sha256        TEXT    NOT NULL UNIQUE,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE TABLE candidates (
    id          INTEGER PRIMARY KEY,
    photo_id    INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    class_id    INTEGER,
    class_name  TEXT,
    score       REAL    NOT NULL,
    label_model TEXT,
    mask_model  TEXT    NOT NULL,
    x0 INTEGER NOT NULL, y0 INTEGER NOT NULL, x1 INTEGER NOT NULL, y1 INTEGER NOT NULL,
    area        INTEGER NOT NULL,
    mask_path   TEXT    NOT NULL,
    objectness  REAL,
    status      TEXT    NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'confirmed', 'discarded')),
    created_at  TEXT    NOT NULL,
    UNIQUE (photo_id, idx)
);

CREATE TABLE assets (
    id           INTEGER PRIMARY KEY,
    photo_id     INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    candidate_id INTEGER REFERENCES candidates(id) ON DELETE SET NULL,
    png_path     TEXT    NOT NULL,
    x0 INTEGER NOT NULL, y0 INTEGER NOT NULL, x1 INTEGER NOT NULL, y1 INTEGER NOT NULL,
    origin_x     INTEGER NOT NULL,
    origin_y     INTEGER NOT NULL,
    width        INTEGER NOT NULL,
    height       INTEGER NOT NULL,
    category     TEXT,
    source_model TEXT    NOT NULL,
    label_model  TEXT,
    mask_model   TEXT    NOT NULL,
    score        REAL,
    created_at   TEXT    NOT NULL
);
CREATE INDEX assets_photo_id ON assets (photo_id);
CREATE INDEX assets_category ON assets (category);
CREATE UNIQUE INDEX assets_candidate_id ON assets (candidate_id);

CREATE TABLE asset_tags (
    asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    kind     TEXT    NOT NULL CHECK (kind IN ('attribute', 'context')),
    value    TEXT    NOT NULL,
    score    REAL,
    source   TEXT    NOT NULL,
    PRIMARY KEY (asset_id, kind, value)
);
CREATE INDEX asset_tags_kind_value ON asset_tags (kind, value);

-- Source of truth for vectors. vec_embeddings (sqlite-vec) is a rebuildable index over it.
-- (No semicolons inside comments: the schema is split on them.)
CREATE TABLE embeddings (
    id              INTEGER PRIMARY KEY,
    asset_id        INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    embedding_model TEXT    NOT NULL,
    backend         TEXT    NOT NULL,
    dim             INTEGER NOT NULL,
    vector          BLOB    NOT NULL,
    created_at      TEXT    NOT NULL,
    UNIQUE (asset_id, embedding_model, backend)
);
"""

# v2 (stage 4): background processing status, CLIP category / objectness per candidate,
# category provenance per asset, photo-level context tags.
SCHEMA_V2 = """
ALTER TABLE photos ADD COLUMN status TEXT NOT NULL DEFAULT 'ready'
    CHECK (status IN ('processing', 'ready', 'failed'));
ALTER TABLE photos ADD COLUMN error TEXT;
ALTER TABLE candidates ADD COLUMN category TEXT;
ALTER TABLE candidates ADD COLUMN category_score REAL;
ALTER TABLE assets ADD COLUMN category_score REAL;
ALTER TABLE assets ADD COLUMN category_source TEXT;

CREATE TABLE photo_tags (
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    kind     TEXT    NOT NULL CHECK (kind IN ('context')),
    value    TEXT    NOT NULL,
    score    REAL,
    source   TEXT    NOT NULL,
    PRIMARY KEY (photo_id, kind, value)
);
"""

# v3 (stage 5): manual tag edits. asset_tags holds the effective tags (what filters and search
# read); overrides record what the user added / removed so automatic re-tagging (retag,
# context refresh) re-applies them instead of overwriting them.
SCHEMA_V3 = """
CREATE TABLE asset_tag_overrides (
    asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    kind     TEXT    NOT NULL CHECK (kind IN ('attribute', 'context')),
    value    TEXT    NOT NULL,
    action   TEXT    NOT NULL CHECK (action IN ('add', 'remove')),
    PRIMARY KEY (asset_id, kind, value)
);
"""

# v4 (stage 6): canvas designs (scene stored as JSON) and their exported images.
SCHEMA_V4 = """
CREATE TABLE designs (
    id         INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    preset     TEXT    NOT NULL,
    width      INTEGER NOT NULL,
    height     INTEGER NOT NULL,
    scene      TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);

CREATE TABLE exports (
    id         INTEGER PRIMARY KEY,
    design_id  INTEGER NOT NULL REFERENCES designs(id) ON DELETE CASCADE,
    path       TEXT    NOT NULL,
    format     TEXT    NOT NULL CHECK (format IN ('png', 'jpeg')),
    width      INTEGER NOT NULL,
    height     INTEGER NOT NULL,
    bytes      INTEGER NOT NULL,
    created_at TEXT    NOT NULL
);
CREATE INDEX exports_design_id ON exports (design_id);
"""

MIGRATIONS = {1: SCHEMA_V1, 2: SCHEMA_V2, 3: SCHEMA_V3, 4: SCHEMA_V4}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """SQLite in autocommit mode; group writes with `transaction()`.

    File databases get one connection per thread (WAL: concurrent readers, one writer,
    writers wait via busy_timeout), so API worker threads and the background queue can
    share a Database. ":memory:" uses a single shared connection (tests).
    """

    def __init__(self, path: Union[str, Path], load_vec: bool = True):
        self.path = str(path)
        self._load_vec = load_vec
        self._local = threading.local()
        self._all: List[sqlite3.Connection] = []
        self._all_lock = threading.Lock()
        self._shared: Optional[sqlite3.Connection] = None
        self.has_vec = False
        first = self._connect()
        if self.path == ":memory:":
            self._shared = first
        else:
            self._local.conn = first
            first.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._shared is not None:
            return self._shared
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = self._connect()
        return conn

    def _connect(self) -> sqlite3.Connection:
        # Each connection is still used by one thread only (thread-local); the check is off so
        # close() can release connections that worker threads opened.
        conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        if self._load_vec:
            loaded = self._load_sqlite_vec(conn)
            if not self._all:
                self.has_vec = loaded
        with self._all_lock:
            self._all.append(conn)
        return conn

    @staticmethod
    def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            return True
        except Exception as exc:  # missing package, or sqlite3 built without extension support
            log.warning("sqlite-vec unavailable (%s); vector search falls back to numpy", exc)
            return False

    def _migrate(self) -> None:
        self.conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = self.conn.execute("SELECT version FROM schema_version").fetchone()
        version = row[0] if row else 0
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"database schema v{version} is newer than this code (v{SCHEMA_VERSION})")
        for target in range(version + 1, SCHEMA_VERSION + 1):
            with self.transaction():
                for statement in MIGRATIONS[target].split(";"):
                    if statement.strip():
                        self.conn.execute(statement)
                self.conn.execute("DELETE FROM schema_version")
                self.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (target,))
            if version:
                log.info("migrated database schema v%d -> v%d", target - 1, target)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.conn.in_transaction:  # nested: join the outer transaction
            yield self.conn
            return
        self.conn.execute("BEGIN")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    def close(self) -> None:
        with self._all_lock:
            for conn in self._all:
                conn.close()
            self._all.clear()
        self._local = threading.local()


# ----------------------------------------------------------------------------- row types

BBox = Tuple[int, int, int, int]


@dataclass
class Photo:
    id: int
    path: str
    original_name: str
    sha256: str
    width: int
    height: int
    created_at: str
    status: str = "ready"        # processing | ready | failed
    error: Optional[str] = None

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Photo":
        return cls(r["id"], r["path"], r["original_name"], r["sha256"], r["width"], r["height"],
                   r["created_at"], r["status"], r["error"])


@dataclass
class CandidateRecord:
    id: int
    photo_id: int
    idx: int
    class_id: Optional[int]
    class_name: Optional[str]
    score: float
    label_model: Optional[str]
    mask_model: str
    bbox: BBox
    area: int
    mask_path: str
    objectness: Optional[float]
    status: str
    created_at: str
    category: Optional[str] = None        # CLIP zero-shot guess (vocabulary category)
    category_score: Optional[float] = None

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "CandidateRecord":
        return cls(r["id"], r["photo_id"], r["idx"], r["class_id"], r["class_name"], r["score"],
                   r["label_model"], r["mask_model"], (r["x0"], r["y0"], r["x1"], r["y1"]),
                   r["area"], r["mask_path"], r["objectness"], r["status"], r["created_at"],
                   r["category"], r["category_score"])

    def hidden(self, threshold: float) -> bool:
        """Low-objectness candidates are hidden by default, unless YOLO gave them a class."""
        return self.class_name is None and self.objectness is not None and self.objectness < threshold


@dataclass
class Asset:
    id: int
    photo_id: int
    candidate_id: Optional[int]
    png_path: str
    bbox: BBox                 # object box in the source photo
    origin: Tuple[int, int]    # PNG top-left in photo coordinates (padding may make it negative)
    width: int                 # PNG size
    height: int
    category: Optional[str]
    source_model: str
    label_model: Optional[str]
    mask_model: str
    score: Optional[float]
    created_at: str
    category_score: Optional[float] = None
    category_source: Optional[str] = None  # yolo | clip | fallback | user

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Asset":
        return cls(r["id"], r["photo_id"], r["candidate_id"], r["png_path"],
                   (r["x0"], r["y0"], r["x1"], r["y1"]), (r["origin_x"], r["origin_y"]),
                   r["width"], r["height"], r["category"], r["source_model"], r["label_model"],
                   r["mask_model"], r["score"], r["created_at"], r["category_score"], r["category_source"])


@dataclass
class Tag:
    asset_id: int
    kind: str  # attribute | context
    value: str
    score: Optional[float]
    source: str  # e.g. clip-zeroshot, rule:cooccurrence, user
