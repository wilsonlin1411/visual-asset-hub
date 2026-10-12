"""Online demo (VAH_DEMO=1): every visitor works on a private copy of the demo library.

The browser keeps a random session id and sends it with every API call (X-VAH-Session header,
or ?s= on URLs it loads by itself; see frontend/src/api.ts). No cookies: the hosted demo runs in
a cross-site iframe, where browsers block them. Each id gets its own data dir, copied from the
seed library in config.DATA_DIR, and its own create_app() instance. Models, the query encoder
and the photo worker are shared, so memory does not grow with visitors. Sessions idle for
DEMO.idle_minutes are deleted.

    VAH_DEMO=1 python -m backend.api
"""
from __future__ import annotations

import logging
import re
import shutil
import sqlite3
import threading
import time
from collections import deque
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, Optional
from urllib.parse import parse_qs

import anyio
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .. import config
from ..db.vector_index import EmbeddingSpace
from ..services.library import AssetLibrary, default_analyzer, default_pipeline, embedding_space
from ..services.queue import LibraryQueue, PhotoQueue
from ..services.search import SearchService
from .app import DemoInfo, create_app, mount_frontend, vocabulary

log = logging.getLogger(__name__)

SESSION_RE = re.compile(r"[0-9a-f]{32}")
RATE_WINDOW_S = 600
SWEEP_EVERY_S = 60


def default_query_encoder():
    from ..inference import create_backend
    from ..services.embedding import query_encoder
    return query_encoder(create_backend("onnx", "cpu"))


class SharedModels:
    """Each model loads once and serves every session (lazy, thread-safe)."""

    def __init__(self, pipeline_factory=default_pipeline, analyzer_factory=default_analyzer,
                 query_encoder_factory=default_query_encoder, space: Optional[EmbeddingSpace] = None):
        self._factories = {"pipeline": pipeline_factory, "analyzer": analyzer_factory,
                           "query": query_encoder_factory}
        self._loaded: dict = {}
        self._lock = threading.Lock()
        self.space = space or embedding_space()

    def _get(self, name: str):
        with self._lock:
            if name not in self._loaded:
                self._loaded[name] = self._factories[name]()
            return self._loaded[name]

    def pipeline(self):
        return self._get("pipeline")

    def analyzer(self):
        return self._get("analyzer")

    def search(self, lib: AssetLibrary) -> SearchService:
        return SearchService(lib, self._get("query"), vocabulary(), self.space)

    def warm_up(self) -> None:
        """Load everything up front so the first visitor does not wait for it."""
        t0 = time.perf_counter()
        for name in self._factories:
            self._get(name)
        log.info("demo models loaded in %.1fs", time.perf_counter() - t0)


@dataclass
class Session:
    id: str
    dir: Path
    library: AssetLibrary
    queue: LibraryQueue
    app: FastAPI
    last_used: float


class SessionLimit(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def copy_database(src: Path, dst: Path) -> None:
    """Consistent copy of a live SQLite file (includes what is still in its WAL)."""
    with closing(sqlite3.connect(src)) as s, closing(sqlite3.connect(dst)) as d:
        s.backup(d)


class SessionManager:
    def __init__(self, seed_dir: Path = config.DATA_DIR, models: Optional[SharedModels] = None,
                 cfg: config.DemoConfig = config.DEMO, clock: Callable[[], float] = time.monotonic):
        self.seed_dir = Path(seed_dir)
        self.cfg = cfg
        self.models = models or SharedModels()
        self.queue = PhotoQueue()
        self.clock = clock
        self._sessions: Dict[str, Session] = {}
        self._created_by_ip: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()
        self.seed_db = self.seed_dir / config.DB_FILENAME
        self.seed_photos = 0
        if self.seed_db.is_file():
            with closing(sqlite3.connect(self.seed_db)) as db:
                self.seed_photos = db.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
        else:
            log.warning("no demo library in %s; sessions start empty", self.seed_dir)
        cfg.sessions_dir.mkdir(parents=True, exist_ok=True)
        for d in cfg.sessions_dir.iterdir():  # left over from a previous run
            if d.is_dir() and SESSION_RE.fullmatch(d.name):
                shutil.rmtree(d, ignore_errors=True)

    def __len__(self) -> int:
        return len(self._sessions)

    def lookup(self, sid: str) -> Optional[Session]:
        # No lock: runs on the event loop, and open() holds the lock while it copies files.
        # Only open/sweep/close (worker threads, under the lock) change the dict; get() is atomic.
        s = self._sessions.get(sid)
        if s is not None:
            s.last_used = self.clock()
        return s

    def open(self, sid: str, client: str) -> Session:
        """The session of `sid`, created from the seed library if new (blocking: copies files)."""
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                self._check_rate(client)
                self._make_room()
                s = self._sessions[sid] = self._create(sid)
                log.info("demo session %s… opened (%d active)", sid[:8], len(self._sessions))
            s.last_used = self.clock()
            return s

    def sweep(self) -> int:
        """Delete sessions idle longer than cfg.idle_minutes; returns how many."""
        with self._lock:
            now = self.clock()
            expired = [s for s in self._sessions.values()
                       if now - s.last_used >= self.cfg.idle_minutes * 60 and not s.queue.busy()]
            for s in expired:
                self._drop(s)
            for ip in [ip for ip, q in self._created_by_ip.items() if not q or now - q[-1] > RATE_WINDOW_S]:
                del self._created_by_ip[ip]
            return len(expired)

    def close(self) -> None:
        with self._lock:
            for s in list(self._sessions.values()):
                self._drop(s)

    # ----------------------------------------------------------------------- helpers

    def _check_rate(self, client: str) -> None:
        now = self.clock()
        q = self._created_by_ip.setdefault(client, deque())
        while q and now - q[0] > RATE_WINDOW_S:
            q.popleft()
        if len(q) >= self.cfg.new_sessions_per_ip:
            raise SessionLimit(429, "這個網路建立了太多 Demo 工作階段，請稍後再試")
        q.append(now)

    def _make_room(self) -> None:
        if len(self._sessions) < self.cfg.max_sessions:
            return
        idle = sorted((s for s in self._sessions.values() if not s.queue.busy()), key=lambda s: s.last_used)
        if not idle or self.clock() - idle[0].last_used < self.cfg.evict_idle_minutes * 60:
            raise SessionLimit(503, "線上 Demo 目前使用人數已滿，請稍後再試")
        self._drop(idle[0])

    def _create(self, sid: str) -> Session:
        path = self.cfg.sessions_dir / sid
        if self.seed_db.is_file():
            shutil.copytree(self.seed_dir, path, ignore=shutil.ignore_patterns(
                f"{config.DB_FILENAME}*", "cache", self.cfg.sessions_dir.name))
            copy_database(self.seed_db, path / config.DB_FILENAME)
        lib = AssetLibrary(path, pipeline_factory=self.models.pipeline, analyzer_factory=self.models.analyzer)
        queue = LibraryQueue(self.queue, lib)
        app = create_app(lib, search_factory=self.models.search, start_worker=False, serve_frontend=False,
                         worker=queue, max_photos=self.seed_photos + self.cfg.max_uploads,
                         max_upload_bytes=self.cfg.max_upload_mb * 2**20)
        return Session(sid, path, lib, queue, app, self.clock())

    def _drop(self, s: Session) -> None:
        del self._sessions[s.id]
        s.library.close()
        shutil.rmtree(s.dir, ignore_errors=True)
        log.info("demo session %s… closed (%d active)", s.id[:8], len(self._sessions))


def session_id(scope) -> Optional[str]:
    sid = next((v.decode("latin-1") for k, v in scope["headers"] if k == b"x-vah-session"), None)
    if sid is None:
        sid = parse_qs(scope.get("query_string", b"").decode("latin-1")).get("s", [None])[0]
    return sid if sid and SESSION_RE.fullmatch(sid) else None


def client_ip(scope) -> str:
    forwarded = next((v.decode("latin-1") for k, v in scope["headers"] if k == b"x-forwarded-for"), "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return scope["client"][0] if scope.get("client") else "unknown"


class SessionRouter:
    """ASGI entry point: /api/* goes to the caller's session app, the rest to the shell app."""

    def __init__(self, shell: FastAPI, manager: SessionManager):
        self.shell = shell
        self.manager = manager

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not path.startswith("/api/") or path == "/api/demo":
            return await self.shell(scope, receive, send)
        sid = session_id(scope)
        if sid is None:
            return await JSONResponse({"detail": "missing demo session id (X-VAH-Session)"}, 400)(scope, receive, send)
        session = self.manager.lookup(sid)
        if session is None:
            try:
                session = await anyio.to_thread.run_sync(self.manager.open, sid, client_ip(scope))
            except SessionLimit as exc:
                return await JSONResponse({"detail": exc.detail}, exc.status)(scope, receive, send)
        await session.app(scope, receive, send)


def create_demo_app(manager: Optional[SessionManager] = None, serve_frontend: bool = True,
                    warm_up: bool = True) -> SessionRouter:
    if manager is None:
        manager = SessionManager()
    cfg = manager.cfg

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.queue.start()
        stop = threading.Event()
        if warm_up:
            threading.Thread(target=manager.models.warm_up, name="demo-warm-up", daemon=True).start()

        def sweep_loop():
            while not stop.wait(SWEEP_EVERY_S):
                manager.sweep()

        threading.Thread(target=sweep_loop, name="demo-sweeper", daemon=True).start()
        yield
        stop.set()
        manager.queue.stop()
        manager.close()

    shell = FastAPI(title="Visual Asset Hub demo", lifespan=lifespan, docs_url=None, redoc_url=None,
                    openapi_url=None)
    shell.state.sessions = manager

    @shell.get("/api/demo", response_model=DemoInfo)
    def demo_info():
        return DemoInfo(enabled=True, seed_photos=manager.seed_photos, idle_minutes=cfg.idle_minutes,
                        max_uploads=cfg.max_uploads, source_url=cfg.source_url)

    if serve_frontend:
        mount_frontend(shell)
    return SessionRouter(shell, manager)
