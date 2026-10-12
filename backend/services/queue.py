"""Single background worker for photo processing (segmentation + CLIP take seconds).

One worker on purpose: the inference device (CPU now, one Hailo-10H later) is the bottleneck,
and serial jobs keep memory bounded. Photos left in 'processing' by a crash are re-queued
on start.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from ..db import repo

log = logging.getLogger(__name__)


class PhotoQueue:
    def __init__(self, library):
        self.library = library
        self._q: "queue.Queue[Optional[int]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._pending = set()
        self._lock = threading.Lock()
        self.current: Optional[int] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="photo-worker", daemon=True)
        self._thread.start()
        for pid in repo.photo_ids_with_status(self.library.db, "processing"):
            log.info("re-queueing photo %d left in 'processing'", pid)
            self.submit(pid)

    def submit(self, photo_id: int) -> None:
        with self._lock:
            if photo_id in self._pending:
                return
            self._pending.add(photo_id)
        self._q.put(photo_id)

    def position(self, photo_id: int) -> Optional[int]:
        """0 = being processed, n = n jobs ahead, None = not queued."""
        if self.current == photo_id:
            return 0
        with self._lock:
            if photo_id not in self._pending:
                return None
        return list(self._q.queue).index(photo_id) + 1 if photo_id in self._q.queue else None

    def join(self) -> None:
        """Wait until every submitted job is done (tests, CLI)."""
        self._q.join()

    def stop(self) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=30)
            self._thread = None

    def _run(self) -> None:
        while True:
            pid = self._q.get()
            try:
                if pid is None:
                    return
                self.current = pid
                self.library.process_photo(pid)
            except Exception:
                log.exception("processing photo %s failed", pid)
            finally:
                self.current = None
                if pid is not None:
                    with self._lock:
                        self._pending.discard(pid)
                self._q.task_done()
