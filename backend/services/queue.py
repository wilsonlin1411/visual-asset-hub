"""Single background worker for photo processing (segmentation + CLIP take seconds).

One worker on purpose: the inference device (CPU now, one Hailo-10H later) is the bottleneck,
and serial jobs keep memory bounded. Photos left in 'processing' by a crash are re-queued
on start. The online demo shares this one worker between every visitor's library
(LibraryQueue); jobs are (library, photo_id).
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional, Tuple

from ..db import repo

log = logging.getLogger(__name__)

Job = Tuple[object, int]  # (library, photo_id)


class PhotoQueue:
    def __init__(self, library=None):
        self.library = library  # used when submit / position get no library
        self._q: "queue.Queue[Optional[Job]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._pending = set()  # (id(library), photo_id)
        self._lock = threading.Lock()
        self._current: Optional[Job] = None

    @property
    def current(self) -> Optional[int]:
        """Photo being processed, of any library."""
        job = self._current
        return job[1] if job is not None else None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="photo-worker", daemon=True)
        self._thread.start()
        if self.library is not None:
            for pid in repo.photo_ids_with_status(self.library.db, "processing"):
                log.info("re-queueing photo %d left in 'processing'", pid)
                self.submit(pid)

    def submit(self, photo_id: int, library=None) -> None:
        lib = library if library is not None else self.library
        with self._lock:
            if (id(lib), photo_id) in self._pending:
                return
            self._pending.add((id(lib), photo_id))
        self._q.put((lib, photo_id))

    def position(self, photo_id: int, library=None) -> Optional[int]:
        """0 = being processed, n = n jobs ahead, None = not queued."""
        lib = library if library is not None else self.library
        job = self._current
        if job is not None and job[0] is lib and job[1] == photo_id:
            return 0
        with self._lock:
            if (id(lib), photo_id) not in self._pending:
                return None
        for n, queued in enumerate(list(self._q.queue), 1):
            if queued is not None and queued[0] is lib and queued[1] == photo_id:
                return n
        return None

    def processing(self, library) -> Optional[int]:
        """Photo of `library` being processed right now."""
        job = self._current
        return job[1] if job is not None and job[0] is library else None

    def busy(self, library) -> bool:
        """Any job of `library` queued or running."""
        with self._lock:
            return any(lib_id == id(library) for lib_id, _ in self._pending)

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
            job = self._q.get()
            try:
                if job is None:
                    return
                self._current = job
                job[0].process_photo(job[1])
            except Exception:
                log.exception("processing photo %s failed", job[1] if job else None)
            finally:
                self._current = None
                if job is not None:
                    with self._lock:
                        self._pending.discard((id(job[0]), job[1]))
                self._q.task_done()


class LibraryQueue:
    """One library's view of a shared PhotoQueue, with the interface create_app uses."""

    def __init__(self, shared: PhotoQueue, library):
        self.shared = shared
        self.library = library

    @property
    def current(self) -> Optional[int]:
        return self.shared.processing(self.library)

    def submit(self, photo_id: int) -> None:
        self.shared.submit(photo_id, self.library)

    def position(self, photo_id: int) -> Optional[int]:
        return self.shared.position(photo_id, self.library)

    def busy(self) -> bool:
        return self.shared.busy(self.library)
