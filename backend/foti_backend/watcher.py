"""Folder watcher — auto-rescan on filesystem changes.

A single ``FolderWatchService`` instance watches all enabled scan roots,
debouncing bursts of events into one ``scan_root`` call per root.

The watcher is started at app boot and gracefully stops on shutdown.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import get_settings
from .db import connect
from .scanner import scan_root

log = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 5.0


class _RootHandler(FileSystemEventHandler):
    """Per-root handler that schedules a debounced scan."""

    def __init__(self, root: Path, scheduler: "FolderWatchService") -> None:
        self.root = root
        self.scheduler = scheduler

    def _maybe_relevant(self, event_src_path: str) -> bool:
        p = Path(event_src_path)
        ext = p.suffix.lower()
        return ext in get_settings().scan_extensions

    def on_created(self, event) -> None:
        if not event.is_directory and self._maybe_relevant(event.src_path):
            self.scheduler.schedule(self.root)

    def on_modified(self, event) -> None:
        if not event.is_directory and self._maybe_relevant(event.src_path):
            self.scheduler.schedule(self.root)

    def on_moved(self, event) -> None:
        self.scheduler.schedule(self.root)


class FolderWatchService:
    """Watch every enabled scan root, run a debounced scan on changes."""

    def __init__(self) -> None:
        self._observer: Observer | None = None
        self._pending: dict[Path, float] = {}
        self._pending_lock = threading.Lock()
        self._stop = threading.Event()
        self._dispatcher: threading.Thread | None = None

    def start(self) -> None:
        if self._observer is not None:
            return
        self._observer = Observer()
        self._refresh_watches()
        self._observer.start()

        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="foti-watch-dispatcher",
            daemon=True,
        )
        self._dispatcher.start()
        log.info("FolderWatchService started")

    def stop(self) -> None:
        self._stop.set()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    def schedule(self, root: Path) -> None:
        with self._pending_lock:
            self._pending[root] = time.monotonic() + DEBOUNCE_SECONDS

    def _refresh_watches(self) -> None:
        """Read scan_root from the catalog and ensure each is being watched."""
        if self._observer is None:
            return
        conn = connect()
        roots = conn.execute(
            "SELECT path FROM scan_root WHERE enabled = 1"
        ).fetchall()
        for r in roots:
            root = Path(r["path"])
            if not root.is_dir():
                continue
            handler = _RootHandler(root, self)
            try:
                self._observer.schedule(handler, str(root), recursive=True)
                log.info("watching %s", root)
            except Exception as exc:
                log.warning("could not watch %s: %s", root, exc)

    def _dispatch_loop(self) -> None:
        """Pick up debounced root-scan tasks once they're due."""
        while not self._stop.is_set():
            time.sleep(1.0)
            now = time.monotonic()
            due: list[Path] = []
            with self._pending_lock:
                for root, when in list(self._pending.items()):
                    if when <= now:
                        due.append(root)
                        del self._pending[root]
            for root in due:
                log.info("watcher fired scan for %s", root)
                try:
                    scan_root(root)
                except Exception:
                    log.exception("watcher scan failed for %s", root)


_singleton: FolderWatchService | None = None


def get_watcher() -> FolderWatchService:
    global _singleton
    if _singleton is None:
        _singleton = FolderWatchService()
    return _singleton
