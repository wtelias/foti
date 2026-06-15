"""Importer for an external ``photos.db`` (qwen2.5vl-tagged corpus).

Reads a sibling SQLite catalog (the user's pre-existing vision-LLM pipeline at
``/mnt/data/photo-ingest/photos.db``) and folds its tagged rows into Foti's
``photo`` + ``photo_tag_auto`` tables. No re-decode of the source images is
needed — we trust the upstream EXIF / phash / dimensions. Thumbnails and CLIP
embeddings are generated lazily later (thumbs on first view, embeddings via
a backfill scan).

Idempotent: paths already in the Foti catalog are skipped. Captions land in
``photo.caption`` (added as an incremental column); per-tag rows land in
``photo_tag_auto`` with score 1.0 (qwen-confirmed).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .db import connect, transaction

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Source-DB helpers
# ----------------------------------------------------------------------


def _connect_source(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    src = sqlite3.connect(uri, uri=True, check_same_thread=False)
    src.row_factory = sqlite3.Row
    return src


def _count_tagged(src: sqlite3.Connection) -> int:
    row = src.execute("SELECT COUNT(*) FROM files WHERE tagged_at IS NOT NULL").fetchone()
    return int(row[0])


def _iter_rows(src: sqlite3.Connection, batch: int = 500) -> Iterable[sqlite3.Row]:
    """Stream tagged rows from the source DB ordered by id."""
    last_id = 0
    while True:
        chunk = src.execute(
            """
            SELECT id, path, source, size, mtime, ext, sha256, phash, dhash,
                   width, height, exif_dt, caption, tags_json, model, cluster_id
            FROM files
            WHERE tagged_at IS NOT NULL AND id > ?
            ORDER BY id
            LIMIT ?
            """,
            (last_id, batch),
        ).fetchall()
        if not chunk:
            return
        for row in chunk:
            yield row
        last_id = chunk[-1]["id"]


# ----------------------------------------------------------------------
# Field conversion
# ----------------------------------------------------------------------


def _exif_dt_to_iso(raw: str | None) -> str | None:
    """Convert EXIF DateTime ("YYYY:MM:DD HH:MM:SS") → ISO-8601 UTC string.

    Returns ``None`` for unparseable input.
    """
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return None


def _parse_tags(tags_json: str | None) -> list[str]:
    if not tags_json:
        return []
    try:
        data = json.loads(tags_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for t in data:
        if not isinstance(t, str):
            continue
        s = t.strip().lower()
        if not s or s in seen:
            continue
        # Drop obvious noise: notes-style fragments, runaway captions, JSON debris.
        if s.startswith((":", "//", "[", "/", "\\", "{")) or s.endswith((":",)):
            continue
        if len(s) > 40 or "\n" in s:
            continue
        seen.add(s)
        out.append(s)
    return out


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def import_from_photos_db(
    src_path: Path | str,
    *,
    limit: int | None = None,
    check_files_exist: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Import tagged rows from ``src_path`` into the Foti catalog.

    Args:
        src_path: path to the source ``photos.db``.
        limit: stop after importing this many *new* photos (None = all).
        check_files_exist: when True, ``stat()`` every source path and skip
            ones that no longer exist on disk. Slow on large NFS mounts.
        progress: callback ``(done, total) -> None`` invoked every 200 rows.

    Returns a summary dict.
    """
    src_path = Path(src_path).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(src_path)

    src = _connect_source(src_path)
    total_tagged = _count_tagged(src)
    log.info("photos.db import: source=%s tagged_total=%d", src_path, total_tagged)

    conn = connect()
    provenance = f"photos.db:{src_path.name}"
    now_iso = datetime.now(timezone.utc).isoformat()

    summary = {
        "source": str(src_path),
        "tagged_total": total_tagged,
        "inserted": 0,
        "skipped_existing": 0,
        "skipped_missing_file": 0,
        "skipped_no_path": 0,
        "tags_written": 0,
        "errors": 0,
    }
    started = time.time()

    # Pre-load existing paths to make the dedup check cheap (a set lookup
    # instead of a per-row SELECT). For 95k imports this is the difference
    # between ~30 s and several minutes.
    existing = {
        r[0] for r in conn.execute("SELECT path FROM photo").fetchall()
    }
    log.info("photos.db import: %d photos already in foti catalog", len(existing))

    pending_photos: list[tuple] = []
    pending_tags: list[tuple] = []  # (path, label, score)
    chunk_size = 500
    processed = 0

    def _flush() -> None:
        if not pending_photos:
            return
        with transaction(conn):
            for photo_row in pending_photos:
                conn.execute(
                    """
                    INSERT INTO photo (
                        path, mtime, size_bytes, sha256, width, height,
                        captured_at, indexed_at, phash, caption, imported_from
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    photo_row,
                )
            # Resolve path -> photo_id for the tag inserts. Doing it in one
            # SELECT per chunk is cheap; the path index is unique.
            paths = [p[0] for p in pending_photos]
            placeholders = ",".join("?" for _ in paths)
            id_map = {
                r["path"]: r["id"]
                for r in conn.execute(
                    f"SELECT id, path FROM photo WHERE path IN ({placeholders})",
                    tuple(paths),
                ).fetchall()
            }
            for path, label, score in pending_tags:
                pid = id_map.get(path)
                if pid is None:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO photo_tag_auto (photo_id, label, score)
                    VALUES (?, ?, ?)
                    """,
                    (pid, label, score),
                )
        pending_photos.clear()
        pending_tags.clear()

    for row in _iter_rows(src, batch=chunk_size):
        processed += 1

        path = (row["path"] or "").strip()
        if not path:
            summary["skipped_no_path"] += 1
            continue
        if path in existing:
            summary["skipped_existing"] += 1
            continue
        if check_files_exist and not Path(path).is_file():
            summary["skipped_missing_file"] += 1
            continue

        try:
            captured = _exif_dt_to_iso(row["exif_dt"])
            tags = _parse_tags(row["tags_json"])
            caption = (row["caption"] or "").strip() or None

            pending_photos.append((
                path,
                float(row["mtime"] or 0.0),
                int(row["size"] or 0),
                row["sha256"],
                row["width"],
                row["height"],
                captured,
                now_iso,
                row["phash"],
                caption,
                provenance,
            ))
            for tag in tags[:12]:  # keep top-12 to avoid runaway-cardinality rows
                pending_tags.append((path, tag, 1.0))
            existing.add(path)
            summary["inserted"] += 1
            summary["tags_written"] += min(len(tags), 12)
        except Exception:
            log.exception("import failed for src row id=%s path=%s", row["id"], path)
            summary["errors"] += 1
            continue

        if len(pending_photos) >= chunk_size:
            _flush()
            if progress is not None:
                progress(summary["inserted"], total_tagged)
            if limit is not None and summary["inserted"] >= limit:
                break

    _flush()
    if progress is not None:
        progress(summary["inserted"], total_tagged)

    summary["elapsed_sec"] = round(time.time() - started, 1)
    log.info("photos.db import done: %s", summary)
    return summary
