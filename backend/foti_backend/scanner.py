"""Folder scan + embedding pipeline.

Walk a scan root, find new/changed photos, extract EXIF + thumbnail, run
CLIP, upsert into the catalog. Designed to be re-run cheaply: photos that
haven't changed since their last ``indexed_at`` are skipped.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import exifread
import imagehash
from PIL import ImageOps

from .colors import extract_palette
from .config import Settings, get_settings
from .db import EMBEDDING_DIM, connect, transaction
from .embedder import EMBEDDING_VER, get_embedder
from .imageio import open_image

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class Candidate:
    path: Path
    mtime: float
    size_bytes: int


def _iter_files(root: Path, extensions: tuple[str, ...]) -> Iterator[Candidate]:
    """Yield candidate photo files under root in deterministic (alpha) order."""
    norm_ext = {e.lower() for e in extensions}
    for entry in sorted(root.rglob("*")):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in norm_ext:
            continue
        try:
            st = entry.stat()
        except OSError:
            continue
        yield Candidate(path=entry.resolve(), mtime=st.st_mtime, size_bytes=st.st_size)


def _needs_index(conn: sqlite3.Connection, c: Candidate) -> bool:
    row = conn.execute(
        """
        SELECT mtime, size_bytes, embedding_ver, phash, dominant_colors
        FROM photo WHERE path = ?
        """,
        (str(c.path),),
    ).fetchone()
    if row is None:
        return True
    if row["mtime"] != c.mtime or row["size_bytes"] != c.size_bytes:
        return True
    if (row["embedding_ver"] or 0) < EMBEDDING_VER:
        return True
    if not row["phash"]:
        return True
    if not row["dominant_colors"]:
        return True
    return False


def _read_exif(path: Path) -> tuple[str | None, str | None]:
    """Return (exif_json, captured_at_iso)."""
    try:
        with path.open("rb") as fh:
            tags = exifread.process_file(fh, details=False, stop_tag="EXIF DateTimeOriginal")
    except Exception:
        return None, None

    if not tags:
        return None, None

    flat = {k: str(v) for k, v in tags.items() if not k.startswith("JPEGThumbnail")}
    captured = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
    captured_iso = None
    if captured is not None:
        try:
            dt = datetime.strptime(str(captured), "%Y:%m:%d %H:%M:%S")
            captured_iso = dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            captured_iso = None

    return json.dumps(flat), captured_iso


def _write_thumbs_and_features(path: Path, photo_id: int,
                                settings: Settings) -> tuple[Path, Path, str, str]:
    """Write thumbs, compute pHash + dominant-color palette in one image-open.

    Returns ``(small_path, large_path, phash_hex, colors_json)``.
    """
    small = settings.thumbs_dir / f"{photo_id}_s.webp"
    large = settings.thumbs_dir / f"{photo_id}_l.webp"

    img = open_image(path)
    img = ImageOps.exif_transpose(img)

    # pHash on the upright RGB pixels — robust to scale, JPEG re-encoding.
    phash_hex = str(imagehash.phash(img))

    # Dominant-color palette (top 5).
    palette = extract_palette(img)
    colors_json = json.dumps(palette)

    small_img = img.copy()
    small_img.thumbnail((settings.thumbnail_size_small, settings.thumbnail_size_small))
    small_img.save(small, format="WEBP", quality=80, method=4)

    large_img = img.copy()
    large_img.thumbnail((settings.thumbnail_size_large, settings.thumbnail_size_large))
    large_img.save(large, format="WEBP", quality=85, method=4)
    return small, large, phash_hex, colors_json


def _sha256_head(path: Path, n: int = 65536) -> str:
    """Cheap fingerprint: SHA256 of the first 64 KB. Full hash is overkill
    for change-detection; mtime+size handles that. This is for duplicate
    candidate clustering later."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(n))
    return h.hexdigest()


def _upsert_photo(conn: sqlite3.Connection, c: Candidate, *, embedding) -> int:
    settings = get_settings()
    exif_json, captured_at = _read_exif(c.path)

    try:
        with open_image(c.path) as img:
            width, height = img.size
            try:
                orientation = img.getexif().get(0x0112)
            except Exception:
                orientation = None
    except Exception as exc:
        log.warning("dimension probe failed for %s: %s", c.path, exc)
        width, height, orientation = None, None, None

    now_iso = datetime.now(timezone.utc).isoformat()
    sha_head = _sha256_head(c.path)

    cur = conn.execute("SELECT id FROM photo WHERE path = ?", (str(c.path),))
    existing = cur.fetchone()

    if existing is None:
        conn.execute(
            """
            INSERT INTO photo (path, mtime, size_bytes, sha256, width, height,
                               orientation, exif_json, captured_at, indexed_at,
                               embedding_ver)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(c.path), c.mtime, c.size_bytes, sha_head, width, height,
             orientation, exif_json, captured_at, now_iso, EMBEDDING_VER),
        )
        photo_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    else:
        photo_id = existing["id"]
        conn.execute(
            """
            UPDATE photo SET mtime=?, size_bytes=?, sha256=?, width=?, height=?,
                             orientation=?, exif_json=?, captured_at=?,
                             indexed_at=?, embedding_ver=?
            WHERE id = ?
            """,
            (c.mtime, c.size_bytes, sha_head, width, height, orientation,
             exif_json, captured_at, now_iso, EMBEDDING_VER, photo_id),
        )

    small, large, phash, colors_json = _write_thumbs_and_features(
        c.path, photo_id, settings
    )
    conn.execute(
        "UPDATE photo SET thumb_small=?, thumb_large=?, phash=?, dominant_colors=? WHERE id=?",
        (str(small), str(large), phash, colors_json, photo_id),
    )

    blob = embedding.astype("float32").tobytes()
    conn.execute(
        "INSERT OR REPLACE INTO photo_embedding(photo_id, embedding) VALUES (?, ?)",
        (photo_id, blob),
    )
    return photo_id


def scan_root(root: Path, *, progress=None) -> dict:
    """Scan a single folder root. Returns a summary dict."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")

    settings = get_settings()
    embedder = get_embedder()
    conn = connect()

    candidates = list(_iter_files(root, settings.scan_extensions))
    log.info("scan %s — %d candidates", root, len(candidates))

    todo: list[Candidate] = []
    for c in candidates:
        if _needs_index(conn, c):
            todo.append(c)

    log.info("scan %s — %d need indexing", root, len(todo))

    summary = {"root": str(root), "candidates": len(candidates), "indexed": 0, "errors": 0}
    started = time.time()

    for i in range(0, len(todo), embedder.batch_size):
        batch = todo[i : i + embedder.batch_size]
        try:
            embeddings = embedder.encode_images([c.path for c in batch])
        except Exception as exc:
            log.exception("batch embed failed: %s", exc)
            summary["errors"] += len(batch)
            continue

        with transaction(conn):
            for c, emb in zip(batch, embeddings):
                try:
                    _upsert_photo(conn, c, embedding=emb)
                    summary["indexed"] += 1
                except Exception:
                    log.exception("upsert failed for %s", c.path)
                    summary["errors"] += 1

        if progress:
            progress(summary["indexed"], len(todo))

    summary["elapsed_sec"] = time.time() - started
    log.info("scan %s done: %s", root, summary)
    return summary
