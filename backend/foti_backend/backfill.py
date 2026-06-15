"""CLIP-embedding backfill for photos that arrived via the photos.db
importer without an image-side pass.

Walks the ``photo`` table (NOT the filesystem) for rows missing a row
in ``photo_embedding`` or with ``embedding_ver`` < current. For each
such photo, opens the source file once and writes:

- L2-normalised CLIP embedding (always)
- dominant-colour palette (if ``with_colors`` and not yet stored)
- WebP thumbnails (if ``with_thumbs`` and not yet stored)

Designed to run as a long background job: progress is reported via
the same ``scan_job`` table, so the existing UI poll path works.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PIL import ImageOps

from .colors import extract_palette
from .config import get_settings
from .db import connect, transaction
from .embedder import EMBEDDING_VER, get_embedder
from .imageio import open_image

log = logging.getLogger(__name__)


def _candidate_count() -> int:
    conn = connect()
    return int(conn.execute(
        """
        SELECT COUNT(*) FROM photo p
        WHERE NOT EXISTS (SELECT 1 FROM photo_embedding e WHERE e.photo_id = p.id)
           OR COALESCE(p.embedding_ver, 0) < ?
        """,
        (EMBEDDING_VER,),
    ).fetchone()[0])


def _next_batch(limit: int) -> list[dict]:
    conn = connect()
    rows = conn.execute(
        """
        SELECT id, path, dominant_colors, thumb_small, thumb_large
        FROM photo p
        WHERE NOT EXISTS (SELECT 1 FROM photo_embedding e WHERE e.photo_id = p.id)
           OR COALESCE(p.embedding_ver, 0) < ?
        ORDER BY id
        LIMIT ?
        """,
        (EMBEDDING_VER, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def backfill_embeddings(
    *,
    limit: int | None = None,
    batch_size: int | None = None,
    with_thumbs: bool = True,
    with_colors: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Backfill CLIP embeddings for photos missing one.

    Args:
        limit: stop after processing this many photos (None = all).
        batch_size: override the embedder's default batch size.
        with_thumbs: also generate WebP thumbs when the photo has none.
        with_colors: also extract a dominant-colour palette when missing.
        progress: callback ``(done, total) -> None`` every batch.
    """
    embedder = get_embedder()
    bs = batch_size or embedder.batch_size
    settings = get_settings()

    total_initial = _candidate_count()
    log.info("backfill start: %d candidates (limit=%s, batch=%d, device=%s)",
             total_initial, limit, bs, embedder.device)

    summary = {
        "candidates_initial": total_initial,
        "encoded": 0,
        "thumbs_written": 0,
        "palettes_written": 0,
        "errors": 0,
    }
    started = time.time()

    while True:
        remaining = (limit - summary["encoded"]) if limit is not None else bs
        if remaining <= 0:
            break
        batch = _next_batch(min(bs, remaining))
        if not batch:
            break

        paths = [Path(p["path"]) for p in batch]

        # 1) Single image-open per photo: pre-render thumbs + palette if asked.
        #    The embedder will re-open via its own preprocess transform — we
        #    accept that double-open to keep the embedder API stable and avoid
        #    holding ~16 huge PIL Images in memory at once.
        side_payloads: list[dict] = []
        for row, path in zip(batch, paths):
            payload = {"phash": None, "colors_json": None,
                       "small_path": None, "large_path": None}
            try:
                if (with_colors and not row.get("dominant_colors")) or \
                   (with_thumbs and (not row.get("thumb_small") or
                                     not row.get("thumb_large"))):
                    img = open_image(path)
                    img = ImageOps.exif_transpose(img)
                    if with_colors and not row.get("dominant_colors"):
                        payload["colors_json"] = json.dumps(extract_palette(img))
                    if with_thumbs and not row.get("thumb_small"):
                        small = settings.thumbs_dir / f"{row['id']}_s.webp"
                        si = img.copy()
                        si.thumbnail((settings.thumbnail_size_small,
                                      settings.thumbnail_size_small))
                        si.save(small, format="WEBP", quality=80, method=4)
                        payload["small_path"] = str(small)
                    if with_thumbs and not row.get("thumb_large"):
                        large = settings.thumbs_dir / f"{row['id']}_l.webp"
                        li = img.copy()
                        li.thumbnail((settings.thumbnail_size_large,
                                      settings.thumbnail_size_large))
                        li.save(large, format="WEBP", quality=85, method=4)
                        payload["large_path"] = str(large)
            except Exception as exc:
                log.debug("side-pass skipped for %s: %s", path, exc)
            side_payloads.append(payload)

        # 2) CLIP encode (batched on GPU).
        try:
            embeddings = embedder.encode_images(paths)
        except Exception:
            log.exception("CLIP batch failed for %d photos starting %s",
                          len(paths), paths[0])
            summary["errors"] += len(paths)
            continue

        # 3) Persist embeddings + side-pass results.
        conn = connect()
        now_iso = datetime.now(timezone.utc).isoformat()
        with transaction(conn):
            for row, emb, side in zip(batch, embeddings, side_payloads):
                photo_id = row["id"]
                try:
                    blob = emb.astype("float32").tobytes()
                    conn.execute(
                        "INSERT OR REPLACE INTO photo_embedding(photo_id, embedding) VALUES (?, ?)",
                        (photo_id, blob),
                    )
                    conn.execute(
                        "UPDATE photo SET embedding_ver = ?, indexed_at = ? WHERE id = ?",
                        (EMBEDDING_VER, now_iso, photo_id),
                    )
                    if side["colors_json"]:
                        conn.execute(
                            "UPDATE photo SET dominant_colors = ? WHERE id = ?",
                            (side["colors_json"], photo_id),
                        )
                        summary["palettes_written"] += 1
                    if side["small_path"]:
                        conn.execute(
                            "UPDATE photo SET thumb_small = ? WHERE id = ?",
                            (side["small_path"], photo_id),
                        )
                        summary["thumbs_written"] += 1
                    if side["large_path"]:
                        conn.execute(
                            "UPDATE photo SET thumb_large = ? WHERE id = ?",
                            (side["large_path"], photo_id),
                        )
                    summary["encoded"] += 1
                except Exception:
                    log.exception("persist failed for photo %d", photo_id)
                    summary["errors"] += 1

        if progress is not None:
            progress(summary["encoded"], total_initial)

        if limit is not None and summary["encoded"] >= limit:
            break

    summary["elapsed_sec"] = round(time.time() - started, 1)
    if summary["encoded"]:
        summary["photos_per_sec"] = round(
            summary["encoded"] / max(summary["elapsed_sec"], 0.001), 2
        )
    log.info("backfill done: %s", summary)
    return summary
