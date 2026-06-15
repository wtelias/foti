"""Dominant-color extraction + similarity search.

For each photo we extract the top-K palette colors from a downsampled
thumbnail using PIL's Median Cut quantizer, store them as a JSON array
of ``{r, g, b, weight}`` rows, and offer a search that ranks photos by
how close their dominant colors come to a query color.

This is intentionally not LAB-or-CIEDE-2000 — RGB Euclidean is good
enough for "find my teal photos" and stays fast at SQL scan speed.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image

from .db import connect, transaction

log = logging.getLogger(__name__)

PALETTE_K = 5          # how many dominant colors to extract per photo
SAMPLE_SIZE = 128      # downsample side for the quantizer


def extract_palette(image: Image.Image, k: int = PALETTE_K) -> list[dict]:
    """Return up to ``k`` dominant colors with their weights (0..1)."""
    img = image.convert("RGB")
    img.thumbnail((SAMPLE_SIZE, SAMPLE_SIZE))

    # Median Cut palette via PIL — produces a 256-entry palette with the
    # most-used color at index 0.
    quantized = img.quantize(colors=k, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()  # flat [r,g,b, r,g,b, ...]

    # Pixel→palette-index counts give us weight per color.
    counts = Counter(quantized.getdata())
    total = sum(counts.values()) or 1
    out = []
    for idx, count in counts.most_common(k):
        r = palette[idx * 3]
        g = palette[idx * 3 + 1]
        b = palette[idx * 3 + 2]
        out.append({"r": int(r), "g": int(g), "b": int(b),
                    "weight": round(count / total, 4)})
    return out


def backfill_colors(batch_size: int = 500) -> dict:
    """Fill ``dominant_colors`` for photos that don't have them yet.

    Decodes the small thumbnail (local SSD WebP) when present — the palette
    quantizer downsamples to 128px anyway, so a 256px thumb loses nothing —
    and only falls back to the original file path. This keeps the pass off
    the (slow, possibly fuse-mounted) source volumes.

    Call repeatedly until ``{"colored": 0}``.
    """
    from .config import get_settings

    thumbs_dir = get_settings().thumbs_dir
    conn = connect()
    rows = conn.execute(
        """
        SELECT id, path, thumb_small
        FROM photo
        WHERE dominant_colors IS NULL
        ORDER BY id
        LIMIT ?
        """,
        (batch_size,),
    ).fetchall()

    colored = 0
    failed = 0
    for row in rows:
        src: Path | None = None
        if row["thumb_small"]:
            cand = thumbs_dir / row["thumb_small"]
            if cand.is_file():
                src = cand
        if src is None:
            cand = Path(row["path"])
            if cand.is_file():
                src = cand
        # Decode OUTSIDE any transaction — the write lock must never be held
        # across image work (the face indexer commits concurrently).
        palette: list[dict] | None = None
        if src is not None:
            try:
                with Image.open(src) as img:
                    palette = extract_palette(img)
            except Exception as exc:
                log.warning("color backfill: decode failed for %s: %s", src, exc)
        with transaction(conn):
            if palette is None:
                # Nothing decodable; mark with empty palette so we don't
                # rescan it every batch. Distinguishable from NULL.
                conn.execute("UPDATE photo SET dominant_colors = '[]' WHERE id = ?",
                             (row["id"],))
                failed += 1
            else:
                conn.execute("UPDATE photo SET dominant_colors = ? WHERE id = ?",
                             (json.dumps(palette), row["id"]))
                colored += 1

    remaining = conn.execute(
        "SELECT COUNT(*) FROM photo WHERE dominant_colors IS NULL").fetchone()[0]
    return {"colored": colored, "failed": failed, "remaining": int(remaining)}


def _rgb_dist_sq(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> int:
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    return dr * dr + dg * dg + db * db


def search_by_color(rgb: tuple[int, int, int], tolerance: int = 60,
                    limit: int = 100) -> list[dict]:
    """Return photos whose dominant palette contains a color near ``rgb``.

    Score is ``weight / (1 + sqrt(distance))`` summed across matching palette
    rows. Photos with zero matches are dropped.
    """
    conn = connect()
    rows = conn.execute(
        """
        SELECT id, path, captured_at, width, height,
               thumb_small, thumb_large, dominant_colors
        FROM photo
        WHERE dominant_colors IS NOT NULL
        """
    ).fetchall()

    tol_sq = tolerance * tolerance
    results: list[tuple[float, dict]] = []
    for row in rows:
        try:
            palette = json.loads(row["dominant_colors"])
        except (TypeError, ValueError):
            continue
        score = 0.0
        for col in palette:
            dist_sq = _rgb_dist_sq(rgb, (col["r"], col["g"], col["b"]))
            if dist_sq <= tol_sq:
                score += col["weight"] / (1 + dist_sq ** 0.5)
        if score > 0:
            d = dict(row)
            d["score"] = score
            results.append((score, d))

    results.sort(key=lambda x: -x[0])
    return [d for _, d in results[:limit]]
