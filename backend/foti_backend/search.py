"""Search over the catalog: text-to-image and image-to-image.

Both routes encode a query into a 768-dim CLIP embedding and rank photos
by cosine distance via sqlite-vec's ``vec_distance_cosine``.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .db import connect
from .embedder import get_embedder

log = logging.getLogger(__name__)


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    # captured_at + indexed_at are ISO strings already; nothing to coerce.
    return d


def _rank_by_embedding(conn: sqlite3.Connection, query_vec: np.ndarray, limit: int) -> list[dict]:
    """Run a cosine-distance search and join back to photo metadata."""
    blob = query_vec.astype("float32").tobytes()
    rows = conn.execute(
        """
        SELECT
            p.id, p.path, p.captured_at, p.width, p.height,
            p.thumb_small, p.thumb_large,
            e.distance AS distance
        FROM photo_embedding e
        JOIN photo p ON p.id = e.photo_id
        WHERE e.embedding MATCH ?
          AND k = ?
        ORDER BY distance
        """,
        (blob, limit),
    ).fetchall()

    results = []
    for r in rows:
        d = _row_to_dict(r)
        # sqlite-vec cosine "distance" is 1 - cosine_similarity in [0, 2].
        # Surface a friendlier "score" in [0, 1].
        d["score"] = max(0.0, 1.0 - 0.5 * d["distance"])
        results.append(d)
    return results


def search_text(query: str, limit: int = 50) -> list[dict]:
    if not query.strip():
        return []
    embedder = get_embedder()
    vec = embedder.encode_text([query])[0]
    conn = connect(read_only=False)  # vec0 requires writable conn for MATCH
    return _rank_by_embedding(conn, vec, limit)


def search_similar(photo_id: int, limit: int = 50) -> list[dict]:
    conn = connect()
    row = conn.execute(
        "SELECT embedding FROM photo_embedding WHERE photo_id = ?",
        (photo_id,),
    ).fetchone()
    if row is None:
        return []
    vec = np.frombuffer(row["embedding"], dtype=np.float32)
    results = _rank_by_embedding(conn, vec, limit + 1)
    # Drop the query photo itself from the results.
    return [r for r in results if r["id"] != photo_id][:limit]


def search_by_image_file(path: Path, limit: int = 50) -> list[dict]:
    embedder = get_embedder()
    vec = embedder.encode_images([path])[0]
    conn = connect()
    return _rank_by_embedding(conn, vec, limit)
