"""Face indexing pipeline.

Runs detection over every photo that doesn't yet have a face_detection row,
writes detections + embeddings, and clusters by simple greedy nearest-
centroid assignment (cosine ≥ ``cluster_threshold``). The clustering is
deliberately simple: no sklearn dependency, online (one pass), and easy
for the UI to invalidate by row.

If you ever swap in HDBSCAN/Agglomerative for higher-quality clusters,
keep the per-detection embedding row — re-clustering is one query away.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .db import connect, transaction
from .faces import get_face_model

log = logging.getLogger(__name__)

# Cosine ≥ 0.55 ≈ same person at decent quality for ArcFace embeddings.
# Lower → more aggressive merging (false positives), higher → more splits.
CLUSTER_THRESHOLD = 0.55


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unindexed_photos(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """Photos that have no face row yet AND are likely to contain faces.

    "No face row" includes both "we haven't looked" and "we looked and
    found nothing" — distinguishing those requires a sentinel; for now
    we just look at face_count NULL.
    """
    return conn.execute(
        """
        SELECT p.id, p.path
        FROM photo p
        WHERE p.face_count IS NULL
        ORDER BY p.id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def _existing_centroids(conn: sqlite3.Connection) -> dict[int, np.ndarray]:
    """Average embedding per cluster_id. Computed once per pipeline run."""
    rows = conn.execute(
        """
        SELECT fd.cluster_id, fe.embedding
        FROM face_detection fd
        JOIN face_embedding fe ON fe.face_id = fd.id
        WHERE fd.cluster_id IS NOT NULL
        """
    ).fetchall()
    buckets: dict[int, list[np.ndarray]] = {}
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        buckets.setdefault(r["cluster_id"], []).append(emb)
    return {cid: np.mean(np.stack(es), axis=0) for cid, es in buckets.items()}


def _max_cluster_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(cluster_id) AS m FROM face_detection").fetchone()
    return int(row["m"]) if row and row["m"] is not None else 0


def _assign_cluster(emb: np.ndarray, centroids: dict[int, np.ndarray],
                    next_id: int) -> tuple[int, dict[int, np.ndarray], int]:
    """Greedy nearest-centroid assignment. Returns (cluster_id, updated_centroids, next_id)."""
    if not centroids:
        centroids = {next_id: emb}
        return next_id, centroids, next_id + 1

    cluster_ids = list(centroids.keys())
    cents = np.stack([centroids[c] for c in cluster_ids])
    sims = cents @ emb  # both L2-normalized
    best_idx = int(np.argmax(sims))
    best_cid = cluster_ids[best_idx]
    if float(sims[best_idx]) >= CLUSTER_THRESHOLD:
        # Online centroid update: streaming average is fine for our scale.
        centroids[best_cid] = (centroids[best_cid] + emb) / 2.0
        # Re-normalize so subsequent dots stay cosine-meaningful.
        n = np.linalg.norm(centroids[best_cid])
        if n > 0:
            centroids[best_cid] = centroids[best_cid] / n
        return best_cid, centroids, next_id

    centroids[next_id] = emb
    return next_id, centroids, next_id + 1


def index_faces(batch_size: int = 200) -> dict:
    """Index faces for up to ``batch_size`` unindexed photos. Returns summary."""
    fm = get_face_model()
    conn = connect()

    photos = _unindexed_photos(conn, batch_size)
    centroids = _existing_centroids(conn)
    next_cid = _max_cluster_id(conn) + 1

    summary = {"photos_processed": 0, "faces_found": 0, "clusters_created": 0}

    for p in photos:
        detections = fm.detect(Path(p["path"]))
        with transaction(conn):
            for d in detections:
                bbox_json = json.dumps(d["bbox"])
                cur = conn.execute(
                    """
                    INSERT INTO face_detection (photo_id, bbox_json, det_score, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (p["id"], bbox_json, d["score"], _now()),
                )
                face_id = cur.lastrowid

                conn.execute(
                    "INSERT OR REPLACE INTO face_embedding(face_id, embedding) VALUES (?, ?)",
                    (face_id, d["embedding"].tobytes()),
                )

                cluster_id, centroids, new_next = _assign_cluster(
                    d["embedding"], centroids, next_cid
                )
                if new_next != next_cid:
                    summary["clusters_created"] += 1
                next_cid = new_next
                conn.execute(
                    "UPDATE face_detection SET cluster_id = ? WHERE id = ?",
                    (cluster_id, face_id),
                )

            conn.execute(
                "UPDATE photo SET face_count = ? WHERE id = ?",
                (len(detections), p["id"]),
            )
            summary["faces_found"] += len(detections)
            summary["photos_processed"] += 1

    return summary


def list_people(min_size: int = 2, limit: int = 100) -> list[dict]:
    """Return clusters sorted by size descending, each with one cover thumb id."""
    conn = connect()
    rows = conn.execute(
        """
        SELECT fd.cluster_id,
               COUNT(*) AS face_count,
               COUNT(DISTINCT fd.photo_id) AS photo_count,
               p.name AS person_name,
               p.id AS person_id,
               MIN(fd.id) AS sample_face_id,
               MIN(fd.photo_id) AS sample_photo_id
        FROM face_detection fd
        LEFT JOIN person p ON p.id = fd.person_id
        WHERE fd.cluster_id IS NOT NULL
        GROUP BY fd.cluster_id, p.id, p.name
        HAVING face_count >= ?
        ORDER BY face_count DESC
        LIMIT ?
        """,
        (min_size, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def list_photos_in_cluster(cluster_id: int, limit: int = 200) -> list[dict]:
    conn = connect()
    rows = conn.execute(
        """
        SELECT DISTINCT p.id, p.path, p.thumb_small, p.thumb_large,
               p.captured_at, p.width, p.height
        FROM face_detection fd
        JOIN photo p ON p.id = fd.photo_id
        WHERE fd.cluster_id = ?
        ORDER BY COALESCE(p.captured_at, p.indexed_at) DESC
        LIMIT ?
        """,
        (cluster_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]
