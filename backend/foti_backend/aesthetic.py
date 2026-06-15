"""Lightweight aesthetic scoring using CLIP zero-shot.

We don't ship a separate aesthetic head (NIMA, LAION-aesthetic-predictor v2,
etc.) for the MVP. Instead, the existing CLIP encoder rates each photo
against two prompt sets:

    positive prompts ≈ what an appealing photo looks like
    negative prompts ≈ what a snapshot you'd archive looks like

The score is ``cos(img, mean(pos)) - cos(img, mean(neg))`` rescaled to a
1..10 range that matches NIMA expectations. Crude but free and useful.

Re-rank with a stronger head later (LAION V2 MLP, NIMA ONNX); the
``aesthetic`` column on photo is the single migration handle.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np

from .db import connect, transaction
from .embedder import get_embedder

log = logging.getLogger(__name__)

POSITIVE_PROMPTS = [
    "a stunning, well-composed, professionally lit photograph",
    "an emotionally evocative, sharp, magazine-worthy photo",
    "an artistic photo with great color, depth, and composition",
]

NEGATIVE_PROMPTS = [
    "a blurry out-of-focus snapshot",
    "an overexposed, poorly composed amateur photo",
    "a low-quality phone screenshot",
]


def _score_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Return (pos_centroid, neg_centroid) — both 1×768, L2-normalized.

    Computed once per process; the prompts are constants.
    """
    embedder = get_embedder()
    pos = embedder.encode_text(POSITIVE_PROMPTS).mean(axis=0)
    pos /= np.linalg.norm(pos) + 1e-12
    neg = embedder.encode_text(NEGATIVE_PROMPTS).mean(axis=0)
    neg /= np.linalg.norm(neg) + 1e-12
    return pos.astype("float32"), neg.astype("float32")


_pos: np.ndarray | None = None
_neg: np.ndarray | None = None


def _vectors() -> tuple[np.ndarray, np.ndarray]:
    global _pos, _neg
    if _pos is None or _neg is None:
        _pos, _neg = _score_vectors()
    return _pos, _neg


def _scale(raw: float) -> float:
    """Map raw cosine-diff (roughly -0.15..0.25) onto 1..10 with sigmoid."""
    # cosine-diff is well centered around 0 for everyday photos. Map to ~5.5
    # midpoint, 0.05 difference ≈ 1 point.
    return float(round(max(1.0, min(10.0, 5.5 + 30 * raw)), 2))


def score_catalog(batch_size: int = 500) -> dict:
    """Score photos that don't yet have an aesthetic. Returns summary."""
    pos, neg = _vectors()
    conn = connect()
    rows = conn.execute(
        """
        SELECT p.id, e.embedding
        FROM photo p
        JOIN photo_embedding e ON e.photo_id = p.id
        WHERE p.aesthetic IS NULL
        LIMIT ?
        """,
        (batch_size,),
    ).fetchall()
    summary = {"scored": 0, "skipped": 0}
    if not rows:
        return summary

    embs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    pos_sim = embs @ pos
    neg_sim = embs @ neg
    raw = pos_sim - neg_sim
    scaled = [_scale(float(r)) for r in raw]

    with transaction(conn):
        for r, s in zip(rows, scaled):
            conn.execute("UPDATE photo SET aesthetic = ? WHERE id = ?", (s, r["id"]))
            summary["scored"] += 1
    return summary
