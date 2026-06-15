"""Auto-tagging via CLIP zero-shot.

We define a fixed vocabulary of ~180 visual concepts (people, places,
times, objects, activities, abstract qualities). For each photo we
compute cosine similarity between its CLIP image embedding and each
concept's text embedding, and keep the top-K tags above a threshold.

This runs on the embeddings already stored in ``photo_embedding`` — no
re-decode of the image. A backfill for 10k photos is a single matrix
multiply (~3 ms on GPU). Re-running with a different vocabulary just
re-runs the matmul.

Tag results live in their own table ``photo_tag_auto`` so the user's
hand-curated ``photo_tag`` table stays untouched.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from .db import connect, transaction
from .embedder import get_embedder

log = logging.getLogger(__name__)

# Vocabulary — kept compact on purpose. Add freely; re-running the
# backfill is cheap.
VOCAB: list[str] = [
    # people & faces
    "a portrait of a person", "a group photo of people", "a selfie",
    "a baby", "a child", "a wedding photo", "a candid photo of someone smiling",
    # places & landscapes
    "a beach", "a mountain", "a forest", "a desert", "a city skyline",
    "a city street", "a village", "the countryside", "a snowy landscape",
    "a sunset", "a sunrise", "a starry night sky", "fog or mist",
    "a body of water", "a river", "a lake", "the sea", "a waterfall",
    # interiors & buildings
    "an interior of a home", "a kitchen", "a bedroom", "a living room",
    "a restaurant interior", "a cafe", "a bar", "a church", "a museum",
    "a stage with performers", "a concert", "a stadium",
    "a building exterior", "modern architecture", "historic architecture",
    # objects & food
    "a plate of food", "a meal at a restaurant", "drinks or cocktails",
    "coffee", "a cake", "a fruit",
    "a car", "a motorcycle", "a bicycle", "a boat", "an airplane",
    "a train", "a road", "a bridge",
    "a flower", "flowers in a bouquet", "a tree", "a leaf close-up",
    "an animal", "a dog", "a cat", "a horse", "a bird", "wildlife",
    # documents & screens
    "a document or piece of paper", "a screenshot",
    "a handwritten note", "a printed photo", "a poster or sign",
    "a whiteboard", "a slide from a presentation",
    # activities
    "people at a party", "people working", "people playing sports",
    "people hiking outdoors", "people swimming", "people cycling",
    "people dancing",
    # abstract / qualitative
    "an artistic black and white photo", "an abstract close-up",
    "a long-exposure light trail", "an aerial drone shot",
    "a vintage photo", "a blurry or out-of-focus photo",
    # times of day
    "a photo taken during golden hour", "a photo taken at night",
    "a photo taken on a sunny day", "a photo taken on a rainy day",
    "a photo taken indoors with artificial light",
]

# Short labels for UI display; derived from the prompt by stripping leading "a "/"an ".
def _label(prompt: str) -> str:
    p = prompt.removeprefix("an ").removeprefix("a ").removeprefix("the ")
    return p

LABELS = [_label(p) for p in VOCAB]

TOP_K = 5
MIN_SIM = 0.18  # cosine threshold — below this, the match isn't meaningful


_concept_vecs: np.ndarray | None = None


def _concept_vectors() -> np.ndarray:
    """Embed the vocabulary once and cache it for the process."""
    global _concept_vecs
    if _concept_vecs is None:
        embedder = get_embedder()
        vecs = embedder.encode_text(VOCAB)
        # Already L2-normalized by embedder.encode_text.
        _concept_vecs = vecs.astype("float32")
    return _concept_vecs


def _ensure_tables(conn) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS photo_tag_auto (
            photo_id INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
            label    TEXT NOT NULL,
            score    REAL NOT NULL,
            PRIMARY KEY (photo_id, label)
        );
        CREATE INDEX IF NOT EXISTS photo_tag_auto_label_idx
            ON photo_tag_auto(label);
        """
    )
    conn.commit()


def tag_catalog(batch_size: int = 1000) -> dict:
    """Score every un-tagged photo against the vocabulary and persist the
    top-K matches above ``MIN_SIM``.
    """
    conn = connect()
    _ensure_tables(conn)
    rows = conn.execute(
        """
        SELECT p.id, e.embedding
        FROM photo p
        JOIN photo_embedding e ON e.photo_id = p.id
        WHERE NOT EXISTS (SELECT 1 FROM photo_tag_auto t WHERE t.photo_id = p.id)
        LIMIT ?
        """,
        (batch_size,),
    ).fetchall()
    summary = {"tagged": 0}
    if not rows:
        return summary

    embs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    concepts = _concept_vectors()
    sims = embs @ concepts.T  # (N, vocab_size)

    with transaction(conn):
        for row, sim in zip(rows, sims):
            topk_idx = np.argsort(sim)[-TOP_K:][::-1]
            for idx in topk_idx:
                score = float(sim[idx])
                if score < MIN_SIM:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO photo_tag_auto (photo_id, label, score) VALUES (?, ?, ?)",
                    (row["id"], LABELS[int(idx)], round(score, 4)),
                )
            summary["tagged"] += 1
    return summary


def top_labels(limit: int = 30) -> list[dict]:
    """Most-common auto-tags across the catalog. Used to build the sidebar facet."""
    conn = connect()
    _ensure_tables(conn)
    rows = conn.execute(
        """
        SELECT label, COUNT(*) AS n
        FROM photo_tag_auto
        GROUP BY label
        ORDER BY n DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [{"label": r["label"], "count": r["n"]} for r in rows]


def photos_for_label(label: str, limit: int = 200) -> list[dict]:
    conn = connect()
    rows = conn.execute(
        """
        SELECT p.id, p.path, p.captured_at, p.width, p.height,
               p.thumb_small, p.thumb_large, t.score AS score
        FROM photo_tag_auto t
        JOIN photo p ON p.id = t.photo_id
        WHERE t.label = ?
        ORDER BY t.score DESC
        LIMIT ?
        """,
        (label, limit),
    ).fetchall()
    return [dict(r) for r in rows]
