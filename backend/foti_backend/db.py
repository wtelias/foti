"""SQLite catalog with sqlite-vec for vector search.

The schema lives in ``schema.sql`` and is applied idempotently on connect.
sqlite-vec is loaded as an extension; vector tables are created via the
``vec0`` virtual-table module.

Connections are NOT shared across threads — each worker calls
:func:`connect` to get its own. Foreign keys and WAL mode are enabled.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import sqlite_vec

from .config import get_settings

# CLIP ViT-L/14 produces 768-dim embeddings, normalized to unit length so
# that cosine = 1 - 0.5 * L2_squared.
EMBEDDING_DIM = 768

FACE_EMBEDDING_DIM = 512  # InsightFace ArcFace output

VEC_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS photo_embedding USING vec0(
    photo_id INTEGER PRIMARY KEY,
    embedding FLOAT[{EMBEDDING_DIM}]
);

CREATE VIRTUAL TABLE IF NOT EXISTS face_embedding USING vec0(
    face_id INTEGER PRIMARY KEY,
    embedding FLOAT[{FACE_EMBEDDING_DIM}]
);
"""

# Columns that may be added by a later schema version. Migrating from older
# catalogs by attempting to add and ignoring the "duplicate column" error.
_INCREMENTAL_COLUMNS = [
    ("photo", "phash TEXT"),
    ("photo", "face_count INTEGER"),
    ("photo", "aesthetic REAL"),
    ("photo", "dominant_colors TEXT"),  # JSON array of {r,g,b,weight}
    ("photo", "caption TEXT"),          # qwen2.5vl long-form caption (from photos.db import)
    ("photo", "imported_from TEXT"),    # provenance tag, e.g. "photos.db:macphoto"
]


def _apply_incremental_migrations(conn: "sqlite3.Connection") -> None:
    for table, column_def in _INCREMENTAL_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
    conn.commit()


def _load_schema(conn: sqlite3.Connection) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    conn.executescript(schema_path.read_text())
    conn.executescript(VEC_SCHEMA)
    _apply_incremental_migrations(conn)
    conn.commit()


def connect(path: Path | None = None, *, read_only: bool = False) -> sqlite3.Connection:
    settings = get_settings()
    db_path = path or settings.catalog_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    uri = f"file:{db_path}?mode={'ro' if read_only else 'rwc'}"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Writers must wait for a held write lock instead of failing instantly —
    # the face indexer, backfills, and API mutations all write concurrently.
    conn.execute("PRAGMA busy_timeout = 5000")
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        _load_schema(conn)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Wrap a unit of work in a transaction with automatic rollback on error."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return int(row["value"]) if row else 0
