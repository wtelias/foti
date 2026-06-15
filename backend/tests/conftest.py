"""Shared test fixtures.

Every test runs against a throwaway catalog under a tmp dir. The FOTI_DATA_DIR
env var must be set BEFORE any foti_backend module resolves settings, so the
fixture clears the settings cache and re-imports nothing heavy: model loads
(CLIP / InsightFace) stay lazy and are never triggered here.

Basic auth: tests set FOTI_BASIC_USER/PASS before importing the api module so
the auth middleware is installed, mirroring production.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Auth env must exist before foti_backend.api is imported anywhere.
os.environ.setdefault("FOTI_BASIC_USER", "test")
os.environ.setdefault("FOTI_BASIC_PASS", "test-pass")

import pytest
from PIL import Image

TEST_AUTH = (os.environ["FOTI_BASIC_USER"], os.environ["FOTI_BASIC_PASS"])


@pytest.fixture()
def catalog(tmp_path, monkeypatch):
    """Fresh empty catalog in a tmp data dir. Yields the sqlite connection."""
    monkeypatch.setenv("FOTI_DATA_DIR", str(tmp_path / "foti-data"))

    from foti_backend.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    from foti_backend.db import connect

    conn = connect()
    yield conn
    conn.close()
    get_settings.cache_clear()


@pytest.fixture()
def client(catalog):
    """httpx TestClient over the API, sharing the catalog fixture's data dir."""
    from fastapi.testclient import TestClient

    from foti_backend import api

    with TestClient(api.app) as c:
        c.auth = TEST_AUTH
        yield c


def make_image(path: Path, color=(200, 30, 30), size=(64, 64)) -> Path:
    """Write a solid-color image usable as photo or thumb."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def seed_photo(conn, *, path: str, phash: str | None = None,
               thumb_small: str | None = None, dominant_colors: str | None = None,
               caption: str | None = None, aesthetic: float | None = None,
               captured_at: str | None = None) -> int:
    """Insert a minimal photo row, return its id."""
    cur = conn.execute(
        """
        INSERT INTO photo (path, mtime, size_bytes, indexed_at, phash,
                           thumb_small, dominant_colors, caption, aesthetic, captured_at)
        VALUES (?, 0, 0, '2026-01-01T00:00:00Z', ?, ?, ?, ?, ?, ?)
        """,
        (path, phash, thumb_small, dominant_colors, caption, aesthetic, captured_at),
    )
    conn.commit()
    return cur.lastrowid


def seed_face(conn, photo_id: int, cluster_id: int | None = None,
              person_id: int | None = None, embedding: bytes | None = None) -> int:
    """Insert a face_detection row (+ optional embedding), return face id."""
    cur = conn.execute(
        """
        INSERT INTO face_detection (photo_id, bbox_json, det_score, cluster_id,
                                    person_id, created_at)
        VALUES (?, '[0,0,10,10]', 0.9, ?, ?, '2026-01-01T00:00:00Z')
        """,
        (photo_id, cluster_id, person_id),
    )
    conn.commit()
    if embedding is not None:
        conn.execute(
            "INSERT OR REPLACE INTO face_embedding(face_id, embedding) VALUES (?, ?)",
            (cur.lastrowid, embedding),
        )
        conn.commit()
    return cur.lastrowid
