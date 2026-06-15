"""Colors backfill: thumb-preferred decode, fallback, failure marking, API."""

import json

from foti_backend.config import get_settings

from tests.conftest import make_image, seed_photo


def test_backfill_prefers_thumb(catalog, tmp_path):
    """T-04: when a thumb exists, the ORIGINAL file must not be touched."""
    thumbs = get_settings().thumbs_dir
    make_image(thumbs / "t1.webp", color=(0, 200, 0))
    # Original path deliberately does NOT exist — only the thumb does.
    pid = seed_photo(catalog, path="/nonexistent/orig1.jpg", thumb_small="t1.webp")

    from foti_backend.colors import backfill_colors

    summary = backfill_colors(batch_size=10)
    assert summary["colored"] == 1
    assert summary["failed"] == 0

    row = catalog.execute("SELECT dominant_colors FROM photo WHERE id=?", (pid,)).fetchone()
    palette = json.loads(row["dominant_colors"])
    assert palette, "palette must be non-empty"
    # Solid green thumb → dominant color is green.
    top = palette[0]
    assert top["g"] > 150 and top["r"] < 80 and top["b"] < 80


def test_backfill_falls_back_to_original(catalog, tmp_path):
    orig = make_image(tmp_path / "orig2.jpg", color=(0, 0, 220))
    pid = seed_photo(catalog, path=str(orig))

    from foti_backend.colors import backfill_colors

    summary = backfill_colors(batch_size=10)
    assert summary["colored"] == 1
    row = catalog.execute("SELECT dominant_colors FROM photo WHERE id=?", (pid,)).fetchone()
    assert json.loads(row["dominant_colors"])[0]["b"] > 150


def test_backfill_marks_unreadable_as_empty(catalog):
    """A photo with no thumb and a missing original gets '[]', not retried forever."""
    pid = seed_photo(catalog, path="/gone/missing.jpg")

    from foti_backend.colors import backfill_colors

    s1 = backfill_colors(batch_size=10)
    assert s1["failed"] == 1
    row = catalog.execute("SELECT dominant_colors FROM photo WHERE id=?", (pid,)).fetchone()
    assert row["dominant_colors"] == "[]"
    # Second pass finds nothing left to do.
    s2 = backfill_colors(batch_size=10)
    assert s2["colored"] == 0 and s2["failed"] == 0 and s2["remaining"] == 0


def test_backfill_skips_already_colored(catalog, tmp_path):
    orig = make_image(tmp_path / "o3.jpg")
    seed_photo(catalog, path=str(orig), dominant_colors='[{"r":1,"g":2,"b":3,"weight":1.0}]')

    from foti_backend.colors import backfill_colors

    assert backfill_colors(batch_size=10)["colored"] == 0


def test_colors_backfill_endpoint(client, catalog, tmp_path):
    orig = make_image(tmp_path / "o4.jpg", color=(250, 250, 0))
    seed_photo(catalog, path=str(orig))

    r = client.post("/colors/backfill?batch_size=10")
    assert r.status_code == 200
    body = r.json()
    assert body["colored"] == 1
    assert body["remaining"] == 0
