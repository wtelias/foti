"""Smart Collections — saved queries.

A collection persists a "what to show" recipe so the user can return to
it from a sidebar. We re-use the existing ``collection`` table from
``schema.sql``, storing the query in ``query_json``:

    {
      "kind": "text" | "color" | "filter" | "similar" | "cluster",
      "q":      "...",                # for kind=text
      "hex":    "#7ab8ff",            # for kind=color
      "tolerance": 60,                # for kind=color
      "filters": {"year": "2024", "camera": "iPhone 14 Pro"},
      "photo_id": 42,                 # for kind=similar
      "cluster_id": 3,                # for kind=cluster
      "sort": "best",
    }

Resolution is done in-process by re-running the underlying search / list
query, so collections always reflect the live catalog (no stale denorm).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from .colors import search_by_color
from .db import connect, transaction
from .face_pipeline import list_photos_in_cluster
from .search import search_similar, search_text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_collections() -> list[dict]:
    conn = connect()
    rows = conn.execute(
        """
        SELECT id, name, is_smart, query_json, created_at
        FROM collection
        ORDER BY id DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def create_collection(name: str, query: dict) -> dict:
    conn = connect()
    with transaction(conn):
        cur = conn.execute(
            """
            INSERT INTO collection (name, is_smart, query_json, created_at)
            VALUES (?, 1, ?, ?)
            """,
            (name, json.dumps(query), _now()),
        )
        cid = cur.lastrowid
    row = conn.execute(
        "SELECT id, name, is_smart, query_json, created_at FROM collection WHERE id = ?",
        (cid,),
    ).fetchone()
    return dict(row)


def delete_collection(collection_id: int) -> int:
    conn = connect()
    with transaction(conn):
        cur = conn.execute("DELETE FROM collection WHERE id = ?", (collection_id,))
    return cur.rowcount


def resolve(collection_id: int, limit: int = 200) -> dict:
    """Re-run the collection's query and return the live results."""
    conn = connect()
    row = conn.execute(
        "SELECT name, query_json FROM collection WHERE id = ?", (collection_id,)
    ).fetchone()
    if row is None:
        return {"error": "unknown collection"}
    q = json.loads(row["query_json"])
    kind = q.get("kind", "filter")

    if kind == "text":
        results = search_text(q["q"], limit=limit)
    elif kind == "similar":
        results = search_similar(q["photo_id"], limit=limit)
    elif kind == "cluster":
        results = list_photos_in_cluster(q["cluster_id"], limit=limit)
    elif kind == "color":
        h = q["hex"].lstrip("#")
        rgb = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        results = search_by_color(rgb, tolerance=q.get("tolerance", 60), limit=limit)
    elif kind == "filter":
        # Reuse the photos list query directly.
        filters = q.get("filters", {})
        sort = q.get("sort", "newest")
        order_by = {
            "newest": "COALESCE(captured_at, indexed_at) DESC",
            "oldest": "COALESCE(captured_at, indexed_at) ASC",
            "best":   "aesthetic DESC NULLS LAST, COALESCE(captured_at, indexed_at) DESC",
            "worst":  "aesthetic ASC NULLS LAST, COALESCE(captured_at, indexed_at) DESC",
        }.get(sort, "COALESCE(captured_at, indexed_at) DESC")
        where = ["1 = 1"]
        params: list = []
        if filters.get("year"):
            where.append("substr(COALESCE(captured_at, indexed_at), 1, 4) = ?")
            params.append(str(filters["year"]))
        if filters.get("camera"):
            where.append("exif_json LIKE ?")
            params.append(f'%"Image Model": "{filters["camera"]}"%')
        rows = conn.execute(
            f"""
            SELECT id, path, captured_at, width, height, thumb_small, thumb_large, aesthetic
            FROM photo
            WHERE {' AND '.join(where)}
            ORDER BY {order_by}
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        results = [dict(r) for r in rows]
    else:
        results = []

    return {"name": row["name"], "kind": kind, "query": q, "results": results}
