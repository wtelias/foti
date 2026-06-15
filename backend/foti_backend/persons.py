"""Person management over face clusters.

A "person" is a named identity (``person`` table). Naming a cluster
creates-or-reuses the person and stamps ``person_id`` on every detection in
that cluster. Merging clusters re-points the source cluster's detections to
the target cluster (and its person, if any).

Every mutation appends a row to ``person_audit`` — merges rewrite cluster
ids, so the trail is the only way to answer "what happened to cluster 17?".
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

from .db import connect, transaction

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit(conn: sqlite3.Connection, action: str, detail: dict) -> None:
    conn.execute(
        "INSERT INTO person_audit (action, detail, created_at) VALUES (?, ?, ?)",
        (action, json.dumps(detail), _now()),
    )


def _cluster_exists(conn: sqlite3.Connection, cluster_id: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM face_detection WHERE cluster_id = ? LIMIT 1",
        (cluster_id,),
    ).fetchone() is not None


def name_cluster(cluster_id: int, name: str) -> dict:
    """Assign a (created-or-existing) person to every face in a cluster."""
    name = name.strip()
    if not name:
        raise ValueError("name must be non-empty")
    conn = connect()
    if not _cluster_exists(conn, cluster_id):
        raise KeyError(f"cluster {cluster_id} not found")
    with transaction(conn):
        row = conn.execute("SELECT id FROM person WHERE name = ?", (name,)).fetchone()
        if row:
            person_id = row["id"]
        else:
            cur = conn.execute(
                "INSERT INTO person (name, created_at) VALUES (?, ?)", (name, _now()))
            person_id = cur.lastrowid
        updated = conn.execute(
            "UPDATE face_detection SET person_id = ? WHERE cluster_id = ?",
            (person_id, cluster_id),
        ).rowcount
        _audit(conn, "name", {"cluster_id": cluster_id, "person_id": person_id,
                              "name": name, "faces": updated})
    return {"person_id": person_id, "name": name, "faces_updated": updated}


def rename_person(person_id: int, new_name: str) -> dict:
    new_name = new_name.strip()
    if not new_name:
        raise ValueError("name must be non-empty")
    conn = connect()
    row = conn.execute("SELECT name FROM person WHERE id = ?", (person_id,)).fetchone()
    if row is None:
        raise KeyError(f"person {person_id} not found")
    clash = conn.execute(
        "SELECT id FROM person WHERE name = ? AND id != ?", (new_name, person_id)).fetchone()
    if clash:
        raise ValueError(f"a person named {new_name!r} already exists (id {clash['id']})")
    with transaction(conn):
        conn.execute("UPDATE person SET name = ? WHERE id = ?", (new_name, person_id))
        _audit(conn, "rename", {"person_id": person_id,
                                "old_name": row["name"], "new_name": new_name})
    return {"person_id": person_id, "name": new_name}


def unname_person(person_id: int) -> dict:
    """Remove a named identity: detach it from every face and delete the person.

    The faces themselves (and their cluster) are untouched — only the name is
    removed, so the cluster reappears as an unnamed group in /faces/people.
    Excire parity: a person tag must be removable, not just renameable.
    """
    conn = connect()
    row = conn.execute("SELECT name FROM person WHERE id = ?", (person_id,)).fetchone()
    if row is None:
        raise KeyError(f"person {person_id} not found")
    with transaction(conn):
        detached = conn.execute(
            "UPDATE face_detection SET person_id = NULL WHERE person_id = ?",
            (person_id,),
        ).rowcount
        conn.execute("DELETE FROM person WHERE id = ?", (person_id,))
        _audit(conn, "unname", {"person_id": person_id, "name": row["name"],
                                "faces_detached": detached})
    return {"person_id": person_id, "name": row["name"], "faces_detached": detached}


def merge_clusters(source_cluster_id: int, target_cluster_id: int) -> dict:
    """Fold the source cluster into the target ("these are the same person").

    Detections move to the target cluster id and adopt the target's person
    (the target's identity wins; if only the source was named, the name
    carries over instead of being lost).
    """
    if source_cluster_id == target_cluster_id:
        raise ValueError("source and target cluster are the same")
    conn = connect()
    for cid in (source_cluster_id, target_cluster_id):
        if not _cluster_exists(conn, cid):
            raise KeyError(f"cluster {cid} not found")

    def _cluster_person(cid: int) -> int | None:
        row = conn.execute(
            """
            SELECT person_id FROM face_detection
            WHERE cluster_id = ? AND person_id IS NOT NULL LIMIT 1
            """,
            (cid,),
        ).fetchone()
        return row["person_id"] if row else None

    with transaction(conn):
        # Resolve persons INSIDE the transaction — a concurrent naming between
        # read and update must not be silently overwritten.
        target_person = _cluster_person(target_cluster_id)
        source_person = _cluster_person(source_cluster_id)
        final_person = target_person if target_person is not None else source_person

        moved = conn.execute(
            "UPDATE face_detection SET cluster_id = ? WHERE cluster_id = ?",
            (target_cluster_id, source_cluster_id),
        ).rowcount
        if final_person is not None:
            conn.execute(
                "UPDATE face_detection SET person_id = ? WHERE cluster_id = ?",
                (final_person, target_cluster_id),
            )
        _audit(conn, "merge", {"source_cluster_id": source_cluster_id,
                               "target_cluster_id": target_cluster_id,
                               "faces_moved": moved,
                               "person_id": final_person})
    return {"target_cluster_id": target_cluster_id, "faces_moved": moved,
            "person_id": final_person}


def list_persons() -> list[dict]:
    conn = connect()
    rows = conn.execute(
        """
        SELECT p.id, p.name, p.created_at,
               COUNT(fd.id) AS face_count,
               COUNT(DISTINCT fd.photo_id) AS photo_count
        FROM person p
        LEFT JOIN face_detection fd ON fd.person_id = p.id
        GROUP BY p.id
        ORDER BY p.name COLLATE NOCASE
        """
    ).fetchall()
    return [dict(r) for r in rows]
