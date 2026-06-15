"""XMP sidecar export.

Writes one ``<stem>.xmp`` per photo (Lightroom/Excire sidecar convention)
carrying:

- ``dc:subject``                 — manual tags + auto tags (score ≥ min_score)
- ``dc:description``             — the long-form caption (qwen import)
- ``xmp:Rating``                 — aesthetic 1..10 mapped to 0..5 stars
- ``Iptc4xmpExt:PersonInImage``  — named persons detected in the photo
- ``photoshop:DateCreated``      — captured_at when known

Safety posture (see threats T-02/T-03):
- Default target is an OUTPUT DIRECTORY mirroring the source tree — never
  next to the originals (the Apple-Photos source volume is a read-only-ish
  apfs-fuse mount). ``next_to_original=True`` must be requested explicitly.
- We refuse to overwrite any existing file whose suffix is not ``.xmp``.
- Sidecars are written 0o600 inside 0o700 directories.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from .db import connect

log = logging.getLogger(__name__)

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "Iptc4xmpExt": "http://iptc.org/std/Iptc4xmpExt/2008-02-29/",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

AUTO_TAG_MIN_SCORE = 0.5


def _q(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def build_xmp(*, tags: list[str], caption: str | None, rating: int | None,
              persons: list[str], date_created: str | None) -> str:
    """Return a serialized XMP packet for one photo."""
    xmpmeta = ET.Element(_q("x", "xmpmeta"))
    rdf = ET.SubElement(xmpmeta, _q("rdf", "RDF"))
    desc = ET.SubElement(rdf, _q("rdf", "Description"), {_q("rdf", "about"): ""})

    if tags:
        subject = ET.SubElement(desc, _q("dc", "subject"))
        bag = ET.SubElement(subject, _q("rdf", "Bag"))
        for t in tags:
            ET.SubElement(bag, _q("rdf", "li")).text = t

    if caption:
        d = ET.SubElement(desc, _q("dc", "description"))
        alt = ET.SubElement(d, _q("rdf", "Alt"))
        li = ET.SubElement(alt, _q("rdf", "li"))
        li.set("{http://www.w3.org/XML/1998/namespace}lang", "x-default")
        li.text = caption

    if rating is not None:
        ET.SubElement(desc, _q("xmp", "Rating")).text = str(rating)

    if persons:
        pii = ET.SubElement(desc, _q("Iptc4xmpExt", "PersonInImage"))
        bag = ET.SubElement(pii, _q("rdf", "Bag"))
        for p in persons:
            ET.SubElement(bag, _q("rdf", "li")).text = p

    if date_created:
        ET.SubElement(desc, _q("photoshop", "DateCreated")).text = date_created

    body = ET.tostring(xmpmeta, encoding="unicode")
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        + body
        + '\n<?xpacket end="w"?>\n'
    )


def aesthetic_to_rating(aesthetic: float | None) -> int | None:
    """Map the 1..10 aesthetic score to 0..5 XMP stars."""
    if aesthetic is None:
        return None
    return max(0, min(5, round(aesthetic / 2)))


def _photo_rows(conn, *, photo_ids=None, collection_id=None,
                path_prefix=None, limit=None) -> list[dict]:
    clauses, params = [], []
    join = ""
    if photo_ids is not None:
        if not photo_ids:
            return []  # an explicit empty selection means "export nothing"
        placeholders = ",".join("?" for _ in photo_ids)
        clauses.append(f"p.id IN ({placeholders})")
        params.extend(photo_ids)
    if collection_id is not None:
        join = "JOIN collection_photo cp ON cp.photo_id = p.id"
        clauses.append("cp.collection_id = ?")
        params.append(collection_id)
    if path_prefix:
        clauses.append("p.path LIKE ?")
        params.append(str(path_prefix).rstrip("/") + "/%")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    lim = f"LIMIT {int(limit)}" if limit else ""
    rows = conn.execute(
        f"""
        SELECT p.id, p.path, p.caption, p.aesthetic, p.captured_at
        FROM photo p {join} {where}
        ORDER BY p.id {lim}
        """,
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def _tags_for(conn, photo_id: int, min_score: float) -> list[str]:
    manual = [r["name"] for r in conn.execute(
        """
        SELECT t.name FROM photo_tag pt JOIN tag t ON t.id = pt.tag_id
        WHERE pt.photo_id = ? ORDER BY t.name
        """, (photo_id,))]
    auto = [r["label"] for r in conn.execute(
        """
        SELECT label FROM photo_tag_auto
        WHERE photo_id = ? AND score >= ? ORDER BY score DESC
        """, (photo_id, min_score))]
    seen, out = set(), []
    for t in manual + auto:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _persons_for(conn, photo_id: int) -> list[str]:
    return [r["name"] for r in conn.execute(
        """
        SELECT DISTINCT pe.name
        FROM face_detection fd JOIN person pe ON pe.id = fd.person_id
        WHERE fd.photo_id = ? ORDER BY pe.name
        """, (photo_id,))]


def _sidecar_target(photo_path: Path, out_dir: Path | None) -> Path:
    sidecar_name = photo_path.stem + ".xmp"
    if out_dir is None:
        return photo_path.parent / sidecar_name
    rel = photo_path.parent.relative_to(photo_path.anchor)
    return out_dir / rel / sidecar_name


def export_sidecars(*, out_dir: Path | None, photo_ids: list[int] | None = None,
                    collection_id: int | None = None, path_prefix: str | None = None,
                    min_score: float = AUTO_TAG_MIN_SCORE,
                    limit: int | None = None) -> dict:
    """Write XMP sidecars. ``out_dir=None`` means next-to-original (explicit opt-in
    at the API layer). Returns counts + the resolved output root."""
    conn = connect()
    photos = _photo_rows(conn, photo_ids=photo_ids, collection_id=collection_id,
                         path_prefix=path_prefix, limit=limit)

    written, skipped, failed = 0, 0, 0
    for ph in photos:
        src = Path(ph["path"])
        try:
            target = _sidecar_target(src, out_dir)
        except ValueError:
            log.warning("xmp: cannot map %s under out_dir", src)
            failed += 1
            continue
        if out_dir is not None:
            # Containment check: a hand-edited DB path with '..' components
            # must not let the sidecar escape the chosen output root.
            resolved = target.resolve()
            root = out_dir.resolve()
            if root != resolved and root not in resolved.parents:
                log.warning("xmp: refusing target outside out_dir: %s", target)
                failed += 1
                continue
            target = resolved
        if target.exists() and target.suffix.lower() != ".xmp":
            log.warning("xmp: refusing to overwrite non-xmp file %s", target)
            skipped += 1
            continue
        packet = build_xmp(
            tags=_tags_for(conn, ph["id"], min_score),
            caption=ph["caption"],
            rating=aesthetic_to_rating(ph["aesthetic"]),
            persons=_persons_for(conn, ph["id"]),
            date_created=ph["captured_at"],
        )
        try:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(packet)
            written += 1
        except OSError as exc:
            log.warning("xmp: write failed for %s: %s", target, exc)
            failed += 1

    return {"written": written, "skipped": skipped, "failed": failed,
            "total": len(photos),
            "out_dir": str(out_dir) if out_dir else None}
