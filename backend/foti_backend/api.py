"""HTTP API for the Foti backend.

Surfaces:
- ``GET  /health``                    — daemon liveness + version
- ``GET  /roots``                     — list scan roots
- ``POST /roots``                     — add a scan root
- ``DELETE /roots/{id}``              — remove a scan root
- ``POST /scan/{root_id}``            — kick off a scan (async)
- ``GET  /scan/{job_id}``             — poll scan progress
- ``GET  /photos``                    — list (paginated)
- ``GET  /photos/{id}``               — one photo with full metadata
- ``GET  /photos/{id}/thumb/{size}``  — thumbnail bytes
- ``GET  /search/text?q=...``         — text search
- ``GET  /search/similar/{photo_id}`` — image similarity
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import base64
import json

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__
from .config import get_settings
from .db import connect, transaction
from .aesthetic import score_catalog as aesthetic_score
from .collections import (create_collection, delete_collection,
                          list_collections, resolve as resolve_collection)
from .colors import backfill_colors, search_by_color
from .dupes import find_duplicate_clusters, ignore_cluster, unignore_cluster
from .backfill import backfill_embeddings
from .face_pipeline import index_faces, list_people, list_photos_in_cluster
from .importer import import_from_photos_db
from .persons import (list_persons, merge_clusters, name_cluster,
                      rename_person, unname_person)
from .scanner import scan_root
from .search import search_similar, search_text
from .tagging import photos_for_label, tag_catalog, top_labels
from .watcher import get_watcher
from .xmp import export_sidecars

_WEB_DIR = Path(__file__).parent / "web"

# Optional HTTP Basic auth: when FOTI_BASIC_USER + FOTI_BASIC_PASS are set
# in the environment, every request must carry matching credentials.
# Use this when exposing the daemon outside localhost / the tailnet.
_BASIC_USER = os.environ.get("FOTI_BASIC_USER")
_BASIC_PASS = os.environ.get("FOTI_BASIC_PASS")
_security = HTTPBasic(auto_error=False) if _BASIC_USER and _BASIC_PASS else None


def _require_auth(credentials: HTTPBasicCredentials = Depends(_security)) -> None:
    if _security is None:
        return
    if credentials is None:
        raise HTTPException(401, "auth required",
                            headers={"WWW-Authenticate": 'Basic realm="foti"'})
    user_ok = secrets.compare_digest(credentials.username, _BASIC_USER or "")
    pass_ok = secrets.compare_digest(credentials.password, _BASIC_PASS or "")
    if not (user_ok and pass_ok):
        raise HTTPException(401, "bad credentials",
                            headers={"WWW-Authenticate": 'Basic realm="foti"'})

log = logging.getLogger(__name__)

app = FastAPI(title="Foti", version=__version__)


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Gate every request (including StaticFiles) with HTTP Basic auth.

    FastAPI's ``dependencies=[]`` only applies to routes, not to mounted
    sub-apps like StaticFiles. A middleware runs in front of the whole
    ASGI tree, so it covers the static UI bundle too.
    """

    async def dispatch(self, request: Request, call_next):
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("basic "):
            return self._challenge()
        try:
            decoded = base64.b64decode(header[6:]).decode("utf-8")
            user, _, password = decoded.partition(":")
        except Exception:
            return self._challenge()
        if not (secrets.compare_digest(user, _BASIC_USER or "")
                and secrets.compare_digest(password, _BASIC_PASS or "")):
            return self._challenge()
        return await call_next(request)

    @staticmethod
    def _challenge() -> Response:
        return Response(
            status_code=401,
            content="auth required",
            headers={"WWW-Authenticate": 'Basic realm="foti"'},
        )


if _BASIC_USER and _BASIC_PASS:
    app.add_middleware(BasicAuthMiddleware)

# Tauri dev server runs on localhost:1420; in production the UI is served
# from the bundle and uses the IPC bridge, but during dev it hits HTTP.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "tauri://localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _start_watcher() -> None:
    get_watcher().start()


@app.on_event("shutdown")
async def _stop_watcher() -> None:
    get_watcher().stop()


class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str
    catalog: str


class ScanRootIn(BaseModel):
    path: str


class ScanRootOut(BaseModel):
    id: int
    path: str
    added_at: str
    last_scan: str | None
    enabled: bool


class ScanJobOut(BaseModel):
    id: int
    root_id: int | None
    state: str
    started_at: str | None
    finished_at: str | None
    total: int
    processed: int
    error: str | None


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    s = get_settings()
    return HealthOut(status="ok", version=__version__, catalog=str(s.catalog_path))


@app.get("/roots", response_model=list[ScanRootOut])
def list_roots() -> list[ScanRootOut]:
    conn = connect()
    rows = conn.execute(
        "SELECT id, path, added_at, last_scan, enabled FROM scan_root ORDER BY id"
    ).fetchall()
    return [
        ScanRootOut(id=r["id"], path=r["path"], added_at=r["added_at"],
                    last_scan=r["last_scan"], enabled=bool(r["enabled"]))
        for r in rows
    ]


@app.post("/roots", response_model=ScanRootOut)
def add_root(body: ScanRootIn) -> ScanRootOut:
    path = Path(body.path).expanduser().resolve()
    if not path.is_dir():
        raise HTTPException(400, f"not a directory: {path}")
    now = datetime.now(timezone.utc).isoformat()
    conn = connect()
    with transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO scan_root (path, added_at) VALUES (?, ?)",
            (str(path), now),
        )
        row = conn.execute(
            "SELECT id, path, added_at, last_scan, enabled FROM scan_root WHERE path = ?",
            (str(path),),
        ).fetchone()
    # Pick up the new root in the watch loop.
    get_watcher()._refresh_watches()
    return ScanRootOut(id=row["id"], path=row["path"], added_at=row["added_at"],
                       last_scan=row["last_scan"], enabled=bool(row["enabled"]))


@app.delete("/roots/{root_id}")
def remove_root(root_id: int) -> dict:
    conn = connect()
    with transaction(conn):
        cur = conn.execute("DELETE FROM scan_root WHERE id = ?", (root_id,))
    return {"deleted": cur.rowcount}


def _run_scan_job(job_id: int, root_id: int, root_path: Path) -> None:
    conn = connect()
    now = datetime.now(timezone.utc).isoformat()
    with transaction(conn):
        conn.execute(
            "UPDATE scan_job SET state='running', started_at=? WHERE id=?",
            (now, job_id),
        )

    def progress(done: int, total: int) -> None:
        c = connect()
        with transaction(c):
            c.execute(
                "UPDATE scan_job SET processed=?, total=? WHERE id=?",
                (done, total, job_id),
            )

    try:
        summary = scan_root(root_path, progress=progress)
        finished = datetime.now(timezone.utc).isoformat()
        with transaction(conn):
            conn.execute(
                "UPDATE scan_job SET state='done', finished_at=?, processed=?, total=? WHERE id=?",
                (finished, summary["indexed"], summary["candidates"], job_id),
            )
            conn.execute(
                "UPDATE scan_root SET last_scan=? WHERE id=?",
                (finished, root_id),
            )
        # Best-effort face + aesthetic passes for the newly indexed photos.
        # Both run after the job is marked done so the UI returns quickly.
        try:
            index_faces(batch_size=max(summary["indexed"] * 5, 100))
        except Exception:
            log.exception("post-scan face indexing failed for job %d", job_id)
        try:
            aesthetic_score(batch_size=max(summary["indexed"] * 5, 200))
        except Exception:
            log.exception("post-scan aesthetic scoring failed for job %d", job_id)
        try:
            tag_catalog(batch_size=max(summary["indexed"] * 5, 500))
        except Exception:
            log.exception("post-scan auto-tagging failed for job %d", job_id)
    except Exception as exc:
        log.exception("scan job %d failed", job_id)
        with transaction(conn):
            conn.execute(
                "UPDATE scan_job SET state='failed', finished_at=?, error=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), str(exc), job_id),
            )


@app.post("/scan/{root_id}", response_model=ScanJobOut)
def start_scan(root_id: int) -> ScanJobOut:
    conn = connect()
    row = conn.execute(
        "SELECT id, path FROM scan_root WHERE id = ?", (root_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(404, "unknown scan root")

    with transaction(conn):
        cur = conn.execute(
            "INSERT INTO scan_job (root_id, state) VALUES (?, 'queued')",
            (root_id,),
        )
        job_id = cur.lastrowid

    threading.Thread(
        target=_run_scan_job,
        args=(job_id, root_id, Path(row["path"])),
        name=f"scan-job-{job_id}",
        daemon=True,
    ).start()

    return _job_record(job_id)


@app.get("/scan/{job_id}", response_model=ScanJobOut)
def get_scan(job_id: int) -> ScanJobOut:
    return _job_record(job_id)


def _job_record(job_id: int) -> ScanJobOut:
    conn = connect()
    row = conn.execute(
        "SELECT id, root_id, state, started_at, finished_at, total, processed, error "
        "FROM scan_job WHERE id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "unknown job")
    return ScanJobOut(**dict(row))


_SORT_CLAUSES = {
    "newest": "COALESCE(captured_at, indexed_at) DESC",
    "oldest": "COALESCE(captured_at, indexed_at) ASC",
    "best":   "aesthetic DESC NULLS LAST, COALESCE(captured_at, indexed_at) DESC",
    "worst":  "aesthetic ASC NULLS LAST, COALESCE(captured_at, indexed_at) DESC",
}


@app.get("/photos")
def list_photos(limit: int = Query(100, le=500), offset: int = 0,
                sort: Literal["newest", "oldest", "best", "worst"] = "newest",
                year: int | None = None,
                camera: str | None = None,
                date_from: str | None = None,
                date_to: str | None = None) -> dict:
    order_by = _SORT_CLAUSES[sort]
    where = ["1 = 1"]
    params: list = []
    if year is not None:
        where.append("substr(COALESCE(captured_at, indexed_at), 1, 4) = ?")
        params.append(str(year))
    if camera:
        where.append("exif_json LIKE ?")
        params.append(f'%"Image Model": "{camera}"%')
    if date_from:
        where.append("COALESCE(captured_at, indexed_at) >= ?")
        params.append(date_from)
    if date_to:
        where.append("COALESCE(captured_at, indexed_at) <= ?")
        params.append(date_to)
    where_sql = " AND ".join(where)

    conn = connect()
    rows = conn.execute(
        f"""
        SELECT id, path, captured_at, width, height, thumb_small, thumb_large, aesthetic
        FROM photo
        WHERE {where_sql}
        ORDER BY {order_by}
        LIMIT ? OFFSET ?
        """,
        (*params, limit, offset),
    ).fetchall()
    total = conn.execute(
        f"SELECT COUNT(*) FROM photo WHERE {where_sql}", tuple(params)
    ).fetchone()[0]
    return {"total": total, "limit": limit, "offset": offset, "sort": sort,
            "items": [dict(r) for r in rows]}


@app.post("/tags/index")
def tags_index_endpoint(batch_size: int = Query(1000, ge=1, le=10000)) -> dict:
    """Auto-tag photos via CLIP zero-shot over the built-in vocabulary."""
    return tag_catalog(batch_size=batch_size)


@app.get("/tags")
def tags_top_endpoint(limit: int = Query(30, le=200)) -> dict:
    return {"labels": top_labels(limit=limit)}


@app.get("/tags/{label}")
def tags_label_endpoint(label: str, limit: int = Query(200, le=500)) -> dict:
    return {"label": label, "results": photos_for_label(label, limit=limit)}


@app.get("/facets")
def facets_endpoint() -> dict:
    """Return histogram buckets the UI can use to build filter chips."""
    conn = connect()
    years = conn.execute(
        """
        SELECT substr(COALESCE(captured_at, indexed_at), 1, 4) AS year, COUNT(*) AS n
        FROM photo
        GROUP BY year
        ORDER BY year DESC
        """
    ).fetchall()
    cameras = conn.execute(
        """
        SELECT json_extract(exif_json, '$."Image Model"') AS camera, COUNT(*) AS n
        FROM photo
        WHERE exif_json IS NOT NULL
          AND json_extract(exif_json, '$."Image Model"') IS NOT NULL
        GROUP BY camera
        ORDER BY n DESC
        LIMIT 30
        """
    ).fetchall()
    return {
        "years": [{"year": r["year"], "count": r["n"]} for r in years if r["year"]],
        "cameras": [{"camera": r["camera"], "count": r["n"]} for r in cameras],
    }


@app.get("/photos/{photo_id}")
def get_photo(photo_id: int) -> dict:
    conn = connect()
    row = conn.execute("SELECT * FROM photo WHERE id = ?", (photo_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown photo")
    return dict(row)


@app.get("/photos/{photo_id}/thumb/{size}")
def get_thumb(photo_id: int, size: Literal["small", "large"]) -> FileResponse:
    conn = connect()
    row = conn.execute(
        "SELECT path, thumb_small, thumb_large FROM photo WHERE id = ?", (photo_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(404, "unknown photo")
    p = row["thumb_small"] if size == "small" else row["thumb_large"]
    if p and Path(p).is_file():
        return FileResponse(p, media_type="image/webp")
    # Lazy render: imported photos arrive without thumbs; build them on first view.
    return _render_thumb_on_miss(photo_id, row["path"], size)


def _render_thumb_on_miss(photo_id: int, src_path: str,
                          size: Literal["small", "large"]) -> FileResponse:
    from PIL import ImageOps
    from .imageio import open_image
    settings = get_settings()
    src = Path(src_path)
    if not src.is_file():
        raise HTTPException(410, f"file missing on disk: {src}")
    out_dir = settings.thumbs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{photo_id}_{'s' if size == 'small' else 'l'}.webp"
    max_dim = settings.thumbnail_size_small if size == "small" else settings.thumbnail_size_large
    try:
        with open_image(src) as raw:
            img = ImageOps.exif_transpose(raw).copy()
        img.thumbnail((max_dim, max_dim))
        img.save(out, format="WEBP", quality=80 if size == "small" else 85, method=4)
    except Exception as exc:
        log.warning("on-miss thumb render failed for photo %d: %s", photo_id, exc)
        raise HTTPException(500, f"thumb render failed: {exc}")
    conn = connect()
    col = "thumb_small" if size == "small" else "thumb_large"
    with transaction(conn):
        conn.execute(f"UPDATE photo SET {col} = ? WHERE id = ?", (str(out), photo_id))
    return FileResponse(out, media_type="image/webp")


@app.get("/photos/{photo_id}/full")
def get_full(photo_id: int) -> FileResponse:
    """Stream the original file. Useful for the lightbox and for downloads."""
    conn = connect()
    row = conn.execute("SELECT path FROM photo WHERE id = ?", (photo_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown photo")
    src = Path(row["path"])
    if not src.is_file():
        raise HTTPException(410, f"file missing on disk: {src}")
    # Let the browser sniff the type; for RAW we have no good mimetype anyway.
    return FileResponse(src)


@app.get("/search/text")
def search_text_endpoint(q: str, limit: int = Query(50, le=200)) -> dict:
    return {"query": q, "results": search_text(q, limit=limit)}


@app.get("/search/similar/{photo_id}")
def search_similar_endpoint(photo_id: int, limit: int = Query(50, le=200)) -> dict:
    return {"photo_id": photo_id, "results": search_similar(photo_id, limit=limit)}


@app.get("/search/color")
def search_color_endpoint(hex: str = Query(..., pattern=r"^#?[0-9a-fA-F]{6}$"),
                          tolerance: int = Query(60, ge=0, le=255),
                          limit: int = Query(100, le=500)) -> dict:
    """Find photos whose dominant palette contains a color near the query."""
    h = hex.lstrip("#")
    rgb = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    results = search_by_color(rgb, tolerance=tolerance, limit=limit)
    return {"hex": f"#{h}", "rgb": rgb, "tolerance": tolerance, "results": results}


@app.get("/duplicates")
def duplicates_endpoint(threshold: int = Query(5, ge=0, le=20),
                       limit: int = Query(50, le=200),
                       include_ignored: bool = Query(False)) -> dict:
    clusters = find_duplicate_clusters(threshold=threshold, limit_clusters=limit,
                                       include_ignored=include_ignored)
    return {"threshold": threshold, "clusters": clusters}


class DupeIgnoreIn(BaseModel):
    photo_ids: list[int]


@app.post("/duplicates/ignore")
def duplicates_ignore_endpoint(body: DupeIgnoreIn) -> dict:
    if len(body.photo_ids) < 2:
        raise HTTPException(400, "a duplicate cluster has at least 2 photos")
    return {"cluster_key": ignore_cluster(body.photo_ids)}


@app.delete("/duplicates/ignore/{cluster_key}")
def duplicates_unignore_endpoint(cluster_key: str) -> dict:
    return {"removed": unignore_cluster(cluster_key)}


@app.post("/aesthetic/score")
def aesthetic_score_endpoint(batch_size: int = Query(500, ge=1, le=5000)) -> dict:
    """Compute aesthetic scores for photos that don't have one yet."""
    return aesthetic_score(batch_size=batch_size)


@app.post("/colors/backfill")
def colors_backfill_endpoint(batch_size: int = Query(500, ge=1, le=5000)) -> dict:
    """Extract dominant colors for photos missing them. Run repeatedly."""
    return backfill_colors(batch_size=batch_size)


@app.get("/best-photos")
def photos_best_endpoint(limit: int = Query(50, le=200)) -> dict:
    """Top photos by aesthetic score (descending)."""
    conn = connect()
    rows = conn.execute(
        """
        SELECT id, path, captured_at, width, height,
               thumb_small, thumb_large, aesthetic
        FROM photo
        WHERE aesthetic IS NOT NULL
        ORDER BY aesthetic DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return {"limit": limit, "items": [dict(r) for r in rows]}


@app.post("/faces/index")
def faces_index_endpoint(batch_size: int = Query(200, ge=1, le=2000)) -> dict:
    """Process a batch of un-faced photos. Run repeatedly to cover the catalog."""
    summary = index_faces(batch_size=batch_size)
    return summary


@app.get("/faces/people")
def faces_people_endpoint(min_size: int = Query(2, ge=1),
                         limit: int = Query(100, le=500)) -> dict:
    return {"people": list_people(min_size=min_size, limit=limit)}


@app.get("/faces/cluster/{cluster_id}")
def faces_cluster_endpoint(cluster_id: int, limit: int = Query(200, le=500)) -> dict:
    return {"cluster_id": cluster_id, "photos": list_photos_in_cluster(cluster_id, limit=limit)}


class XmpExportIn(BaseModel):
    out_dir: str | None = None
    next_to_original: bool = False
    photo_ids: list[int] | None = Field(None, max_length=5000)
    collection_id: int | None = None
    path_prefix: str | None = None
    all: bool = False
    min_score: float = 0.5
    limit: int | None = None


@app.post("/export/xmp")
def export_xmp_endpoint(body: XmpExportIn) -> dict:
    has_selector = bool(body.photo_ids) or body.collection_id is not None \
        or bool(body.path_prefix)
    if not has_selector and not body.all:
        raise HTTPException(400, "no selector given — pass photo_ids, collection_id, "
                                 "path_prefix, or all=true for the whole catalog")
    if body.next_to_original:
        out_dir = None
    else:
        if not body.out_dir:
            raise HTTPException(400, "out_dir is required (or set next_to_original=true "
                                     "to write sidecars beside the photos)")
        out_dir = Path(body.out_dir)
        if not out_dir.is_absolute():
            raise HTTPException(400, "out_dir must be an absolute path")
    return export_sidecars(out_dir=out_dir, photo_ids=body.photo_ids,
                           collection_id=body.collection_id,
                           path_prefix=body.path_prefix,
                           min_score=body.min_score, limit=body.limit)


@app.get("/collections")
def collections_list_endpoint() -> dict:
    return {"items": list_collections()}


class CollectionIn(BaseModel):
    name: str
    query: dict


@app.post("/collections")
def collections_create_endpoint(body: CollectionIn) -> dict:
    return create_collection(body.name, body.query)


@app.delete("/collections/{cid}")
def collections_delete_endpoint(cid: int) -> dict:
    return {"deleted": delete_collection(cid)}


@app.get("/collections/{cid}/photos")
def collections_resolve_endpoint(cid: int, limit: int = Query(200, le=500)) -> dict:
    return resolve_collection(cid, limit=limit)


class ClusterNameIn(BaseModel):
    name: str


@app.post("/faces/cluster/{cluster_id}/name")
def cluster_name_endpoint(cluster_id: int, body: ClusterNameIn) -> dict:
    try:
        return name_cluster(cluster_id, body.name)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


class ClusterMergeIn(BaseModel):
    target_cluster_id: int


@app.post("/faces/cluster/{cluster_id}/merge")
def cluster_merge_endpoint(cluster_id: int, body: ClusterMergeIn) -> dict:
    try:
        return merge_clusters(cluster_id, body.target_cluster_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/persons")
def persons_list_endpoint() -> dict:
    return {"persons": list_persons()}


class PersonRenameIn(BaseModel):
    name: str


@app.post("/persons/{person_id}/rename")
def person_rename_endpoint(person_id: int, body: PersonRenameIn) -> dict:
    try:
        return rename_person(person_id, body.name)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.delete("/persons/{person_id}")
def person_unname_endpoint(person_id: int) -> dict:
    """Remove a person's name — detaches it from its faces; cluster stays."""
    try:
        return unname_person(person_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/faces/{face_id}/crop")
def face_crop_endpoint(face_id: int) -> Response:
    """Return a 256-px cropped + centered face thumbnail."""
    from PIL import Image as PILImage  # local — keeps cold-start light
    import io as _io

    conn = connect()
    row = conn.execute(
        """
        SELECT fd.bbox_json, p.thumb_large, p.path
        FROM face_detection fd
        JOIN photo p ON p.id = fd.photo_id
        WHERE fd.id = ?
        """,
        (face_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "unknown face")
    bbox = json.loads(row["bbox_json"])  # [x, y, w, h] in pixels of original
    x, y, w, h = bbox

    # Crop from the large thumbnail to keep this cheap. Scale bbox to thumb
    # dimensions using the original photo width/height stored in `photo`.
    src_path = row["thumb_large"] or row["path"]
    img = PILImage.open(src_path)
    # Compute scale: bbox is in original-image coords, thumb_large is bounded.
    orig_w_row = conn.execute(
        "SELECT width, height FROM photo WHERE id = (SELECT photo_id FROM face_detection WHERE id = ?)",
        (face_id,),
    ).fetchone()
    if orig_w_row and orig_w_row["width"] and orig_w_row["height"]:
        scale_x = img.width / orig_w_row["width"]
        scale_y = img.height / orig_w_row["height"]
    else:
        scale_x = scale_y = 1.0
    pad = 0.25  # expand bbox by 25 % on each side for some chin/hair context
    cx = (x + w / 2) * scale_x
    cy = (y + h / 2) * scale_y
    side = max(w * scale_x, h * scale_y) * (1 + pad * 2)
    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side / 2))
    x2 = min(img.width, int(cx + side / 2))
    y2 = min(img.height, int(cy + side / 2))

    crop = img.crop((x1, y1, x2, y2))
    crop.thumbnail((256, 256))
    buf = _io.BytesIO()
    crop.convert("RGB").save(buf, format="WEBP", quality=85)
    return Response(content=buf.getvalue(), media_type="image/webp")


class ImportIn(BaseModel):
    path: str
    limit: int | None = None
    check_files: bool = False


def _run_import_job(job_id: int, src_path: str, limit: int | None,
                    check_files: bool) -> None:
    conn = connect()
    now = datetime.now(timezone.utc).isoformat()
    with transaction(conn):
        conn.execute(
            "UPDATE scan_job SET state='running', started_at=? WHERE id=?",
            (now, job_id),
        )

    def progress(done: int, total: int) -> None:
        c = connect()
        with transaction(c):
            c.execute(
                "UPDATE scan_job SET processed=?, total=? WHERE id=?",
                (done, total, job_id),
            )

    try:
        summary = import_from_photos_db(
            src_path, limit=limit, check_files_exist=check_files,
            progress=progress,
        )
        finished = datetime.now(timezone.utc).isoformat()
        with transaction(conn):
            conn.execute(
                "UPDATE scan_job SET state='done', finished_at=?, processed=?, total=? WHERE id=?",
                (finished, summary["inserted"], summary["tagged_total"], job_id),
            )
    except Exception as exc:
        log.exception("import job %d failed", job_id)
        with transaction(conn):
            conn.execute(
                "UPDATE scan_job SET state='failed', finished_at=?, error=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), str(exc), job_id),
            )


class BackfillIn(BaseModel):
    limit: int | None = None
    batch_size: int | None = None
    with_thumbs: bool = True
    with_colors: bool = True


def _run_backfill_job(job_id: int, limit: int | None, batch_size: int | None,
                     with_thumbs: bool, with_colors: bool) -> None:
    conn = connect()
    with transaction(conn):
        conn.execute(
            "UPDATE scan_job SET state='running', started_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), job_id),
        )

    def progress(done: int, total: int) -> None:
        c = connect()
        with transaction(c):
            c.execute(
                "UPDATE scan_job SET processed=?, total=? WHERE id=?",
                (done, total, job_id),
            )

    try:
        summary = backfill_embeddings(
            limit=limit, batch_size=batch_size,
            with_thumbs=with_thumbs, with_colors=with_colors,
            progress=progress,
        )
        with transaction(conn):
            conn.execute(
                "UPDATE scan_job SET state='done', finished_at=?, processed=?, total=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(),
                 summary["encoded"], summary["candidates_initial"], job_id),
            )
        # Post-backfill: score aesthetics + auto-tag against the new embeddings.
        # Both reuse the embedding cache and don't re-touch the photo files.
        try:
            aesthetic_score(batch_size=max(summary["encoded"], 1000))
        except Exception:
            log.exception("post-backfill aesthetic scoring failed (job %d)", job_id)
        try:
            tag_catalog(batch_size=max(summary["encoded"], 1000))
        except Exception:
            log.exception("post-backfill auto-tagging failed (job %d)", job_id)
    except Exception as exc:
        log.exception("backfill job %d failed", job_id)
        with transaction(conn):
            conn.execute(
                "UPDATE scan_job SET state='failed', finished_at=?, error=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), str(exc), job_id),
            )


@app.post("/backfill/embeddings", response_model=ScanJobOut)
def backfill_embeddings_endpoint(body: BackfillIn) -> ScanJobOut:
    """Kick off a CLIP-embedding backfill in the background."""
    conn = connect()
    with transaction(conn):
        cur = conn.execute(
            "INSERT INTO scan_job (root_id, state) VALUES (NULL, 'queued')"
        )
        job_id = cur.lastrowid
    threading.Thread(
        target=_run_backfill_job,
        args=(job_id, body.limit, body.batch_size, body.with_thumbs, body.with_colors),
        name=f"backfill-job-{job_id}",
        daemon=True,
    ).start()
    return _job_record(job_id)


@app.get("/backfill/status")
def backfill_status_endpoint() -> dict:
    """Quick read of how much backfill remains to do."""
    conn = connect()
    total = conn.execute("SELECT COUNT(*) FROM photo").fetchone()[0]
    embedded = conn.execute("SELECT COUNT(*) FROM photo_embedding").fetchone()[0]
    return {
        "total_photos": total,
        "with_embedding": embedded,
        "missing": total - embedded,
        "percent_done": round(100.0 * embedded / max(total, 1), 1),
    }


@app.post("/import/photos-db", response_model=ScanJobOut)
def import_photos_db_endpoint(body: ImportIn) -> ScanJobOut:
    """Kick off a background import of a sibling photos.db catalog."""
    src = Path(body.path).expanduser().resolve()
    if not src.is_file():
        raise HTTPException(400, f"not a file: {src}")
    conn = connect()
    with transaction(conn):
        cur = conn.execute(
            "INSERT INTO scan_job (root_id, state) VALUES (NULL, 'queued')"
        )
        job_id = cur.lastrowid
    threading.Thread(
        target=_run_import_job,
        args=(job_id, str(src), body.limit, body.check_files),
        name=f"import-job-{job_id}",
        daemon=True,
    ).start()
    return _job_record(job_id)


# Bundled static web preview. Mount last so JSON routes win when paths overlap.
if _WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
