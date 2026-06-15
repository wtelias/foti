"""Duplicate detection via perceptual hashing.

We store a 64-bit pHash per photo as a hex string. Duplicate clusters are
formed by grouping photos with Hamming distance ≤ ``threshold`` (default 5).

Scaling: instead of the naive O(N²) scan, we use multi-index LSH banding —
the 64-bit hash is split into ``threshold + 1`` disjoint bands. By pigeonhole,
two hashes within ``threshold`` bits of each other must agree EXACTLY on at
least one band, so only photos sharing a band bucket are candidate pairs.
Candidates are verified with the true Hamming distance, then merged via
union-find. This keeps the live 97k-photo catalog interactive.

A pathological bucket (e.g. thousands of near-black photos hashing alike)
is capped: identical full hashes are unioned in O(n), and distinct hashes
beyond ``BUCKET_PAIR_CAP`` are skipped with a log line rather than letting
one bucket reintroduce the quadratic blowup.

Clusters the user dismissed are persisted in ``dupe_ignore`` keyed by a
sha256 over the sorted member ids, so the same grouping stays hidden.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from datetime import datetime, timezone

from .db import connect, transaction

log = logging.getLogger(__name__)

BUCKET_PAIR_CAP = 2000  # max distinct hashes in one band bucket we'll pair-verify


def _hamming_hex(a: str, b: str) -> int:
    """Hamming distance between two equal-length hex strings."""
    return (int(a, 16) ^ int(b, 16)).bit_count()


def cluster_key(photo_ids: list[int]) -> str:
    """Stable identity for a cluster grouping: sha256 over sorted member ids."""
    canon = ",".join(str(i) for i in sorted(photo_ids))
    return hashlib.sha256(canon.encode()).hexdigest()


def _band_values(h: int, nbands: int) -> list[tuple[int, int]]:
    """Split a 64-bit int into ``nbands`` disjoint (band_index, value) chunks."""
    base = 64 // nbands
    extra = 64 % nbands
    out = []
    shift = 0
    for i in range(nbands):
        width = base + (1 if i < extra else 0)
        mask = (1 << width) - 1
        out.append((i, (h >> shift) & mask))
        shift += width
    return out


def ignore_cluster(photo_ids: list[int]) -> str:
    """Persist a dismissed cluster. Returns its key."""
    key = cluster_key(photo_ids)
    conn = connect()
    with transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO dupe_ignore (cluster_key, created_at) VALUES (?, ?)",
            (key, datetime.now(timezone.utc).isoformat()),
        )
    return key


def unignore_cluster(key: str) -> bool:
    conn = connect()
    with transaction(conn):
        cur = conn.execute("DELETE FROM dupe_ignore WHERE cluster_key = ?", (key,))
    return cur.rowcount > 0


# Clustering the live ~100k catalog takes ~10s; cache per parameter set and
# invalidate when the phash population changes. The lock prevents two
# concurrent cache-misses from both running the expensive compute.
_cache: dict = {}
_cache_lock = threading.Lock()


def find_duplicate_clusters(threshold: int = 5, limit_clusters: int = 200,
                            include_ignored: bool = False) -> list[dict]:
    """Group photos whose pHash differs by at most ``threshold`` bits."""
    threshold = max(0, min(threshold, 20))
    conn = connect()

    # The clustering itself is the ~10s part on a ~100k catalog; cache it
    # keyed on the phash population only. Ignore-filtering happens per call
    # below so dismissing a cluster never triggers a recompute.
    state = conn.execute(
        """
        SELECT (SELECT COUNT(*) FROM photo WHERE phash IS NOT NULL) AS n,
               (SELECT COALESCE(MAX(id), 0) FROM photo) AS max_id
        """
    ).fetchone()
    from .config import get_settings

    cache_key = (str(get_settings().catalog_path), threshold,
                 state["n"], state["max_id"])
    with _cache_lock:
        if cache_key not in _cache:
            _cache.clear()  # keep at most one parameter set resident
            _cache[cache_key] = _compute_clusters(conn, threshold)
        all_clusters = _cache[cache_key]
    return _apply_ignore_filter(conn, all_clusters, include_ignored, limit_clusters)


def _compute_clusters(conn, threshold: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, path, phash, captured_at, width, height, size_bytes,
               thumb_small, thumb_large
        FROM photo
        WHERE phash IS NOT NULL
        ORDER BY id
        """
    ).fetchall()
    if not rows:
        return []

    parent: dict[int, int] = {r["id"]: r["id"] for r in rows}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ph_int: dict[int, int] = {}
    for r in rows:
        try:
            ph_int[r["id"]] = int(r["phash"], 16)
        except (TypeError, ValueError):
            continue

    # 1) Identical hashes union trivially (covers exact re-uploads cheaply).
    by_hash: dict[int, list[int]] = {}
    for pid, h in ph_int.items():
        by_hash.setdefault(h, []).append(pid)
    for ids in by_hash.values():
        for other in ids[1:]:
            union(ids[0], other)

    # 2) LSH banding over DISTINCT hashes only.
    if threshold > 0:
        nbands = threshold + 1
        buckets: dict[tuple[int, int], list[int]] = {}
        distinct = list(by_hash.keys())
        for h in distinct:
            for band in _band_values(h, nbands):
                buckets.setdefault(band, []).append(h)

        # No cross-band pair-dedup set: at ~100k photos it would transiently
        # hold millions of tuples (~1 GB). union() is idempotent, so a pair
        # that shares several bands just gets Hamming-checked a few times —
        # far cheaper than the allocation.
        for band, hashes in buckets.items():
            if len(hashes) < 2:
                continue
            if len(hashes) > BUCKET_PAIR_CAP:
                log.warning("dupes: band bucket %s has %d distinct hashes — capped",
                            band, len(hashes))
                hashes = hashes[:BUCKET_PAIR_CAP]
            for i in range(len(hashes)):
                hi = hashes[i]
                for j in range(i + 1, len(hashes)):
                    hj = hashes[j]
                    if (hi ^ hj).bit_count() <= threshold:
                        union(by_hash[hi][0], by_hash[hj][0])

    # Bucket photos by cluster root.
    clusters: dict[int, list[int]] = {}
    for pid in ph_int:
        clusters.setdefault(find(pid), []).append(pid)

    photo_by_id = {r["id"]: dict(r) for r in rows}
    out = []
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        out.append({
            "cluster_id": root,
            "cluster_key": cluster_key(members),
            "size": len(members),
            "photos": [photo_by_id[pid] for pid in members],
        })
    out.sort(key=lambda c: -c["size"])
    return out


def _apply_ignore_filter(conn, all_clusters: list[dict], include_ignored: bool,
                         limit_clusters: int) -> list[dict]:
    ignored_keys = {r["cluster_key"] for r in conn.execute(
        "SELECT cluster_key FROM dupe_ignore")}
    out = []
    for c in all_clusters:
        ignored = c["cluster_key"] in ignored_keys
        if not include_ignored and ignored:
            continue
        out.append({**c, "ignored": ignored})
        if len(out) >= limit_clusters:
            break
    return out
