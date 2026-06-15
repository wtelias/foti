"""Dupes: LSH banding equivalence vs brute force, scaling, ignore persistence."""

import random

from foti_backend.dupes import _band_values, cluster_key, find_duplicate_clusters

from tests.conftest import seed_photo


def _brute_force_pairs(hashes: dict[int, int], threshold: int) -> set[tuple[int, int]]:
    ids = list(hashes)
    pairs = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if (hashes[ids[i]] ^ hashes[ids[j]]).bit_count() <= threshold:
                pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))
    return pairs


def _components(pairs: set[tuple[int, int]], ids: list[int]) -> set[frozenset[int]]:
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(b)] = find(a)
    comps: dict[int, set[int]] = {}
    for i in ids:
        comps.setdefault(find(i), set()).add(i)
    return {frozenset(c) for c in comps.values() if len(c) > 1}


def test_band_values_cover_64_bits():
    h = (1 << 64) - 1
    for nbands in (1, 3, 6, 8):
        bands = _band_values(h, nbands)
        assert len(bands) == nbands
        total_bits = sum(v.bit_length() for _, v in bands)
        assert total_bits == 64


def test_banding_matches_brute_force(catalog):
    """The pigeonhole guarantee: banded clustering == brute-force clustering."""
    rng = random.Random(42)
    threshold = 5
    hashes: dict[int, int] = {}
    base_hashes = [rng.getrandbits(64) for _ in range(30)]
    for b in base_hashes:
        # the base photo + a few perturbations within/outside threshold
        for flips in (0, 2, 5, 9):
            h = b
            for bit in rng.sample(range(64), flips):
                h ^= 1 << bit
            pid = seed_photo(catalog, path=f"/d/{len(hashes)}.jpg", phash=f"{h:016x}")
            hashes[pid] = h

    clusters = find_duplicate_clusters(threshold=threshold, limit_clusters=10_000)
    got = {frozenset(p["id"] for p in c["photos"]) for c in clusters}

    expected = _components(_brute_force_pairs(hashes, threshold), list(hashes))
    assert got == expected


def test_banding_scales(catalog):
    """T-01: 5k random photos must cluster in interactive time (not O(N²) pure-Python)."""
    import time

    rng = random.Random(7)
    rows = [(f"/s/{i}.jpg", f"{rng.getrandbits(64):016x}") for i in range(5000)]
    catalog.executemany(
        "INSERT INTO photo (path, mtime, size_bytes, indexed_at, phash) "
        "VALUES (?, 0, 0, '2026-01-01T00:00:00Z', ?)", rows)
    catalog.commit()

    t0 = time.monotonic()
    find_duplicate_clusters(threshold=5, limit_clusters=200)
    elapsed = time.monotonic() - t0
    # Brute force would be 12.5M pair-XORs in Python (~10s+); banding stays snappy.
    assert elapsed < 5, f"clustering took {elapsed:.1f}s"


def test_identical_hashes_cluster(catalog):
    a = seed_photo(catalog, path="/i/a.jpg", phash="00000000000000aa")
    b = seed_photo(catalog, path="/i/b.jpg", phash="00000000000000aa")
    clusters = find_duplicate_clusters(threshold=0)
    assert len(clusters) == 1
    assert {p["id"] for p in clusters[0]["photos"]} == {a, b}


def test_ignore_persists(client, catalog):
    a = seed_photo(catalog, path="/g/a.jpg", phash="00000000000000ff")
    b = seed_photo(catalog, path="/g/b.jpg", phash="00000000000000ff")

    r = client.get("/duplicates")
    assert len(r.json()["clusters"]) == 1
    key = r.json()["clusters"][0]["cluster_key"]
    assert key == cluster_key([a, b])

    r = client.post("/duplicates/ignore", json={"photo_ids": [a, b]})
    assert r.status_code == 200 and r.json()["cluster_key"] == key

    assert client.get("/duplicates").json()["clusters"] == []
    shown = client.get("/duplicates?include_ignored=true").json()["clusters"]
    assert len(shown) == 1 and shown[0]["ignored"] is True

    r = client.delete(f"/duplicates/ignore/{key}")
    assert r.json()["removed"] is True
    assert len(client.get("/duplicates").json()["clusters"]) == 1


def test_ignore_requires_two_photos(client):
    assert client.post("/duplicates/ignore", json={"photo_ids": [1]}).status_code == 400
