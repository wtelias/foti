"""Tests added from the Phase-3 test-gap review (8 ranked gaps)."""

import random

import pytest

from foti_backend.dupes import find_duplicate_clusters

from tests.conftest import make_image, seed_face, seed_photo

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"


def test_banding_matches_brute_force_threshold_1(catalog):
    """Gap 1: pigeonhole guarantee at threshold=1 incl. band-boundary bit pairs."""
    rng = random.Random(99)
    base = rng.getrandbits(64)
    # Pairs straddling the 2-band boundary (bits 31/32) and within-band.
    variants = [base, base ^ (1 << 31), base ^ (1 << 32), base ^ (1 << 0),
                base ^ (1 << 31) ^ (1 << 32)]  # distance 2 from base — excluded
    ids = {}
    for i, h in enumerate(variants):
        ids[seed_photo(catalog, path=f"/t1/{i}.jpg", phash=f"{h:016x}")] = h

    clusters = find_duplicate_clusters(threshold=1, limit_clusters=100)
    got = {frozenset(p["id"] for p in c["photos"]) for c in clusters}

    # brute force at threshold 1
    id_list = list(ids)
    parent = {i: i for i in id_list}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for i in range(len(id_list)):
        for j in range(i + 1, len(id_list)):
            a, b = id_list[i], id_list[j]
            if (ids[a] ^ ids[b]).bit_count() <= 1:
                parent[find(b)] = find(a)
    comps = {}
    for i in id_list:
        comps.setdefault(find(i), set()).add(i)
    expected = {frozenset(c) for c in comps.values() if len(c) > 1}
    assert got == expected


def test_name_cluster_whitespace_rejected(catalog, client):
    from foti_backend.persons import name_cluster

    pid = seed_photo(catalog, path="/w/a.jpg")
    seed_face(catalog, pid, cluster_id=10)

    with pytest.raises(ValueError):
        name_cluster(10, "   ")
    assert client.post("/faces/cluster/10/name", json={"name": "  "}).status_code == 400


def test_rename_whitespace_rejected_via_api(catalog, client):
    pid = seed_photo(catalog, path="/w/b.jpg")
    seed_face(catalog, pid, cluster_id=10)
    person_id = client.post("/faces/cluster/10/name",
                            json={"name": "Valid"}).json()["person_id"]
    assert client.post(f"/persons/{person_id}/rename",
                       json={"name": "  "}).status_code == 400


def test_manual_tag_beats_auto_tag_case(catalog, tmp_path):
    """Gap 4: manual 'Dog' wins over auto 'dog' — exactly one li, manual casing."""
    from foti_backend.xmp import export_sidecars
    import xml.etree.ElementTree as ET

    pid = seed_photo(catalog, path="/t4/a.jpg")
    catalog.execute("INSERT INTO tag (name) VALUES ('Dog')")
    tag_id = catalog.execute("SELECT id FROM tag WHERE name='Dog'").fetchone()["id"]
    catalog.execute("INSERT INTO photo_tag (photo_id, tag_id) VALUES (?, ?)", (pid, tag_id))
    catalog.execute(
        "INSERT INTO photo_tag_auto (photo_id, label, score) VALUES (?, 'dog', 0.95)", (pid,))
    catalog.commit()

    export_sidecars(out_dir=tmp_path / "x", photo_ids=[pid])
    text = (tmp_path / "x/t4/a.xmp").read_text()
    start = text.index("<x:xmpmeta")
    end = text.index("<?xpacket end")
    root = ET.fromstring(text[start:end])
    subjects = [li.text for li in root.iter(f"{RDF}li")]
    assert subjects.count("Dog") == 1
    assert "dog" not in subjects


def test_null_phash_rows_excluded(catalog):
    a = seed_photo(catalog, path="/t5/a.jpg", phash="00000000000000aa")
    b = seed_photo(catalog, path="/t5/b.jpg", phash="00000000000000aa")
    seed_photo(catalog, path="/t5/c.jpg")  # phash NULL

    clusters = find_duplicate_clusters(threshold=5)
    assert len(clusters) == 1
    assert {p["id"] for p in clusters[0]["photos"]} == {a, b}


def test_export_all_empty_catalog(client, tmp_path):
    r = client.post("/export/xmp", json={"out_dir": str(tmp_path / "o"), "all": True})
    assert r.status_code == 200
    body = r.json()
    assert body["written"] == 0 and body["total"] == 0


def test_unignore_unknown_key_idempotent(client):
    r = client.delete("/duplicates/ignore/nonexistentkey")
    assert r.status_code == 200
    assert r.json()["removed"] is False


def test_xmp_dotdot_path_cannot_escape_out_dir(catalog, tmp_path):
    """Security: a hand-edited '..' DB path must not write outside out_dir."""
    from foti_backend.xmp import export_sidecars

    pid = seed_photo(catalog, path="/safe/../../evil/escape.jpg")
    out = tmp_path / "boxed"
    sentinel = tmp_path / "evil"

    summary = export_sidecars(out_dir=out, photo_ids=[pid])
    assert summary["written"] == 0
    assert summary["failed"] == 1
    assert not sentinel.exists()
    # nothing escaped: anything written (there should be none) sits under out
    outside = [p for p in tmp_path.rglob("*.xmp") if out not in p.parents]
    assert outside == []


def test_xmp_empty_photo_ids_exports_nothing(catalog, tmp_path):
    """Correctness: photo_ids=[] means 'nothing', not 'everything'."""
    from foti_backend.xmp import export_sidecars

    seed_photo(catalog, path="/e/a.jpg")
    summary = export_sidecars(out_dir=tmp_path / "o", photo_ids=[])
    assert summary["total"] == 0 and summary["written"] == 0


def test_backfill_colors_corrupt_file_marked(catalog, tmp_path):
    """Gap 8: a corrupt image marks '[]' (no infinite retry loop)."""
    corrupt = tmp_path / "bad.jpg"
    corrupt.write_bytes(b"NOT_AN_IMAGE")
    pid = seed_photo(catalog, path=str(corrupt))

    from foti_backend.colors import backfill_colors

    s1 = backfill_colors(batch_size=10)
    assert s1["failed"] == 1
    row = catalog.execute("SELECT dominant_colors FROM photo WHERE id=?", (pid,)).fetchone()
    assert row["dominant_colors"] == "[]"
    s2 = backfill_colors(batch_size=10)
    assert s2["remaining"] == 0 and s2["failed"] == 0
