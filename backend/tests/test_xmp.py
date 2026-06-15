"""XMP export: packet validity, content mapping, threat-bound safety tests."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

from foti_backend.xmp import aesthetic_to_rating, build_xmp, export_sidecars

from tests.conftest import seed_face, seed_photo

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
DC = "{http://purl.org/dc/elements/1.1/}"
XMPNS = "{http://ns.adobe.com/xap/1.0/}"
IPTC = "{http://iptc.org/std/Iptc4xmpExt/2008-02-29/}"


def _parse_packet(text: str) -> ET.Element:
    # Strip the xpacket processing instructions, parse the xmpmeta element.
    start = text.index("<x:xmpmeta")
    end = text.index("<?xpacket end")
    return ET.fromstring(text[start:end])


def test_build_xmp_is_valid_xml_with_all_fields():
    packet = build_xmp(tags=["beach", "sunset"], caption="Evening at the sea",
                       rating=4, persons=["Anna"], date_created="2024-07-01T19:00:00")
    root = _parse_packet(packet)
    subjects = [li.text for li in root.iter(f"{RDF}li")]
    assert "beach" in subjects and "Anna" in subjects
    assert "Evening at the sea" in subjects
    ratings = [e.text for e in root.iter(f"{XMPNS}Rating")]
    assert ratings == ["4"]
    assert root.iter(f"{IPTC}PersonInImage") is not None


def test_build_xmp_omits_empty_fields():
    packet = build_xmp(tags=[], caption=None, rating=None, persons=[], date_created=None)
    root = _parse_packet(packet)
    assert list(root.iter(f"{DC}subject")) == []
    assert list(root.iter(f"{XMPNS}Rating")) == []


def test_aesthetic_rating_mapping():
    assert aesthetic_to_rating(None) is None
    assert aesthetic_to_rating(1.0) == 0  # round(0.5) banker's → 0
    assert aesthetic_to_rating(5.0) == 2
    assert aesthetic_to_rating(10.0) == 5
    assert aesthetic_to_rating(99.0) == 5  # clamped


def test_export_mirrors_tree_and_sets_permissions(catalog, tmp_path):
    """T-03: sidecars written 0o600 in 0o700 dirs, mirrored under out_dir."""
    pid = seed_photo(catalog, path="/photos/2024/img1.jpg", caption="hi",
                     aesthetic=8.0, captured_at="2024-01-01T00:00:00")
    catalog.execute("INSERT INTO photo_tag_auto (photo_id, label, score) VALUES (?, 'dog', 0.9)",
                    (pid,))
    catalog.commit()

    out = tmp_path / "export"
    summary = export_sidecars(out_dir=out, photo_ids=[pid])
    assert summary["written"] == 1

    sidecar = out / "photos/2024/img1.xmp"
    assert sidecar.is_file()
    assert (sidecar.stat().st_mode & 0o777) == 0o600
    assert (sidecar.parent.stat().st_mode & 0o777) == 0o700

    root = _parse_packet(sidecar.read_text())
    subjects = [li.text for li in root.iter(f"{RDF}li")]
    assert "dog" in subjects and "hi" in subjects
    assert [e.text for e in root.iter(f"{XMPNS}Rating")] == ["4"]


def test_export_includes_named_persons(catalog, tmp_path):
    from foti_backend.persons import name_cluster

    pid = seed_photo(catalog, path="/p/face.jpg")
    seed_face(catalog, pid, cluster_id=5)
    name_cluster(5, "Elias")

    export_sidecars(out_dir=tmp_path / "x", photo_ids=[pid])
    root = _parse_packet((tmp_path / "x/p/face.xmp").read_text())
    pii = list(root.iter(f"{IPTC}PersonInImage"))
    assert pii and [li.text for li in pii[0].iter(f"{RDF}li")] == ["Elias"]


def test_auto_tags_below_min_score_excluded(catalog, tmp_path):
    pid = seed_photo(catalog, path="/p/a.jpg")
    catalog.executemany(
        "INSERT INTO photo_tag_auto (photo_id, label, score) VALUES (?, ?, ?)",
        [(pid, "strong", 0.9), (pid, "weak", 0.2)])
    catalog.commit()

    export_sidecars(out_dir=tmp_path / "x", photo_ids=[pid])
    subjects = [li.text for li in
                _parse_packet((tmp_path / "x/p/a.xmp").read_text()).iter(f"{RDF}li")]
    assert "strong" in subjects and "weak" not in subjects


def test_no_overwrite_non_xmp(catalog, tmp_path, monkeypatch):
    """T-02: an existing non-xmp file at the target path is never clobbered."""
    pid = seed_photo(catalog, path="/p/clash.jpg")
    out = tmp_path / "x"
    target_dir = out / "p"
    target_dir.mkdir(parents=True)
    decoy = target_dir / "clash.xmp"

    # Simulate the rare path: target exists but is NOT an .xmp — use a dir trick:
    # patch _sidecar_target to point at a .txt file.
    import foti_backend.xmp as xmp_mod
    real = xmp_mod._sidecar_target

    def patched(photo_path, out_dir):
        return real(photo_path, out_dir).with_suffix(".txt")

    monkeypatch.setattr(xmp_mod, "_sidecar_target", patched)
    sentinel = target_dir / "clash.txt"
    sentinel.write_text("precious")

    summary = export_sidecars(out_dir=out, photo_ids=[pid])
    assert summary["skipped"] == 1 and summary["written"] == 0
    assert sentinel.read_text() == "precious"


def test_export_endpoint_validation(client, catalog, tmp_path):
    # No selector → 400
    r = client.post("/export/xmp", json={"out_dir": str(tmp_path)})
    assert r.status_code == 400
    # Relative out_dir → 400
    r = client.post("/export/xmp", json={"out_dir": "rel/dir", "all": True})
    assert r.status_code == 400
    # Neither out_dir nor next_to_original → 400
    r = client.post("/export/xmp", json={"all": True})
    assert r.status_code == 400


def test_export_endpoint_writes(client, catalog, tmp_path):
    pid = seed_photo(catalog, path="/p/e.jpg", caption="endpoint test")
    r = client.post("/export/xmp", json={"out_dir": str(tmp_path / "out"),
                                          "photo_ids": [pid]})
    assert r.status_code == 200
    assert r.json()["written"] == 1
    assert (tmp_path / "out/p/e.xmp").is_file()


def test_export_next_to_original(client, catalog, tmp_path):
    src = tmp_path / "orig"
    src.mkdir()
    pid = seed_photo(catalog, path=str(src / "n.jpg"))
    r = client.post("/export/xmp", json={"next_to_original": True, "photo_ids": [pid]})
    assert r.status_code == 200 and r.json()["written"] == 1
    assert (src / "n.xmp").is_file()
