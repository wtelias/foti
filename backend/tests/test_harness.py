"""Harness sanity: schema applies on a fresh catalog, API answers, auth gates."""

from tests.conftest import seed_photo


def test_schema_applies(catalog):
    tables = {r["name"] for r in catalog.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"photo", "face_detection", "person", "scan_job"} <= tables


def test_seed_photo(catalog):
    pid = seed_photo(catalog, path="/x/a.jpg", phash="00000000000000ff")
    row = catalog.execute("SELECT * FROM photo WHERE id=?", (pid,)).fetchone()
    assert row["phash"] == "00000000000000ff"


def test_health_requires_auth(client):
    import httpx
    r = httpx.Client(transport=client._transport, base_url="http://testserver").get("/health")
    assert r.status_code == 401


def test_health_with_auth(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
