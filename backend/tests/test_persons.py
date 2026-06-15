"""Person ops: naming, renaming, merging, audit trail, API error mapping."""

import json

import pytest

from tests.conftest import seed_face, seed_photo


@pytest.fixture()
def two_clusters(catalog):
    """Two photos, two clusters (10 and 20), one face each + one shared photo."""
    p1 = seed_photo(catalog, path="/f/a.jpg")
    p2 = seed_photo(catalog, path="/f/b.jpg")
    f1 = seed_face(catalog, p1, cluster_id=10)
    f2 = seed_face(catalog, p2, cluster_id=20)
    return {"photos": (p1, p2), "faces": (f1, f2)}


def test_name_cluster_creates_person(catalog, two_clusters):
    from foti_backend.persons import name_cluster

    out = name_cluster(10, "Elias")
    assert out["faces_updated"] == 1
    row = catalog.execute("SELECT name FROM person WHERE id=?", (out["person_id"],)).fetchone()
    assert row["name"] == "Elias"
    fd = catalog.execute("SELECT person_id FROM face_detection WHERE cluster_id=10").fetchone()
    assert fd["person_id"] == out["person_id"]


def test_unname_person_detaches_faces_and_deletes(catalog, two_clusters):
    from foti_backend.persons import name_cluster, unname_person

    pid = name_cluster(10, "Elias")["person_id"]
    out = unname_person(pid)
    assert out["faces_detached"] == 1
    assert catalog.execute("SELECT 1 FROM person WHERE id=?", (pid,)).fetchone() is None
    fd = catalog.execute("SELECT person_id FROM face_detection WHERE cluster_id=10").fetchone()
    assert fd["person_id"] is None


def test_unname_unknown_person_raises(catalog):
    from foti_backend.persons import unname_person

    with pytest.raises(KeyError):
        unname_person(999)


def test_name_cluster_reuses_existing_person(catalog, two_clusters):
    from foti_backend.persons import name_cluster

    a = name_cluster(10, "Elias")
    b = name_cluster(20, "Elias")
    assert a["person_id"] == b["person_id"]


def test_name_unknown_cluster_raises(catalog):
    from foti_backend.persons import name_cluster

    with pytest.raises(KeyError):
        name_cluster(999, "Nobody")


def test_rename_person(catalog, two_clusters):
    from foti_backend.persons import name_cluster, rename_person

    pid = name_cluster(10, "Eliass")["person_id"]
    out = rename_person(pid, "Elias")
    assert out["name"] == "Elias"


def test_rename_clash_rejected(catalog, two_clusters):
    from foti_backend.persons import name_cluster, rename_person

    a = name_cluster(10, "Anna")["person_id"]
    name_cluster(20, "Ben")
    with pytest.raises(ValueError):
        rename_person(a, "Ben")


def test_merge_moves_faces_and_keeps_target_person(catalog, two_clusters):
    from foti_backend.persons import merge_clusters, name_cluster

    target_pid = name_cluster(20, "Target")["person_id"]
    out = merge_clusters(10, 20)
    assert out["faces_moved"] == 1
    assert out["person_id"] == target_pid
    rows = catalog.execute(
        "SELECT cluster_id, person_id FROM face_detection").fetchall()
    assert all(r["cluster_id"] == 20 and r["person_id"] == target_pid for r in rows)


def test_merge_carries_source_name_when_target_unnamed(catalog, two_clusters):
    from foti_backend.persons import merge_clusters, name_cluster

    src_pid = name_cluster(10, "OnlySource")["person_id"]
    out = merge_clusters(10, 20)
    assert out["person_id"] == src_pid


def test_merge_audited(catalog, two_clusters):
    """T-06: a merge leaves a queryable audit row."""
    from foti_backend.persons import merge_clusters

    merge_clusters(10, 20)
    row = catalog.execute(
        "SELECT detail FROM person_audit WHERE action='merge'").fetchone()
    detail = json.loads(row["detail"])
    assert detail["source_cluster_id"] == 10
    assert detail["target_cluster_id"] == 20
    assert detail["faces_moved"] == 1


def test_merge_self_rejected(catalog, two_clusters):
    from foti_backend.persons import merge_clusters

    with pytest.raises(ValueError):
        merge_clusters(10, 10)


def test_api_flow(client, catalog, two_clusters):
    r = client.post("/faces/cluster/10/name", json={"name": "Anna"})
    assert r.status_code == 200
    pid = r.json()["person_id"]

    r = client.get("/persons")
    persons = r.json()["persons"]
    assert any(p["id"] == pid and p["name"] == "Anna" for p in persons)

    r = client.post(f"/persons/{pid}/rename", json={"name": "Anne"})
    assert r.status_code == 200 and r.json()["name"] == "Anne"

    r = client.post("/faces/cluster/10/merge", json={"target_cluster_id": 20})
    assert r.status_code == 200

    assert client.post("/faces/cluster/999/name", json={"name": "X"}).status_code == 404
    assert client.post(f"/persons/9999/rename", json={"name": "X"}).status_code == 404
    assert client.post("/faces/cluster/20/merge",
                       json={"target_cluster_id": 20}).status_code == 400


def test_people_listing_shows_name(client, catalog, two_clusters):
    """list_people (the People tab source) must surface the person name."""
    # Add a second face to cluster 10 so it passes the min_size=2 default.
    p3 = seed_photo(catalog, path="/f/c.jpg")
    seed_face(catalog, p3, cluster_id=10)
    client.post("/faces/cluster/10/name", json={"name": "Anna"})

    people = client.get("/faces/people").json()["people"]
    mine = [p for p in people if p["cluster_id"] == 10]
    assert mine and mine[0]["person_name"] == "Anna"
