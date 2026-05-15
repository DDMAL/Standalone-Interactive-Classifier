"""Integration tests for the FastAPI service.

Each test runs the full app with FastAPI's :class:`TestClient`,
backed by a fresh :class:`InMemorySessionStore` so tests don't
share state. The shared dependency-override pattern lives in the
:func:`client` fixture.

Real ingest is exercised via the sample input under
``core/tests/sample_input/`` so the tests cover the full
HTTP → ingest → session → response path, not a mocked happy case.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ic_api.main import app, get_store
from ic_api.store import InMemorySessionStore

SAMPLE_DIR = (
    Path(__file__).resolve().parents[2]
    / "core"
    / "tests"
    / "sample_input"
)
PAGE_IMAGE = SAMPLE_DIR / "NZ-Wt MSR-03 109v.png"
JSON_PATH = SAMPLE_DIR / "MOTHRA_NZ-Wt MSR-03 109v_annotations.json"


@pytest.fixture
def store() -> InMemorySessionStore:
    """A fresh, per-test store so tests don't leak sessions into each other."""
    return InMemorySessionStore()


@pytest.fixture
def client(store: InMemorySessionStore) -> TestClient:
    """A TestClient wired to the per-test store."""
    app.dependency_overrides[get_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _create_session(client: TestClient) -> str:
    """Helper: create a session from the sample input, return its id."""
    response = client.post(
        "/sessions",
        json={
            "page_image": str(PAGE_IMAGE),
            "annotations": str(JSON_PATH),
            "class_names": ["neume.A", "neume.B"],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_create_session_returns_classifying_state_with_glyphs(client):
    response = client.post(
        "/sessions",
        json={
            "page_image": str(PAGE_IMAGE),
            "annotations": str(JSON_PATH),
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["state"] == "classifying"
    assert len(body["glyphs"]) > 0
    # Every glyph carries the per-page bbox origin from the ingest path.
    first = body["glyphs"][0]
    assert "ulx" in first and "uly" in first
    assert "image_b64" in first


def test_get_session_returns_the_same_payload(client):
    sid = _create_session(client)
    a = client.get(f"/sessions/{sid}").json()
    b = client.get(f"/sessions/{sid}").json()
    assert a == b


def test_get_session_404_for_unknown_id(client):
    response = client.get("/sessions/nope")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_delete_session_removes_it(client):
    sid = _create_session(client)
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404


# ---------------------------------------------------------------------------
# Glyph editing
# ---------------------------------------------------------------------------


def test_update_glyph_to_manual_pins_confidence(client):
    sid = _create_session(client)
    gid = client.get(f"/sessions/{sid}").json()["glyphs"][0]["id"]

    response = client.post(
        f"/sessions/{sid}/glyphs/{gid}",
        json={"class_name": "neume.A", "id_state_manual": True},
    )
    assert response.status_code == 200
    g = response.json()
    assert g["class_name"] == "neume.A"
    assert g["id_state_manual"] is True
    assert g["confidence"] == 1.0
    assert g["id"] == gid  # UUID preserved


def test_update_glyph_404_for_unknown_id(client):
    sid = _create_session(client)
    response = client.post(
        f"/sessions/{sid}/glyphs/nope",
        json={"class_name": "X"},
    )
    assert response.status_code == 404


def test_delete_glyph_removes_from_working_set(client):
    sid = _create_session(client)
    sess = client.get(f"/sessions/{sid}").json()
    gid = sess["glyphs"][0]["id"]
    n_before = len(sess["glyphs"])

    assert client.delete(f"/sessions/{sid}/glyphs/{gid}").status_code == 204

    sess_after = client.get(f"/sessions/{sid}").json()
    assert len(sess_after["glyphs"]) == n_before - 1
    assert gid not in {g["id"] for g in sess_after["glyphs"]}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classify_with_no_training_data_returns_400(client):
    # Fresh session has only UNCLASSIFIED glyphs — training pool is
    # empty, so classify should fail loudly rather than silently
    # producing garbage.
    sid = _create_session(client)
    response = client.post(f"/sessions/{sid}/classify", json={})
    assert response.status_code == 400
    assert response.json()["code"] == "validation_error"


def test_classify_with_one_manual_label_succeeds(client):
    sid = _create_session(client)
    glyphs = client.get(f"/sessions/{sid}").json()["glyphs"]
    # Manually label the first two glyphs so classify has training data.
    for g in glyphs[:2]:
        r = client.post(
            f"/sessions/{sid}/glyphs/{g['id']}",
            json={"class_name": "neume.A", "id_state_manual": True},
        )
        assert r.status_code == 200

    response = client.post(f"/sessions/{sid}/classify", json={"k": 1})
    assert response.status_code == 200
    sess = response.json()
    # Every non-manual glyph should now have a non-UNCLASSIFIED class.
    auto_classes = {
        g["class_name"] for g in sess["glyphs"] if not g["id_state_manual"]
    }
    assert auto_classes == {"neume.A"}


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_manual_group_replaces_originals(client):
    sid = _create_session(client)
    sess = client.get(f"/sessions/{sid}").json()
    a, b = sess["glyphs"][0]["id"], sess["glyphs"][1]["id"]
    n_before = len(sess["glyphs"])

    response = client.post(
        f"/sessions/{sid}/group",
        json={"glyph_ids": [a, b], "class_name": "neume.compound"},
    )
    assert response.status_code == 200
    new_glyph = response.json()
    assert new_glyph["id_state_manual"] is True
    assert new_glyph["confidence"] == 1.0

    sess_after = client.get(f"/sessions/{sid}").json()
    ids_after = {g["id"] for g in sess_after["glyphs"]}
    assert a not in ids_after
    assert b not in ids_after
    assert new_glyph["id"] in ids_after
    # Two glyphs removed, one added = net -1.
    assert len(sess_after["glyphs"]) == n_before - 1


def test_auto_group_returns_501(client):
    sid = _create_session(client)
    response = client.post(f"/sessions/{sid}/auto-group")
    assert response.status_code == 501
    assert response.json()["code"] == "deferred"


def test_auto_group_unknown_session_returns_404_not_501(client):
    response = client.post("/sessions/nope/auto-group")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Class management
# ---------------------------------------------------------------------------


def test_rename_class_propagates_to_class_names(client):
    sid = _create_session(client)
    # Seed a glyph with a manual class so it shows up in class_names.
    glyphs = client.get(f"/sessions/{sid}").json()["glyphs"]
    client.post(
        f"/sessions/{sid}/glyphs/{glyphs[0]['id']}",
        json={"class_name": "neume.A", "id_state_manual": True},
    )

    response = client.post(
        f"/sessions/{sid}/classes/neume.A/rename",
        json={"new_name": "punctum"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "punctum" in body["class_names"]
    assert "neume.A" not in body["class_names"]


def test_delete_class_drops_it_from_imported_list(client):
    sid = _create_session(client)
    # neume.A was seeded as an imported class name in _create_session.
    pre = client.get(f"/sessions/{sid}").json()
    assert "neume.A" in pre["class_names"]

    response = client.delete(f"/sessions/{sid}/classes/neume.A")
    assert response.status_code == 200
    assert "neume.A" not in response.json()["class_names"]


# ---------------------------------------------------------------------------
# Save & complete
# ---------------------------------------------------------------------------


def test_save_is_a_noop_returning_current_state(client):
    sid = _create_session(client)
    before = client.get(f"/sessions/{sid}").json()
    after = client.post(f"/sessions/{sid}/save").json()
    assert before == after


def test_complete_returns_xml_and_transitions_to_export(client):
    sid = _create_session(client)
    # Need at least one labelled glyph for export to be meaningful.
    g = client.get(f"/sessions/{sid}").json()["glyphs"][0]
    client.post(
        f"/sessions/{sid}/glyphs/{g['id']}",
        json={"class_name": "neume.A", "id_state_manual": True},
    )

    response = client.post(f"/sessions/{sid}/complete")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    body = response.content
    assert body.startswith(b"<?xml")
    assert b"<gamera-database" in body
    assert b'name="neume.A"' in body

    # Subsequent mutating endpoints should now 409 (state conflict).
    classify_resp = client.post(f"/sessions/{sid}/classify", json={})
    assert classify_resp.status_code == 409
    assert classify_resp.json()["code"] == "state_conflict"
