"""Integration tests for the FastAPI service.

Each test runs the full app with FastAPI's :class:`TestClient`,
backed by a fresh :class:`InMemorySessionStore` so tests don't
share state. The shared dependency-override pattern lives in the
:func:`client` fixture.

Real ingest is exercised via the sample input under
``core/data/test/`` so the tests cover the full HTTP → ingest →
session → response path, not a mocked happy case.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ic_api.main import app, get_store
from ic_api.store import InMemorySessionStore

TEST_DIR = (
    Path(__file__).resolve().parents[2]
    / "core"
    / "data"
    / "test"
)
PAGE_IMAGE = TEST_DIR / "NZ-Wt MSR-03 109v.png"
JSON_PATH = TEST_DIR / "MOTHRA_NZ-Wt MSR-03 109v_annotations.json"

# Read once at module load — multipart uploads ship bytes, and we
# replay the same payload across most tests.
PAGE_BYTES = PAGE_IMAGE.read_bytes()
JSON_BYTES = JSON_PATH.read_bytes()


def _multipart(
    *,
    class_names: list[str] | None = None,
    annotations_format: str = "json",
    annotations_filename: str = "annotations.json",
) -> dict:
    """Build kwargs for ``TestClient.post`` that emulate a browser upload.

    httpx accepts the ``files`` and ``data`` dict pair to assemble a
    proper ``multipart/form-data`` body — this is what the frontend
    will send once it exists.
    """
    files = {
        "page_image": ("page.png", PAGE_BYTES, "image/png"),
        "annotations": (annotations_filename, JSON_BYTES, "application/json"),
    }
    data: dict[str, str] = {"annotations_format": annotations_format}
    if class_names is not None:
        # See main.py note: class_names is a JSON-encoded string,
        # not a repeated form field, to work around a FastAPI bug
        # in which ``list[X]`` Form params combined with UploadFile
        # break multipart body parsing.
        data["class_names"] = json.dumps(class_names)
    return {"files": files, "data": data}


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
        **_multipart(class_names=["neume.A", "neume.B"]),
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_healthz_reports_the_live_store_backend(client):
    # The probe exists to answer "is this deployment persisting sessions?"
    # from outside the process — the mothra deployments select the backend
    # purely by whether DATABASE_URL is set, and an in-memory store means a
    # restart drops every session.
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["store"]["backend"] in {"in-memory", "postgres"}
    assert isinstance(body["store"]["persistent"], bool)


def test_healthz_counts_sessions_held_by_this_process(client):
    before = client.get("/healthz").json()["sessions"]
    _create_session(client)
    assert client.get("/healthz").json()["sessions"] == before + 1


def test_healthz_does_not_require_a_session(client):
    # Probes must not depend on any session existing, and must stay cheap —
    # k8s hits this on an interval.
    assert client.get("/healthz").status_code == 200


def test_healthz_reachable_is_null_without_a_database(client):
    # The in-memory store has no database to be unreachable, so `reachable`
    # must be null rather than a misleading true/false.
    assert client.get("/healthz").json()["store"]["reachable"] is None


def test_healthz_reports_reachable_when_the_database_answers(client, store):
    # Holding a Postgres store proves nothing — it connects lazily — so the
    # probe round-trips the DB and reports what it found.
    class _Reachable(InMemorySessionStore):
        def ping(self) -> None:
            return None

    app.dependency_overrides[get_store] = _Reachable
    try:
        body = client.get("/healthz").json()
    finally:
        app.dependency_overrides[get_store] = lambda: store
    assert body["store"]["reachable"] is True
    assert "error" not in body["store"]


def test_healthz_reports_unreachable_database_without_failing_the_probe(
    client, store
):
    # The case the `persistent` flag alone can't express: the deployment
    # believes it configured persistence, but the DSN is wrong or the server
    # is down. status stays "ok" so wiring this up as a liveness probe can't
    # turn a DB hiccup into a restart loop.
    class _Unreachable(InMemorySessionStore):
        def ping(self) -> None:
            raise RuntimeError("could not connect to server: Connection refused")

    app.dependency_overrides[get_store] = _Unreachable
    try:
        response = client.get("/healthz")
    finally:
        app.dependency_overrides[get_store] = lambda: store
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["store"]["reachable"] is False
    assert "Connection refused" in body["store"]["error"]


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_create_session_returns_classifying_state_with_glyphs(client):
    response = client.post("/sessions", **_multipart())
    assert response.status_code == 201
    body = response.json()
    assert body["state"] == "classifying"
    assert len(body["glyphs"]) > 0
    # Every glyph carries the per-page bbox origin from the ingest path.
    first = body["glyphs"][0]
    assert "ulx" in first and "uly" in first
    assert "image_b64" in first


def test_create_session_rejects_unknown_annotations_format(client):
    # The endpoint constrains annotations_format to {"json","yolo"};
    # anything else should 422 from FastAPI's Literal validation.
    response = client.post(
        "/sessions",
        **_multipart(annotations_format="csv"),
    )
    assert response.status_code == 422


def test_create_session_does_not_accept_path_strings(client):
    # Regression guard: the old JSON-body API took server-side
    # filesystem paths. Sending one as a plain JSON post must fail
    # — proving the path-based read primitive is gone.
    response = client.post(
        "/sessions",
        json={
            "page_image": str(PAGE_IMAGE),
            "annotations": str(JSON_PATH),
        },
    )
    assert response.status_code == 422


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
# Training-glyph delete
# ---------------------------------------------------------------------------

# A small (99-glyph) GameraXML training set uploaded so the created session
# has a non-empty training pool to delete from.
TRAIN_XML = (
    Path(__file__).resolve().parents[2]
    / "core"
    / "tests"
    / "fixtures"
    / "Hufnagel-example_training_data.xml"
)
TRAIN_XML_BYTES = TRAIN_XML.read_bytes()


def _create_session_with_training(client: TestClient) -> str:
    """Create a session seeded with an uploaded training set; return its id."""
    files = {
        "page_image": ("page.png", PAGE_BYTES, "image/png"),
        "annotations": ("annotations.json", JSON_BYTES, "application/json"),
        "training_files": ("train.xml", TRAIN_XML_BYTES, "application/xml"),
    }
    response = client.post(
        "/sessions", files=files, data={"annotations_format": "json"}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_delete_training_glyph_removes_from_pool(client):
    sid = _create_session_with_training(client)
    sess = client.get(f"/sessions/{sid}").json()
    training = sess["training_glyphs"]
    assert len(training) >= 2, "uploaded training set should load glyphs"
    n_before = len(training)
    uploaded_before = sess["uploaded_training_count"]
    gid = training[0]["id"]

    response = client.delete(f"/sessions/{sid}/training-glyphs/{gid}")
    assert response.status_code == 200, response.text
    after = response.json()
    assert len(after["training_glyphs"]) == n_before - 1
    assert gid not in {g["id"] for g in after["training_glyphs"]}
    # Provenance count follows the pool down, so the export screen stays honest.
    assert after["uploaded_training_count"] == uploaded_before - 1
    # The working set is untouched — only the training pool shrinks.
    assert len(after["glyphs"]) == len(sess["glyphs"])


def test_delete_training_glyph_unknown_id_returns_404(client):
    sid = _create_session_with_training(client)
    response = client.delete(f"/sessions/{sid}/training-glyphs/nope")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


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
    # Only Neumes are classified, so the training pool must be seeded with
    # manually-labelled *neume* glyphs — labelling Text/Staves would leave
    # the neume classifier with nothing to learn from.
    neumes = [g for g in glyphs if g["category"] == "Neumes"]
    assert len(neumes) >= 2, "fixture should contain neume glyphs"
    for g in neumes[:2]:
        r = client.post(
            f"/sessions/{sid}/glyphs/{g['id']}",
            json={"class_name": "neume.A", "id_state_manual": True},
        )
        assert r.status_code == 200

    response = client.post(f"/sessions/{sid}/classify", json={"k": 1})
    assert response.status_code == 200
    sess = response.json()

    # Every non-manual *neume* should now carry the trained label.
    auto_neume_classes = {
        g["class_name"]
        for g in sess["glyphs"]
        if not g["id_state_manual"] and g["category"] == "Neumes"
    }
    assert auto_neume_classes == {"neume.A"}

    # Text and Staves are out of IC's scope: they stay UNCLASSIFIED.
    non_neume_classes = {
        g["class_name"] for g in sess["glyphs"] if g["category"] != "Neumes"
    }
    assert non_neume_classes == {"UNCLASSIFIED"}


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


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def _pick_split_target(client: TestClient, sid: str) -> dict:
    """Pick a working-set glyph with a bbox we can carve into halves."""
    glyphs = client.get(f"/sessions/{sid}").json()["glyphs"]
    for g in glyphs:
        if g["ncols"] >= 2 and g["nrows"] >= 2:
            return g
    pytest.fail("fixture has no glyph large enough to split")


def test_manual_split_replaces_parent_with_children(client):
    sid = _create_session(client)
    parent = _pick_split_target(client, sid)
    # Two side-by-side rectangles each covering half the parent's width.
    half = parent["ncols"] // 2
    regions = [
        [parent["ulx"], parent["uly"], half, parent["nrows"]],
        [parent["ulx"] + half, parent["uly"], parent["ncols"] - half, parent["nrows"]],
    ]

    n_before = len(client.get(f"/sessions/{sid}").json()["glyphs"])
    response = client.post(
        f"/sessions/{sid}/glyphs/{parent['id']}/split",
        json={"regions": regions},
    )
    assert response.status_code == 200, response.text
    children = response.json()
    assert len(children) == 2
    # Algorithm semantic #8: children are UNCLASSIFIED, auto, fresh UUIDs.
    for child in children:
        assert child["class_name"] == "UNCLASSIFIED"
        assert child["confidence"] == 0.0
        assert child["id_state_manual"] is False
        assert child["id"] != parent["id"]

    sess_after = client.get(f"/sessions/{sid}").json()
    ids_after = {g["id"] for g in sess_after["glyphs"]}
    assert parent["id"] not in ids_after
    for child in children:
        assert child["id"] in ids_after
    # One parent removed, two children added = net +1.
    assert len(sess_after["glyphs"]) == n_before + 1


def test_manual_split_unknown_glyph_returns_404(client):
    sid = _create_session(client)
    response = client.post(
        f"/sessions/{sid}/glyphs/nope/split",
        json={"regions": [[0, 0, 5, 5]]},
    )
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_manual_split_unknown_session_returns_404(client):
    response = client.post(
        "/sessions/nope/glyphs/whatever/split",
        json={"regions": [[0, 0, 5, 5]]},
    )
    assert response.status_code == 404


def test_manual_split_empty_regions_returns_422(client):
    # Empty list is rejected by Pydantic ``min_length=1`` before the
    # handler runs — that's a 422 (request validation), not the 400
    # that the core function would have raised.
    sid = _create_session(client)
    parent = _pick_split_target(client, sid)
    response = client.post(
        f"/sessions/{sid}/glyphs/{parent['id']}/split",
        json={"regions": []},
    )
    assert response.status_code == 422


def test_manual_split_all_regions_miss_returns_400(client):
    # Business rule: every region misses the parent → silently
    # deleting the parent would be a UI bug. The core surfaces this
    # as ValueError; the API maps to 400 / validation_error.
    sid = _create_session(client)
    parent = _pick_split_target(client, sid)
    far_away = parent["ulx"] + parent["ncols"] + 1000
    response = client.post(
        f"/sessions/{sid}/glyphs/{parent['id']}/split",
        json={"regions": [[far_away, far_away, 5, 5]]},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "validation_error"
    # Parent must still be present.
    ids = {g["id"] for g in client.get(f"/sessions/{sid}").json()["glyphs"]}
    assert parent["id"] in ids


def test_manual_split_after_complete_returns_409(client):
    sid = _create_session(client)
    parent = _pick_split_target(client, sid)
    # Seed a manual label so /complete has something meaningful to export.
    client.post(
        f"/sessions/{sid}/glyphs/{parent['id']}",
        json={"class_name": "neume.A", "id_state_manual": True},
    )
    assert client.post(f"/sessions/{sid}/complete?page=true").status_code == 200

    response = client.post(
        f"/sessions/{sid}/glyphs/{parent['id']}/split",
        json={"regions": [[parent["ulx"], parent["uly"], 1, 1]]},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "state_conflict"


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


# ---------------------------------------------------------------------------
# Session resume (lookup by owning project + page)
# ---------------------------------------------------------------------------


def _stage(client, *, project_id=None, image_id=None) -> str:
    """Stage a page + bboxes (optionally keyed to a project/image) → staging id."""
    data = {"annotations_format": "json"}
    if project_id is not None:
        data["project_id"] = str(project_id)
    if image_id is not None:
        data["image_id"] = image_id
    response = client.post(
        "/staging",
        files={
            "page_image": ("page.png", PAGE_BYTES, "image/png"),
            "annotations": ("annotations.json", JSON_BYTES, "application/json"),
        },
        data=data,
    )
    assert response.status_code == 201, response.text
    return response.json()["staging_id"]


def test_lookup_resumes_a_keyed_staged_session(client):
    staging_id = _stage(client, project_id=7, image_id="img-abc")
    created = client.post("/sessions/from-staging", data={"staging_id": staging_id})
    assert created.status_code == 201, created.text
    sid = created.json()["id"]

    found = client.get(
        "/sessions/lookup", params={"project_id": 7, "image_id": "img-abc"}
    )
    assert found.status_code == 200
    assert found.json()["session_id"] == sid


def test_lookup_404_for_unknown_page(client):
    found = client.get(
        "/sessions/lookup", params={"project_id": 999, "image_id": "nope"}
    )
    assert found.status_code == 404


def test_lookup_skips_completed_sessions(client):
    staging_id = _stage(client, project_id=8, image_id="img-done")
    sid = client.post(
        "/sessions/from-staging", data={"staging_id": staging_id}
    ).json()["id"]
    # A completed session is terminal (EXPORT) → not offered for resume.
    done = client.post(f"/sessions/{sid}/complete", params={"page": True})
    assert done.status_code == 200, done.text
    found = client.get(
        "/sessions/lookup", params={"project_id": 8, "image_id": "img-done"}
    )
    assert found.status_code == 404


# ---------------------------------------------------------------------------
# Session listing (standalone "resume a saved session")
# ---------------------------------------------------------------------------


def test_list_sessions_empty_by_default(client):
    response = client.get("/sessions")
    assert response.status_code == 200
    assert response.json() == []


def test_list_sessions_returns_created_sessions(client):
    sid1 = _create_session(client)
    sid2 = _create_session(client)

    response = client.get("/sessions")
    assert response.status_code == 200
    body = response.json()
    ids = {row["id"] for row in body}
    assert ids == {sid1, sid2}

    row = next(r for r in body if r["id"] == sid1)
    # Summary carries the metadata a resume list needs — and no glyph masks.
    assert row["state"] == "classifying"
    assert row["n_glyphs"] > 0
    assert "glyphs" not in row
    # IC's own upload path is unkeyed; the in-memory store has no timestamp.
    assert row["project_id"] is None
    assert row["image_id"] is None
    assert row["updated_at"] is None


def test_list_sessions_drops_deleted_session(client):
    sid = _create_session(client)
    assert client.delete(f"/sessions/{sid}").status_code == 204
    response = client.get("/sessions")
    assert response.status_code == 200
    assert all(row["id"] != sid for row in response.json())


def test_list_sessions_includes_completed_sessions(client):
    sid = _create_session(client)
    g = client.get(f"/sessions/{sid}").json()["glyphs"][0]
    client.post(
        f"/sessions/{sid}/glyphs/{g['id']}",
        json={"class_name": "neume.A", "id_state_manual": True},
    )
    assert client.post(f"/sessions/{sid}/complete?page=true").status_code == 200

    # Unlike /sessions/lookup, the list surfaces completed sessions (state
    # exposed) so the user can still reopen a finished page read-only.
    row = next(r for r in client.get("/sessions").json() if r["id"] == sid)
    assert row["state"] == "export"


def test_clear_sessions_removes_all(client):
    sid1 = _create_session(client)
    sid2 = _create_session(client)

    response = client.delete("/sessions")
    assert response.status_code == 200
    assert response.json() == {"deleted": 2}

    assert client.get("/sessions").json() == []
    assert client.get(f"/sessions/{sid1}").status_code == 404
    assert client.get(f"/sessions/{sid2}").status_code == 404


def test_clear_sessions_empty_store_is_a_noop(client):
    response = client.delete("/sessions")
    assert response.status_code == 200
    assert response.json() == {"deleted": 0}


def test_complete_returns_xml_and_transitions_to_export(client):
    sid = _create_session(client)
    # Need at least one labelled glyph for export to be meaningful.
    g = client.get(f"/sessions/{sid}").json()["glyphs"][0]
    client.post(
        f"/sessions/{sid}/glyphs/{g['id']}",
        json={"class_name": "neume.A", "id_state_manual": True},
    )

    response = client.post(f"/sessions/{sid}/complete?page=true")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    body = response.content
    assert body.startswith(b"<?xml")
    assert b"<gamera-database" in body
    assert b'name="neume.A"' in body
    # The export is named after the uploaded bbox document (sans .json) plus
    # the selected sections, so a user exporting several pages/variants gets
    # self-describing files.
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="ic-session-annotations-page.xml"'
    )

    # Subsequent mutating endpoints should now 409 (state conflict).
    classify_resp = client.post(f"/sessions/{sid}/classify", json={})
    assert classify_resp.status_code == 409
    assert classify_resp.json()["code"] == "state_conflict"


def test_complete_with_finalize_false_keeps_the_session_editable(client):
    # An embedding host (mothra) exports the GameraXML as an *intermediate*
    # artefact — it feeds its encoder and the user still reopens the page to
    # correct it. Finalising there would move the session to EXPORT, which
    # /sessions/lookup refuses to resume, silently stranding every
    # correction behind the export.
    staging_id = _stage(client, project_id=11, image_id="img-editable")
    sid = client.post(
        "/sessions/from-staging", data={"staging_id": staging_id}
    ).json()["id"]
    g = client.get(f"/sessions/{sid}").json()["glyphs"][0]
    client.post(
        f"/sessions/{sid}/glyphs/{g['id']}",
        json={"class_name": "neume.A", "id_state_manual": True},
    )

    response = client.post(f"/sessions/{sid}/complete?page=true&finalize=false")
    assert response.status_code == 200, response.text
    assert response.content.startswith(b"<?xml")
    assert b'name="neume.A"' in response.content

    # Still CLASSIFYING: mutations work and the page stays resumable.
    assert client.get(f"/sessions/{sid}").json()["state"] == "classifying"
    relabel = client.post(
        f"/sessions/{sid}/glyphs/{g['id']}",
        json={"class_name": "neume.B", "id_state_manual": True},
    )
    assert relabel.status_code == 200, relabel.text
    found = client.get(
        "/sessions/lookup",
        params={"project_id": 11, "image_id": "img-editable"},
    )
    assert found.status_code == 200
    assert found.json()["session_id"] == sid

    # And a later finalising export still transitions it, so the two modes
    # coexist on one session rather than the flag latching.
    assert client.post(f"/sessions/{sid}/complete?page=true").status_code == 200
    assert client.get(f"/sessions/{sid}").json()["state"] == "export"


def test_complete_export_filename_derives_from_uploaded_name(client):
    # A real-world upload name carries a directory part, spaces and the
    # .json extension; the export keeps a safe stem of it so the file is
    # recognisable without the user renaming it by hand.
    response = client.post(
        "/sessions",
        **_multipart(annotations_filename="MOTHRA_NZ-Wt MSR-03 109v.json"),
    )
    sid = response.json()["id"]

    response = client.post(f"/sessions/{sid}/complete?page=true")
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="ic-session-MOTHRA_NZ-Wt_MSR-03_109v-page.xml"'
    )


def test_complete_export_filename_tags_selected_sections(client):
    sid = client.post(
        "/sessions",
        **_multipart(annotations_filename="page42.json"),
    ).json()["id"]

    response = client.post(
        f"/sessions/{sid}/complete?page=true&manual_neumes=true"
    )
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="ic-session-page42-page-manual-neumes.xml"'
    )


def test_complete_requires_at_least_one_section(client):
    # With no section flags there is nothing to export — reject rather than
    # emit an empty document.
    sid = _create_session(client)
    response = client.post(f"/sessions/{sid}/complete")
    assert response.status_code == 400


def test_complete_is_repeatable_for_multiple_exports(client):
    # The export menu can download several section combinations from the same
    # finalised session. Completing is a one-shot cleanup, but the download
    # must stay repeatable — a second /complete re-serialises rather than
    # 409ing.
    sid = _create_session(client)

    first = client.post(f"/sessions/{sid}/complete?page=true")
    assert first.status_code == 200
    assert first.content.startswith(b"<?xml")

    second = client.post(f"/sessions/{sid}/complete?manual_neumes=true")
    assert second.status_code == 200
    assert second.content.startswith(b"<?xml")

    # Mutations remain forbidden after completion.
    assert client.post(f"/sessions/{sid}/classify", json={}).status_code == 409


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_store_session_context_manager_serializes_same_id():
    """Two threads acquiring the same id must not interleave their critical sections.

    Why: the public API hands out the same mutable :class:`Session`
    object to every caller. Without serialisation, a browser
    double-click or async UI retry can interleave mutations and
    corrupt session state. The store's ``session()`` context manager
    is the chokepoint, so this test pins its mutual-exclusion
    guarantee directly.
    """
    import threading
    import time

    from ic_core.state import Session

    store = InMemorySessionStore()
    sess = Session()
    store.create(sess)

    events: list[tuple[str, str]] = []
    events_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def hold(label: str) -> None:
        barrier.wait()
        with store.session(sess.id):
            with events_lock:
                events.append(("enter", label))
            # Sleep inside the critical section so any interleaving
            # would surface as an enter/enter pair.
            time.sleep(0.05)
            with events_lock:
                events.append(("exit", label))

    t1 = threading.Thread(target=hold, args=("A",))
    t2 = threading.Thread(target=hold, args=("B",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    # Expect strictly enter/exit/enter/exit with no interleave —
    # whichever thread wins the lock first finishes before the other starts.
    assert [e[0] for e in events] == ["enter", "exit", "enter", "exit"]
    assert events[0][1] == events[1][1]
    assert events[2][1] == events[3][1]
    assert events[0][1] != events[2][1]


def test_store_session_context_manager_allows_parallelism_across_ids():
    """Different session ids must not block each other.

    Why: serialising *all* session operations on a single lock would
    needlessly stall concurrent users (or concurrent tabs over the
    same backend). Per-session locks let different sessions proceed
    in parallel; this test pins that.
    """
    import threading
    import time

    from ic_core.state import Session

    store = InMemorySessionStore()
    a, b = Session(), Session()
    store.create(a); store.create(b)

    start = threading.Barrier(2)
    durations: dict[str, float] = {}

    def hold(sid: str, label: str) -> None:
        start.wait()
        t0 = time.monotonic()
        with store.session(sid):
            time.sleep(0.1)
        durations[label] = time.monotonic() - t0

    ta = threading.Thread(target=hold, args=(a.id, "a"))
    tb = threading.Thread(target=hold, args=(b.id, "b"))
    ta.start(); tb.start()
    ta.join(); tb.join()

    # If the locks serialised across ids, total wall time would be
    # ~2× the sleep. Both threads should finish in roughly one sleep.
    assert max(durations.values()) < 0.18, durations


def test_concurrent_updates_on_same_session_are_consistent(client):
    """Hammer one session from many threads; final state must add up.

    Without the per-session lock, concurrent ``update_glyph`` calls
    on the same session could see torn intermediate state (the
    handler reads, mutates, and serialises the same mutable object).
    With locking each request observes a consistent snapshot.

    Each worker uses its own ``TestClient`` — the underlying
    ``requests.Session`` isn't thread-safe, so sharing one client
    across threads would test the harness, not the app's locking.
    The dependency override lives on the shared ``app``, so every
    per-thread client still routes to the same in-memory store.
    """
    import concurrent.futures as cf

    sid = _create_session(client)
    glyph_ids = [
        g["id"] for g in client.get(f"/sessions/{sid}").json()["glyphs"][:8]
    ]

    def label(gid: str):
        with TestClient(app) as worker:
            return worker.post(
                f"/sessions/{sid}/glyphs/{gid}",
                json={"class_name": "neume.A", "id_state_manual": True},
            )

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(label, glyph_ids))

    assert all(r.status_code == 200 for r in results), [r.text for r in results]

    final = client.get(f"/sessions/{sid}").json()["glyphs"]
    manual = {g["id"] for g in final if g["id_state_manual"]}
    assert manual == set(glyph_ids)


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------


@pytest.fixture
def train_dir(monkeypatch, tmp_path) -> Path:
    """A hermetic ``core/data/train`` dir with one vocab CSV and one non-vocab CSV."""
    (tmp_path / "vocab_a.csv").write_text(
        "name,classification,width,mei\n"
        "g1,neume.punctum,10,x\n"
        "g2,clef.c,12,y\n"
        "g3,neume.punctum,9,z\n"  # duplicate class — must be de-duped
        "g4,,5,w\n",  # blank class — must be dropped
        encoding="utf-8",
    )
    # A CSV without a 'classification' column must not be listed as a vocab.
    (tmp_path / "annotations.csv").write_text(
        "filename,region_attributes\nimg.png,{}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("IC_TRAIN_DIR", str(tmp_path))
    return tmp_path


def test_list_vocabularies_only_returns_csvs_with_classification_column(
    client, train_dir
):
    response = client.get("/vocabularies")
    assert response.status_code == 200
    assert response.json() == ["vocab_a.csv"]


def test_vocabulary_classes_are_sorted_distinct_and_non_empty(client, train_dir):
    response = client.get("/vocabularies/vocab_a.csv/classes")
    assert response.status_code == 200
    assert response.json() == ["clef.c", "neume.punctum"]


def test_unknown_vocabulary_is_rejected(client, train_dir):
    response = client.get("/vocabularies/../secrets.csv/classes")
    assert response.status_code in (400, 404)


def test_create_session_seeds_class_names_from_vocabulary(client, train_dir):
    files = {
        "page_image": ("page.png", PAGE_BYTES, "image/png"),
        "annotations": ("annotations.json", JSON_BYTES, "application/json"),
    }
    response = client.post(
        "/sessions",
        files=files,
        data={"annotations_format": "json", "vocabulary": "vocab_a.csv"},
    )
    assert response.status_code == 201, response.text
    names = response.json()["class_names"]
    assert "clef.c" in names
    assert "neume.punctum" in names


# ---------------------------------------------------------------------------
# Training-set presets
# ---------------------------------------------------------------------------

# Class labels baked into the hermetic preset / uploaded training fixtures.
PRESET_LABEL = "neume.punctum"
PRESET_GLYPH_COUNT = 5
UPLOAD_LABEL = "neume.virga"
UPLOAD_GLYPH_COUNT = 3


def _labelled_training_xml(label: str, count: int) -> bytes:
    """A real GameraXML training doc: the first ``count`` ingested test glyphs
    labelled ``label``. Using the ingest pipeline gives glyphs with genuine
    masks/features so a classify round over them actually runs."""
    from ic_core.ingest import ingest_page
    from ic_core.io_xml import dumps_glyphs

    glyphs = ingest_page(PAGE_BYTES, JSON_BYTES, format="json")[:count]
    return dumps_glyphs([g.classify_manual(label) for g in glyphs])


@pytest.fixture
def presets_dir(monkeypatch, tmp_path) -> Path:
    """A hermetic core/data/presets dir with one real labelled GameraXML preset."""
    (tmp_path / "SamplePreset.xml").write_bytes(
        _labelled_training_xml(PRESET_LABEL, PRESET_GLYPH_COUNT)
    )
    # A non-xml file must not be listed as a preset.
    (tmp_path / "notes.txt").write_text("not a preset", encoding="utf-8")
    monkeypatch.setenv("IC_PRESETS_DIR", str(tmp_path))
    return tmp_path


def test_list_training_presets_only_returns_xml_files(client, presets_dir):
    response = client.get("/training-presets")
    assert response.status_code == 200
    assert response.json() == ["SamplePreset.xml"]


def test_create_session_seeds_training_pool_from_preset(client, presets_dir):
    files = {
        "page_image": ("page.png", PAGE_BYTES, "image/png"),
        "annotations": ("annotations.json", JSON_BYTES, "application/json"),
    }
    response = client.post(
        "/sessions",
        files=files,
        data={
            "annotations_format": "json",
            "training_presets": json.dumps(["SamplePreset.xml"]),
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    # The preset glyphs seed the training pool ...
    assert len(body["training_glyphs"]) == PRESET_GLYPH_COUNT
    # ... and the auto-classify round labels the working set from them.
    assert any(g["class_name"] == PRESET_LABEL for g in body["glyphs"])


def test_create_session_concatenates_presets_with_uploaded_training(
    client, presets_dir
):
    files = {
        "page_image": ("page.png", PAGE_BYTES, "image/png"),
        "annotations": ("annotations.json", JSON_BYTES, "application/json"),
        "training_files": (
            "uploaded.xml",
            _labelled_training_xml(UPLOAD_LABEL, UPLOAD_GLYPH_COUNT),
            "application/xml",
        ),
    }
    response = client.post(
        "/sessions",
        files=files,
        data={
            "annotations_format": "json",
            "training_presets": json.dumps(["SamplePreset.xml"]),
        },
    )
    assert response.status_code == 201, response.text
    # Preset glyphs + uploaded glyphs are both concatenated into the pool.
    assert (
        len(response.json()["training_glyphs"])
        == PRESET_GLYPH_COUNT + UPLOAD_GLYPH_COUNT
    )


def test_unknown_training_preset_is_rejected(client, presets_dir):
    # A client-supplied name outside the enumerated listing (path traversal
    # or typo) must fail before any disk access.
    files = {
        "page_image": ("page.png", PAGE_BYTES, "image/png"),
        "annotations": ("annotations.json", JSON_BYTES, "application/json"),
    }
    response = client.post(
        "/sessions",
        files=files,
        data={
            "annotations_format": "json",
            "training_presets": json.dumps(["../secrets.xml"]),
        },
    )
    assert response.status_code == 400


def test_dto_reports_preset_and_uploaded_training_counts_separately(
    client, presets_dir
):
    files = {
        "page_image": ("page.png", PAGE_BYTES, "image/png"),
        "annotations": ("annotations.json", JSON_BYTES, "application/json"),
        "training_files": (
            "uploaded.xml",
            _labelled_training_xml(UPLOAD_LABEL, UPLOAD_GLYPH_COUNT),
            "application/xml",
        ),
    }
    sid = client.post(
        "/sessions",
        files=files,
        data={
            "annotations_format": "json",
            "training_presets": json.dumps(["SamplePreset.xml"]),
        },
    ).json()["id"]

    dto = client.get(f"/sessions/{sid}").json()
    assert dto["preset_training_count"] == PRESET_GLYPH_COUNT
    assert dto["uploaded_training_count"] == UPLOAD_GLYPH_COUNT


def test_export_selects_preset_and_uploaded_training_independently(
    client, presets_dir
):
    from ic_core.io_xml import load_glyphs_bytes

    files = {
        "page_image": ("page.png", PAGE_BYTES, "image/png"),
        "annotations": ("annotations.json", JSON_BYTES, "application/json"),
        "training_files": (
            "uploaded.xml",
            _labelled_training_xml(UPLOAD_LABEL, UPLOAD_GLYPH_COUNT),
            "application/xml",
        ),
    }
    sid = client.post(
        "/sessions",
        files=files,
        data={
            "annotations_format": "json",
            "training_presets": json.dumps(["SamplePreset.xml"]),
        },
    ).json()["id"]

    # Preset-only export carries just the preset glyphs, by their label.
    preset = load_glyphs_bytes(
        client.post(f"/sessions/{sid}/complete?preset_training=true").content
    )
    assert len(preset) == PRESET_GLYPH_COUNT
    assert {g.class_name for g in preset} == {PRESET_LABEL}

    # Uploaded-only export carries just the uploaded glyphs.
    uploaded = load_glyphs_bytes(
        client.post(
            f"/sessions/{sid}/complete?uploaded_training=true"
        ).content
    )
    assert len(uploaded) == UPLOAD_GLYPH_COUNT
    assert {g.class_name for g in uploaded} == {UPLOAD_LABEL}


def test_export_manual_neumes_only_includes_hand_labelled_neumes(client):
    from ic_core.io_xml import load_glyphs_bytes

    sid = _create_session(client)
    neume = next(
        g
        for g in client.get(f"/sessions/{sid}").json()["glyphs"]
        if g["category"] == "Neumes"
    )
    client.post(
        f"/sessions/{sid}/glyphs/{neume['id']}",
        json={"class_name": "neume.punctum", "id_state_manual": True},
    )

    exported = load_glyphs_bytes(
        client.post(f"/sessions/{sid}/complete?manual_neumes=true").content
    )
    assert len(exported) == 1
    assert exported[0].class_name == "neume.punctum"


# ---------------------------------------------------------------------------
# Re-binarization — POST /sessions/{id}/binarization
# ---------------------------------------------------------------------------


def test_create_session_reports_default_binarization_method(client):
    body = client.post("/sessions", **_multipart()).json()
    assert body["binarization_method"] == "global"


def test_rebinarize_switches_method_and_rebuilds_masks(client):
    sid = _create_session(client)
    before = client.get(f"/sessions/{sid}").json()
    masks_before = {g["id"]: g["image_b64"] for g in before["glyphs"]}

    r = client.post(f"/sessions/{sid}/binarization", json={"method": "sauvola"})
    assert r.status_code == 200, r.text
    after = r.json()
    assert after["binarization_method"] == "sauvola"
    masks_after = {g["id"]: g["image_b64"] for g in after["glyphs"]}
    # Same glyph set (ids preserved), different pixels.
    assert masks_before.keys() == masks_after.keys()
    assert any(masks_before[i] != masks_after[i] for i in masks_before)


def test_rebinarize_keeps_manual_labels(client):
    sid = _create_session(client)
    gid = client.get(f"/sessions/{sid}").json()["glyphs"][0]["id"]
    client.post(
        f"/sessions/{sid}/glyphs/{gid}",
        json={"class_name": "neume.A", "id_state_manual": True},
    )

    after = client.post(
        f"/sessions/{sid}/binarization", json={"method": "otsu"}
    ).json()
    moved = next(g for g in after["glyphs"] if g["id"] == gid)
    assert moved["class_name"] == "neume.A"
    assert moved["id_state_manual"] is True


def test_rebinarize_keeps_split_children_and_their_labels(client):
    # The reported inconsistency: manual *labels* survived a binarization
    # change, but a split did not — the children vanished and the parent the
    # user had taken apart came back, so the same action lost work or didn't
    # depending on which kind of work it was.
    sid = _create_session(client)
    parent = client.get(f"/sessions/{sid}").json()["glyphs"][0]
    ulx, uly, ncols, nrows = (
        parent["ulx"], parent["uly"], parent["ncols"], parent["nrows"]
    )
    half = max(1, ncols // 2)
    children = client.post(
        f"/sessions/{sid}/glyphs/{parent['id']}/split",
        json={
            "regions": [
                [ulx, uly, half, nrows],
                [ulx + half, uly, ncols - half, nrows],
            ]
        },
    ).json()
    assert len(children) == 2
    client.post(
        f"/sessions/{sid}/glyphs/{children[0]['id']}",
        json={"class_name": "neume.split-a", "id_state_manual": True},
    )

    after = client.post(
        f"/sessions/{sid}/binarization", json={"method": "sauvola"}
    ).json()

    ids = {g["id"] for g in after["glyphs"]}
    assert parent["id"] not in ids, "the split parent must not reappear"
    assert {c["id"] for c in children} <= ids, "both children must survive"
    kept = next(g for g in after["glyphs"] if g["id"] == children[0]["id"])
    assert kept["class_name"] == "neume.split-a"
    assert kept["id_state_manual"] is True
    # The child keeps its own footprint, not the parent's.
    assert kept["ncols"] == children[0]["ncols"]


def test_rebinarize_keeps_grouped_glyphs(client):
    sid = _create_session(client)
    glyphs = client.get(f"/sessions/{sid}").json()["glyphs"]
    pair = [glyphs[0]["id"], glyphs[1]["id"]]
    grouped = client.post(
        f"/sessions/{sid}/group",
        json={"glyph_ids": pair, "class_name": "neume.grouped"},
    ).json()

    after = client.post(
        f"/sessions/{sid}/binarization", json={"method": "otsu"}
    ).json()

    ids = {g["id"] for g in after["glyphs"]}
    assert grouped["id"] in ids
    assert not set(pair) & ids, "grouped members must not reappear"
    kept = next(g for g in after["glyphs"] if g["id"] == grouped["id"])
    assert kept["class_name"] == "neume.grouped"


def test_rebinarize_twice_is_stable(client):
    # Switching back and forth must not erode the working set — an early
    # version of the re-slice shaved columns off edge-clamped glyphs on
    # every switch.
    sid = _create_session(client)
    first = client.post(
        f"/sessions/{sid}/binarization", json={"method": "otsu"}
    ).json()
    client.post(f"/sessions/{sid}/binarization", json={"method": "sauvola"})
    third = client.post(
        f"/sessions/{sid}/binarization", json={"method": "otsu"}
    ).json()

    def boxes(dto):
        return [
            (g["id"], g["ulx"], g["uly"], g["ncols"], g["nrows"])
            for g in dto["glyphs"]
        ]

    assert boxes(third) == boxes(first)
    # Same method, same page: the masks must come back identical too.
    assert [g["image_b64"] for g in third["glyphs"]] == [
        g["image_b64"] for g in first["glyphs"]
    ]


def test_rebinarize_rejects_unknown_method(client):
    sid = _create_session(client)
    r = client.post(f"/sessions/{sid}/binarization", json={"method": "bogus"})
    assert r.status_code == 422


def test_rebinarize_without_retained_page_is_400(client, store):
    # A session created without a page+bbox upload (e.g. legacy XML import)
    # has nothing to re-binarise from.
    from ic_core.state import Session

    session = Session()
    session.ingest([])  # → CLASSIFYING, no page/annotation bytes
    store.create(session)

    r = client.post(f"/sessions/{session.id}/binarization", json={"method": "sauvola"})
    assert r.status_code == 400
