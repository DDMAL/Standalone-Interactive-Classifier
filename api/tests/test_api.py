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
) -> dict:
    """Build kwargs for ``TestClient.post`` that emulate a browser upload.

    httpx accepts the ``files`` and ``data`` dict pair to assemble a
    proper ``multipart/form-data`` body — this is what the frontend
    will send once it exists.
    """
    files = {
        "page_image": ("page.png", PAGE_BYTES, "image/png"),
        "annotations": ("annotations.json", JSON_BYTES, "application/json"),
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
