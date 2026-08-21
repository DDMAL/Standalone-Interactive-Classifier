"""Unit tests for the Postgres store's in-process bookkeeping.

The store's SQL needs a live Postgres, which CI doesn't have, so these
tests cover the part that is pure in-memory bookkeeping — the hot cache —
by stubbing the two methods that touch the database. That is where the
bug these tests pin actually lived: ``_insert`` deletes the row of a
session superseded by a new one for the same ``(project_id, image_id)``,
but the superseded session used to stay in the hot cache, reachable and
writable, with every flush silently updating zero rows. The loss only
surfaced at the next restart, as "Unknown session id", long after the work
had stopped being saved.
"""
from __future__ import annotations

import pytest

from ic_api.db_store import PersistentSessionStore
from ic_core.state import Session


@pytest.fixture()
def store(monkeypatch):
    """A store whose DB writes are stubbed out; the cache is real."""
    s = PersistentSessionStore("postgresql://unused")
    monkeypatch.setattr(s, "_insert", lambda *a, **k: None)
    monkeypatch.setattr(s, "_flush", lambda *a, **k: None)
    # A cache miss must behave like "the row isn't there" — which is exactly
    # what a superseded session sees, because _insert deleted its row.
    def _no_row(session_id):
        raise KeyError(f"Unknown session id: {session_id!r}")

    monkeypatch.setattr(s, "_hydrate", _no_row)
    return s


def _session() -> Session:
    s = Session()
    s.ingest([])
    return s


def test_new_session_for_a_page_evicts_the_superseded_one(store):
    old, new = _session(), _session()
    store.create(old, project_id=7, image_id="page-1")
    store.create(new, project_id=7, image_id="page-1")

    # The new one is live...
    assert store.get(new.id) is new
    # ...and the superseded one is gone from the cache, matching the DB,
    # where _insert deleted its row.
    with pytest.raises(KeyError):
        store.get(old.id)


def test_supersede_only_evicts_the_same_page(store):
    a, b = _session(), _session()
    store.create(a, project_id=7, image_id="page-1")
    store.create(b, project_id=7, image_id="page-2")

    assert store.get(a.id) is a
    assert store.get(b.id) is b


def test_keyless_sessions_never_supersede_each_other(store):
    # IC's own upload screen creates sessions with no (project, image) key;
    # those must not evict one another.
    a, b = _session(), _session()
    store.create(a)
    store.create(b)

    assert store.get(a.id) is a
    assert store.get(b.id) is b


def test_delete_forgets_the_page_key(store, monkeypatch):
    # After a delete, the page is free again: creating a new session for it
    # must be a clean insert, not a supersede of a stale cache entry.
    import contextlib

    class _Cur:
        rowcount = 1

        def execute(self, *a):
            pass

        def close(self):
            pass

    @contextlib.contextmanager
    def _fake_conn():
        class _Con:
            def commit(self):
                pass

        yield _Con(), _Cur()

    monkeypatch.setattr(store, "_conn", _fake_conn)

    a = _session()
    store.create(a, project_id=7, image_id="page-1")
    store.delete(a.id)
    assert a.id not in store._keys

    b = _session()
    store.create(b, project_id=7, image_id="page-1")
    assert store.get(b.id) is b
