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
    # what a superseded session sees, because _insert deleted its row. Tests
    # that need a *successful* hydrate override this with a stub returning
    # _hydrate's real shape, ``(session, key)``.
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


def test_hydrated_session_is_evicted_when_superseded(store, monkeypatch):
    # A session that predates this process (a restart, or another replica)
    # reaches the cache through the hydrate path rather than create(), so it
    # has to be keyed there too. Otherwise create() can't see it, and it
    # survives in the cache — writable, flushing into a row _insert deleted —
    # until the next restart turns it into "Unknown session id".
    hydrated = _session()

    def _row(session_id):
        if session_id == hydrated.id:
            return hydrated, (7, "page-1")
        raise KeyError(f"Unknown session id: {session_id!r}")

    monkeypatch.setattr(store, "_hydrate", _row)

    assert store.get(hydrated.id) is hydrated
    assert store._keys[hydrated.id] == (7, "page-1")

    replacement = _session()
    store.create(replacement, project_id=7, image_id="page-1")

    # Asserted on the cache directly: a get() here would re-hydrate through
    # the stub above, where the real store would raise on the deleted row.
    assert hydrated.id not in store._cache
    assert hydrated.id not in store._keys
    assert store.get(replacement.id) is replacement


def test_hydrated_keyless_session_is_not_keyed(store, monkeypatch):
    # A session with no (project, image) belongs to no page, so it must not be
    # recorded — every key-less session would otherwise share one (None, None)
    # key and supersede the others.
    keyless = _session()

    monkeypatch.setattr(
        store,
        "_hydrate",
        lambda session_id: (keyless, None),
    )

    assert store.get(keyless.id) is keyless
    assert keyless.id not in store._keys

    # And a new page-keyed session must leave it alone.
    store.create(_session(), project_id=7, image_id="page-1")
    assert store._cache[keyless.id] is keyless


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
