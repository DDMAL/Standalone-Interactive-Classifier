"""In-memory session store.

A thin wrapper around a process-local ``dict[str, Session]`` with a
threading lock so concurrent FastAPI workers don't trip over each
other. This is **the** Phase-2 storage layer per the migration plan
recommendation for a single-user tool. When the project needs
persistence across restarts the only thing that has to change is
this module — the public protocol is intentionally narrow.

Why not SQLite from day one?
----------------------------
``docs/migration_plan.md`` §"State persistence" allows either; for
a single-user local tool, the round-trip cost of touching disk on
every endpoint is pure overhead. The session payload (cropped neume
masks, RLE-encoded) is also small enough to live in RAM
comfortably — a 1000-glyph session is on the order of a few MB.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Protocol

from ic_core.state import ClassifierState, Session


@dataclass(frozen=True)
class SessionSummary:
    """Lightweight metadata for one stored session.

    Enough to render a "resume a saved session" list without hydrating the
    whole session (glyph masks, page bytes) — returned by
    :meth:`SessionStore.list_sessions`. ``updated_at`` is an ISO-8601 string
    from the persistent store, or ``None`` for the in-memory store, which
    keeps no modification timestamp.
    """

    id: str
    state: str  # ClassifierState value
    source_name: str
    n_glyphs: int
    updated_at: str | None = None
    project_id: int | None = None
    image_id: str | None = None


class SessionStore(Protocol):
    """Protocol that any persistence backend must satisfy.

    Keeping this as a structural protocol (rather than an ABC) means
    the API code can depend on a tiny surface and tests can use a
    plain dict if they want.

    ``create`` optionally takes the ``(project_id, image_id)`` an
    embedding host (mothra) owns; a persistent backend records them so
    it can :meth:`lookup` and *resume* the session when the user returns
    to that page. Backends that don't persist simply ignore them and
    return ``None`` from ``lookup``.
    """

    def create(
        self,
        session: Session,
        *,
        project_id: int | None = None,
        image_id: str | None = None,
    ) -> None: ...
    def get(self, session_id: str) -> Session: ...
    def session(self, session_id: str): ...  # context manager
    def delete(self, session_id: str) -> None: ...
    def clear(self) -> int: ...
    def lookup(
        self, project_id: int | None, image_id: str | None
    ) -> str | None: ...
    def list_sessions(self) -> list[SessionSummary]: ...
    def __contains__(self, session_id: str) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...


class InMemorySessionStore:
    """Process-local session registry with a registry lock + per-session locks.

    The registry lock guards the ``_sessions`` and ``_locks`` dicts so
    lookups/inserts/deletes are atomic under uvicorn's threaded request
    handling. Each session also has its own :class:`threading.Lock`, lazily
    created on first access; callers use :meth:`session` (a context
    manager) to acquire that lock around any read-then-mutate sequence.
    Different session ids never block each other; concurrent requests
    on the *same* id are serialized.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, threading.Lock] = {}
        # (project_id, image_id) -> session id, so lookup/resume works within
        # a single process run even without a persistent backend.
        self._by_key: dict[tuple[int, str], str] = {}

    def create(
        self,
        session: Session,
        *,
        project_id: int | None = None,
        image_id: str | None = None,
    ) -> None:
        """Insert ``session``. Raises :class:`KeyError` on id collision."""
        with self._lock:
            if session.id in self._sessions:
                raise KeyError(f"Session id collision: {session.id!r}")
            self._sessions[session.id] = session
            self._locks[session.id] = threading.Lock()
            if project_id is not None and image_id is not None:
                key = (project_id, image_id)
                old_id = self._by_key.get(key)
                if old_id is not None:
                    self._sessions.pop(old_id, None)
                    self._locks.pop(old_id, None)
                self._by_key[key] = session.id

    def get(self, session_id: str) -> Session:
        """Return the session or raise :class:`KeyError`.

        Read-only callers can use this directly, but anything that
        mutates the session (or serialises it into a DTO while
        mutations are possible elsewhere) should go through
        :meth:`session` to hold the per-session lock.
        """
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError:
                raise KeyError(f"Unknown session id: {session_id!r}") from None

    @contextmanager
    def session(self, session_id: str) -> Iterator[Session]:
        """Yield the session while holding its per-session lock.

        Use this around every read-then-mutate sequence — and around
        reads whose result is serialised into a DTO — so the operation
        is atomic with respect to other requests on the same id. The
        registry lock is released before the per-session lock is
        acquired, so different ids proceed in parallel.

        Raises :class:`KeyError` if the id is unknown.
        """
        with self._lock:
            try:
                sess = self._sessions[session_id]
                lock = self._locks[session_id]
            except KeyError:
                raise KeyError(f"Unknown session id: {session_id!r}") from None
        with lock:
            yield sess

    def delete(self, session_id: str) -> None:
        """Drop the session; raise :class:`KeyError` if it doesn't exist.

        If another thread is mid-operation under the per-session lock
        when delete races in, that operation will complete on a session
        no longer reachable from the store — harmless, since the
        orphan can't affect any other request.
        """
        with self._lock:
            try:
                del self._sessions[session_id]
            except KeyError:
                raise KeyError(f"Unknown session id: {session_id!r}") from None
            self._locks.pop(session_id, None)
            for key, sid in list(self._by_key.items()):
                if sid == session_id:
                    del self._by_key[key]

    def clear(self) -> int:
        """Drop every session; return how many were removed.

        In-flight operations holding a per-session lock complete on an
        orphaned session (same reasoning as :meth:`delete`) — harmless.
        """
        with self._lock:
            n = len(self._sessions)
            self._sessions.clear()
            self._locks.clear()
            self._by_key.clear()
        return n

    def lookup(
        self, project_id: int | None, image_id: str | None
    ) -> str | None:
        """Return the resumable session id for ``(project_id, image_id)``, if
        one exists in this process. In-memory only — lost on restart. A
        completed (``EXPORT``, terminal) session is treated as not resumable,
        matching the persistent store."""
        if project_id is None or image_id is None:
            return None
        with self._lock:
            sid = self._by_key.get((project_id, image_id))
            if sid is None:
                return None
            session = self._sessions.get(sid)
            if session is None or session.state == ClassifierState.EXPORT:
                return None
            return sid

    def list_sessions(self) -> list[SessionSummary]:
        """Summaries of every session in this process, most-recent first.

        In-memory only — no persistence, so this reflects sessions created
        since the last restart and carries no ``updated_at`` timestamp.
        Insertion order approximates recency, so it's reversed.
        """
        with self._lock:
            key_by_id = {sid: key for key, sid in self._by_key.items()}
            summaries = [
                SessionSummary(
                    id=s.id,
                    state=s.state.value,
                    source_name=s.source_name,
                    n_glyphs=len(s.glyphs),
                    updated_at=None,
                    project_id=key_by_id.get(s.id, (None, None))[0],
                    image_id=key_by_id.get(s.id, (None, None))[1],
                )
                for s in self._sessions.values()
            ]
        summaries.reverse()
        return summaries

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def __iter__(self) -> Iterator[str]:
        # Snapshot the ids under the lock; iterating outside the
        # lock is fine because the snapshot is immutable.
        with self._lock:
            return iter(list(self._sessions))

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


def build_default_store() -> SessionStore:
    """Pick the store backend from the environment.

    When ``IC_DATABASE_URL`` (or, failing that, ``DATABASE_URL`` — shared
    with the mothra backend) is set, sessions are persisted to Postgres so
    they survive a restart and can be resumed per project + page. Otherwise
    the process-local in-memory store is used, preserving the original
    single-user / local-tool behaviour with no external dependency.

    The Postgres store connects lazily (first DB access), so constructing
    it here is cheap and never blocks import on the network. If importing
    the backend fails (e.g. psycopg2 missing), we fall back to in-memory
    and warn rather than refusing to boot.
    """
    dsn = os.environ.get("IC_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if dsn:
        try:
            from ic_api.db_store import PersistentSessionStore

            return PersistentSessionStore(dsn)
        except Exception as exc:  # pragma: no cover - defensive
            import sys

            print(
                "[ic_api.store] DATABASE_URL is set but the Postgres store "
                f"could not be initialised ({exc}); falling back to the "
                "in-memory store — sessions will NOT persist across restarts.",
                file=sys.stderr,
            )
    return InMemorySessionStore()


#: Default app-wide store. The FastAPI app reaches this through a
#: dependency override-friendly factory in :mod:`ic_api.main`, so
#: tests can substitute their own.
default_store: SessionStore = build_default_store()
