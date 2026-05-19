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

import threading
from contextlib import contextmanager
from typing import Iterator, Protocol

from ic_core.state import Session


class SessionStore(Protocol):
    """Protocol that any persistence backend must satisfy.

    Keeping this as a structural protocol (rather than an ABC) means
    the API code can depend on a tiny surface and tests can use a
    plain dict if they want.
    """

    def create(self, session: Session) -> None: ...
    def get(self, session_id: str) -> Session: ...
    def session(self, session_id: str): ...  # context manager
    def delete(self, session_id: str) -> None: ...
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

    def create(self, session: Session) -> None:
        """Insert ``session``. Raises :class:`KeyError` on id collision."""
        with self._lock:
            if session.id in self._sessions:
                raise KeyError(f"Session id collision: {session.id!r}")
            self._sessions[session.id] = session
            self._locks[session.id] = threading.Lock()

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


#: Default app-wide store. The FastAPI app reaches this through a
#: dependency override-friendly factory in :mod:`ic_api.main`, so
#: tests can substitute their own.
default_store = InMemorySessionStore()
