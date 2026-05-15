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
    def delete(self, session_id: str) -> None: ...
    def __contains__(self, session_id: str) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...


class InMemorySessionStore:
    """Process-local session registry guarded by a single ``Lock``.

    The lock keeps the dict's ``__setitem__`` / ``__delitem__`` calls
    safe under uvicorn's threaded request handling. Operations on a
    *retrieved* :class:`Session` are not guarded — the FastAPI
    endpoint code calls into that mutable object directly. That's
    OK for single-user use; for multi-user it would need a per-session
    lock layered on top.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    def create(self, session: Session) -> None:
        """Insert ``session``. Raises :class:`KeyError` on id collision."""
        with self._lock:
            if session.id in self._sessions:
                raise KeyError(f"Session id collision: {session.id!r}")
            self._sessions[session.id] = session

    def get(self, session_id: str) -> Session:
        """Return the session or raise :class:`KeyError`."""
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError:
                raise KeyError(f"Unknown session id: {session_id!r}") from None

    def delete(self, session_id: str) -> None:
        """Drop the session; raise :class:`KeyError` if it doesn't exist."""
        with self._lock:
            try:
                del self._sessions[session_id]
            except KeyError:
                raise KeyError(f"Unknown session id: {session_id!r}") from None

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
