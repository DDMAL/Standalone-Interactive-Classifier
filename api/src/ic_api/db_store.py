"""Postgres-backed session store — the durable swap for the in-memory one.

``store.py`` describes the seam: *"When the project needs persistence
across restarts the only thing that has to change is this module — the
public protocol is intentionally narrow."* This is that implementation.

Design
------

* **In-memory hot cache + write-through.** A retrieved :class:`Session`
  is a plain mutable object; every mutating endpoint already wraps its
  read-mutate sequence in ``with store.session(id) as s:``. We hook the
  *exit* of that context manager to flush the session to Postgres — so
  every mutation auto-persists with **zero endpoint changes**. Reads go
  through the same path, so a pure ``GET`` also re-flushes; that extra
  write is a small, idempotent ``UPDATE`` and acceptable for the
  single-user tool this serves.
* **Immutable blobs written once.** The page image and annotation bytes
  never change after ingest, so they are inserted at :meth:`create` and
  never rewritten. Only the mutable columns (glyphs, class-name sets,
  binarisation method + its derived page mask, and the lifecycle state)
  are rewritten on each flush.
* **Keyed by (project_id, image_id).** The embedding host (mothra) owns
  projects and pages; it stamps each session with the project + image it
  belongs to so it can *resume* the right session when the user returns
  to a page (see :meth:`lookup`). Sessions created without a key
  (IC's own upload screen) simply have ``NULL`` for both.

Glyphs serialise losslessly via their RLE image string (not the base64
PNG preview, which is display-only); the classifier's ``feature_vector``
cache is dropped and lazily recomputed after a hydrate.
"""
from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2 import pool as _pg_pool
from psycopg2.extras import Json

from ic_api.store import SessionSummary
from ic_core.glyph import Glyph
from ic_core.image import array_to_rle, rle_to_array
from ic_core.state import ClassifierState, Session

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ic_sessions (
    id                    TEXT PRIMARY KEY,
    project_id            INTEGER,
    image_id              TEXT,
    state                 TEXT NOT NULL,
    source_name           TEXT DEFAULT '',
    binarization_method   TEXT DEFAULT 'global',
    page_media_type       TEXT,
    annotations_format    TEXT,
    page_bytes            BYTEA,
    annotations_bytes     BYTEA,
    page_mask_rle         TEXT,
    page_mask_ncols       INTEGER,
    page_mask_nrows       INTEGER,
    glyphs                JSONB NOT NULL DEFAULT '[]'::jsonb,
    training_glyphs       JSONB NOT NULL DEFAULT '[]'::jsonb,
    imported_class_names  JSONB NOT NULL DEFAULT '[]'::jsonb,
    preset_training_ids   JSONB NOT NULL DEFAULT '[]'::jsonb,
    uploaded_training_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at            TIMESTAMPTZ DEFAULT NOW()
);
"""

# One live session per (project, image) so re-entering a page resumes the
# same session. Partial so the many key-less (NULL, NULL) sessions from
# IC's own upload screen never collide with each other.
_CREATE_KEY_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_ic_sessions_project_image
    ON ic_sessions (project_id, image_id)
    WHERE project_id IS NOT NULL AND image_id IS NOT NULL;
"""

# Column order shared by _SELECT and _row_to_session.
_COLUMNS = (
    "id, project_id, image_id, state, source_name, binarization_method, "
    "page_media_type, annotations_format, page_bytes, annotations_bytes, "
    "page_mask_rle, page_mask_ncols, page_mask_nrows, "
    "glyphs, training_glyphs, imported_class_names, "
    "preset_training_ids, uploaded_training_ids"
)


# ---------------------------------------------------------------------------
# Glyph (de)serialisation
# ---------------------------------------------------------------------------


def _glyph_to_json(g: Glyph) -> dict:
    """Losslessly serialise a glyph. The RLE image is authoritative; the
    display-only base64 PNG and the recomputable feature cache are dropped."""
    return {
        "id": g.id,
        "class_name": g.class_name,
        "image_rle": g.image_rle,
        "ncols": g.ncols,
        "nrows": g.nrows,
        "ulx": g.ulx,
        "uly": g.uly,
        "id_state_manual": g.id_state_manual,
        "confidence": g.confidence,
        "category": g.category,
        "is_training": g.is_training,
    }


def _glyph_from_json(d: dict) -> Glyph:
    """Inverse of :func:`_glyph_to_json`. Keys line up with ``Glyph.new``."""
    return Glyph.new(**d)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PersistentSessionStore:
    """Postgres-backed store satisfying the ``SessionStore`` protocol.

    Connection handling mirrors the landing-page backend: a lazily-created
    :class:`psycopg2.pool.ThreadedConnectionPool` so uvicorn's threaded
    request handling doesn't share a raw connection across threads.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: _pg_pool.ThreadedConnectionPool | None = None
        self._pool_lock = threading.Lock()
        self._schema_ready = False
        # Hot cache + locks, same shape as InMemorySessionStore.
        self._registry_lock = threading.Lock()
        self._cache: dict[str, Session] = {}
        self._locks: dict[str, threading.Lock] = {}
        # session id -> (project_id, image_id) for every cached session that
        # belongs to a page, so :meth:`create` can evict the one a new session
        # for the same page supersedes. Written by both paths that populate
        # ``_cache`` — :meth:`create` and the hydrate branch of :meth:`get` —
        # so a session created before this process started is covered too.
        # Key-less sessions (IC's own upload screen) are absent by design:
        # they belong to no page and supersede nothing.
        self._keys: dict[str, tuple[int, str]] = {}

    # -- connection pool ---------------------------------------------------

    def _get_pool(self) -> _pg_pool.ThreadedConnectionPool:
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    self._pool = _pg_pool.ThreadedConnectionPool(
                        minconn=1, maxconn=8, dsn=self._dsn
                    )
        return self._pool

    @contextmanager
    def _conn(self) -> Iterator[tuple]:
        pool = self._get_pool()
        con = pool.getconn()
        try:
            self._ensure_schema(con)
            cur = con.cursor()
            try:
                yield con, cur
                con.commit()  # end transaction even for read-only operations
            except Exception:
                con.rollback()
                raise
            finally:
                cur.close()
        finally:
            pool.putconn(con)

    def _ensure_schema(self, con) -> None:
        if self._schema_ready:
            return
        with self._pool_lock:
            if self._schema_ready:
                return
            cur = con.cursor()
            try:
                cur.execute(_CREATE_TABLE)
                cur.execute(_CREATE_KEY_INDEX)
                con.commit()
            finally:
                cur.close()
            self._schema_ready = True

    # -- protocol ----------------------------------------------------------

    def create(
        self,
        session: Session,
        *,
        project_id: int | None = None,
        image_id: str | None = None,
    ) -> None:
        """Insert ``session`` (cache + DB). Raises :class:`KeyError` on id
        collision. A pre-existing session for the same ``(project_id,
        image_id)`` is superseded — starting a new session for a page
        replaces the persisted one for that page."""
        with self._registry_lock:
            if session.id in self._cache:
                raise KeyError(f"Session id collision: {session.id!r}")
            self._cache[session.id] = session
            self._locks[session.id] = threading.Lock()
            # Starting a new session for a page supersedes the old one:
            # :meth:`_insert` deletes its row, so it must leave the hot cache
            # too. Otherwise the superseded session stays reachable and
            # writable in this process — its flushes updating zero rows — and
            # only turns into "Unknown session id" at the next restart, long
            # after the work that was silently going nowhere.
            if project_id is not None and image_id is not None:
                superseded = [
                    sid
                    for sid, key in self._keys.items()
                    if key == (project_id, image_id) and sid != session.id
                ]
                for sid in superseded:
                    self._cache.pop(sid, None)
                    self._locks.pop(sid, None)
                    self._keys.pop(sid, None)
                self._keys[session.id] = (project_id, image_id)
        self._insert(session, project_id, image_id)

    def get(self, session_id: str) -> Session:
        """Return the session, hydrating from the DB on a cache miss.

        Raises :class:`KeyError` if unknown to both cache and DB.
        """
        with self._registry_lock:
            sess = self._cache.get(session_id)
        if sess is not None:
            return sess
        # raises KeyError if absent
        hydrated, key = self._hydrate(session_id)
        with self._registry_lock:
            existing = self._cache.get(session_id)
            if existing is not None:
                return existing  # another thread won the race (and registered)
            self._cache[session_id] = hydrated
            self._locks.setdefault(session_id, threading.Lock())
            # Record the page this session belongs to in the same critical
            # section that caches it, so create() can never see a cached
            # session whose key it doesn't know about.
            if key is not None:
                self._keys[session_id] = key
        return hydrated

    @contextmanager
    def session(self, session_id: str) -> Iterator[Session]:
        """Yield the session under its per-session lock, then flush.

        This is where auto-save happens: every mutating endpoint already
        wraps its work in ``with store.session(id) as s:``, so flushing on
        exit persists the mutation. If the body raises, the flush is
        skipped (a failed op leaves the DB row untouched).
        """
        self.get(session_id)  # hydrate + register lock if needed
        with self._registry_lock:
            sess = self._cache[session_id]
            lock = self._locks[session_id]
        with lock:
            yield sess
            self._flush(sess)

    def delete(self, session_id: str) -> None:
        """Drop the session from cache and DB. Raises :class:`KeyError` if
        it existed in neither."""
        with self._registry_lock:
            in_cache = self._cache.pop(session_id, None) is not None
            self._locks.pop(session_id, None)
            self._keys.pop(session_id, None)
        with self._conn() as (con, cur):
            cur.execute("DELETE FROM ic_sessions WHERE id=%s", (session_id,))
            deleted = cur.rowcount
            con.commit()
        if not in_cache and not deleted:
            raise KeyError(f"Unknown session id: {session_id!r}")

    def clear(self) -> int:
        """Drop every session from cache and DB; return the rows removed.

        The row count comes from the DB ``DELETE`` (authoritative — every
        created session is write-through inserted), so it stays accurate even
        if the hot cache holds only a subset of persisted sessions.
        """
        with self._registry_lock:
            self._cache.clear()
            self._locks.clear()
            self._keys.clear()
        with self._conn() as (con, cur):
            cur.execute("DELETE FROM ic_sessions")
            deleted = cur.rowcount
            con.commit()
        return deleted

    def lookup(self, project_id: int | None, image_id: str | None) -> str | None:
        """Return the resumable session id for a page, or ``None``.

        A session that has already been completed (``EXPORT`` — terminal
        and read-only) is treated as *not resumable*, so re-entering a
        finished page starts fresh rather than reopening a locked session.
        """
        if project_id is None or image_id is None:
            return None
        with self._conn() as (con, cur):
            cur.execute(
                "SELECT id, state FROM ic_sessions "
                "WHERE project_id=%s AND image_id=%s",
                (project_id, image_id),
            )
            row = cur.fetchone()
        if row is None or row[1] == ClassifierState.EXPORT.value:
            return None
        return row[0]

    def list_sessions(self) -> list[SessionSummary]:
        """Summaries of every persisted session, most-recently-updated first.

        Reads only the cheap metadata columns — the glyph count comes from
        ``jsonb_array_length`` server-side, so no glyph blobs cross the wire.
        Completed (``EXPORT``) sessions are included; the frontend surfaces
        the ``state`` so the user can tell finished pages apart.
        """
        with self._conn() as (con, cur):
            cur.execute(
                "SELECT id, state, source_name, "
                "jsonb_array_length(glyphs) AS n_glyphs, "
                "updated_at, project_id, image_id "
                "FROM ic_sessions ORDER BY updated_at DESC NULLS LAST"
            )
            rows = cur.fetchall()
        return [
            SessionSummary(
                id=r[0],
                state=r[1],
                source_name=r[2] or "",
                n_glyphs=r[3] or 0,
                updated_at=r[4].isoformat() if r[4] is not None else None,
                project_id=r[5],
                image_id=r[6],
            )
            for r in rows
        ]

    def __contains__(self, session_id: str) -> bool:
        with self._registry_lock:
            if session_id in self._cache:
                return True
        with self._conn() as (con, cur):
            cur.execute("SELECT 1 FROM ic_sessions WHERE id=%s", (session_id,))
            return cur.fetchone() is not None

    def __iter__(self) -> Iterator[str]:
        with self._conn() as (con, cur):
            cur.execute("SELECT id FROM ic_sessions")
            return iter([r[0] for r in cur.fetchall()])

    # -- persistence -------------------------------------------------------

    def _insert(
        self, session: Session, project_id: int | None, image_id: str | None
    ) -> None:
        mask_rle, mask_ncols, mask_nrows = _encode_mask(session)
        with self._conn() as (con, cur):
            if project_id is not None and image_id is not None:
                cur.execute(
                    "DELETE FROM ic_sessions WHERE project_id=%s AND image_id=%s",
                    (project_id, image_id),
                )
            cur.execute(
                f"INSERT INTO ic_sessions ({_COLUMNS}) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    session.id,
                    project_id,
                    image_id,
                    session.state.value,
                    session.source_name,
                    session.binarization_method,
                    session.page_media_type,
                    session.annotations_format,
                    psycopg2.Binary(session.page_bytes)
                    if session.page_bytes
                    else None,
                    psycopg2.Binary(session.annotations_bytes)
                    if session.annotations_bytes
                    else None,
                    mask_rle,
                    mask_ncols,
                    mask_nrows,
                    Json([_glyph_to_json(g) for g in session.glyphs]),
                    Json([_glyph_to_json(g) for g in session.training_glyphs]),
                    Json(sorted(session.imported_class_names)),
                    Json(sorted(session.preset_training_ids)),
                    Json(sorted(session.uploaded_training_ids)),
                ),
            )
            con.commit()

    def _flush(self, session: Session) -> None:
        """Rewrite the mutable columns of an existing row. The page/annotation
        blobs are immutable after :meth:`create` and never touched here; the
        page mask *is* included because rebinarisation rebuilds it."""
        mask_rle, mask_ncols, mask_nrows = _encode_mask(session)
        with self._conn() as (con, cur):
            cur.execute(
                "UPDATE ic_sessions SET "
                "state=%s, binarization_method=%s, "
                "page_mask_rle=%s, page_mask_ncols=%s, page_mask_nrows=%s, "
                "glyphs=%s, training_glyphs=%s, imported_class_names=%s, "
                "preset_training_ids=%s, uploaded_training_ids=%s, "
                "updated_at=NOW() WHERE id=%s",
                (
                    session.state.value,
                    session.binarization_method,
                    mask_rle,
                    mask_ncols,
                    mask_nrows,
                    Json([_glyph_to_json(g) for g in session.glyphs]),
                    Json([_glyph_to_json(g) for g in session.training_glyphs]),
                    Json(sorted(session.imported_class_names)),
                    Json(sorted(session.preset_training_ids)),
                    Json(sorted(session.uploaded_training_ids)),
                    session.id,
                ),
            )
            if cur.rowcount == 0:
                # The row is gone — most likely a newer session for the same
                # (project_id, image_id) superseded this one (see _insert).
                # Every mutation this session accumulates from here is being
                # discarded, so say so rather than returning as though the
                # write landed. The request itself still succeeds: failing it
                # would not bring the row back.
                print(
                    f"[ic_api.db_store] flush for session {session.id!r} "
                    "updated 0 rows — that row no longer exists, so this "
                    "session is no longer being persisted. It was most "
                    "likely superseded by a newer session for the same page.",
                    file=sys.stderr,
                )
            con.commit()

    def _hydrate(
        self, session_id: str
    ) -> tuple[Session, tuple[int, str] | None]:
        """Load a session from its row, with the page it belongs to.

        Returns the session and its ``(project_id, image_id)`` — or ``None``
        for a key-less session (IC's own upload screen). The caller records
        the key alongside the cache entry so :meth:`create` can evict this
        session when a newer one supersedes the same page; a hydrated session
        is otherwise invisible to that check, since it was created before this
        process started (or by another one).
        """
        with self._conn() as (con, cur):
            cur.execute(
                f"SELECT {_COLUMNS} FROM ic_sessions WHERE id=%s", (session_id,)
            )
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"Unknown session id: {session_id!r}")
        # _COLUMNS order: id, project_id, image_id, ...
        project_id, image_id = row[1], row[2]
        key = (
            (project_id, image_id)
            if project_id is not None and image_id is not None
            else None
        )
        return _row_to_session(row), key


# ---------------------------------------------------------------------------
# Row <-> Session
# ---------------------------------------------------------------------------


def _encode_mask(session: Session) -> tuple[str | None, int | None, int | None]:
    """Serialise the page mask as ``(rle, ncols, nrows)`` or all-``None``."""
    mask = session.page_mask
    if mask is None:
        return None, None, None
    nrows, ncols = int(mask.shape[0]), int(mask.shape[1])
    return array_to_rle(mask), ncols, nrows


def _row_to_session(row: tuple) -> Session:
    (
        id_,
        _project_id,
        _image_id,
        state,
        source_name,
        binarization_method,
        page_media_type,
        annotations_format,
        page_bytes,
        annotations_bytes,
        page_mask_rle,
        page_mask_ncols,
        page_mask_nrows,
        glyphs,
        training_glyphs,
        imported_class_names,
        preset_training_ids,
        uploaded_training_ids,
    ) = row

    session = Session()
    session.id = id_
    session.state = ClassifierState(state)
    session.glyphs = [_glyph_from_json(d) for d in (glyphs or [])]
    session.training_glyphs = [_glyph_from_json(d) for d in (training_glyphs or [])]
    session.imported_class_names = set(imported_class_names or [])
    session.preset_training_ids = set(preset_training_ids or [])
    session.uploaded_training_ids = set(uploaded_training_ids or [])
    session.source_name = source_name or ""
    session.page_bytes = bytes(page_bytes) if page_bytes is not None else None
    session.page_media_type = page_media_type
    session.annotations_bytes = (
        bytes(annotations_bytes) if annotations_bytes is not None else None
    )
    session.annotations_format = annotations_format
    session.binarization_method = binarization_method or "global"
    if page_mask_rle is not None and page_mask_ncols and page_mask_nrows:
        session.page_mask = rle_to_array(
            page_mask_rle, page_mask_ncols, page_mask_nrows
        )
    else:
        session.page_mask = None
    return session
