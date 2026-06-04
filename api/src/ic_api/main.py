"""FastAPI application — HTTP surface for the Interactive Classifier.

This is the Phase-2 layer per ``../docs/migration_plan.md``: a
**thin** translation from HTTP into :mod:`ic_core.state.Session`
operations and back. No algorithm logic lives here; the endpoints
exist only to map JSON requests onto session methods and serialise
the result.

Lifecycle and state mapping
---------------------------

The mapping is direct: every endpoint operates on a single
:class:`ic_core.state.Session` resolved by id from the
:class:`ic_api.store.InMemorySessionStore`. State transitions are
enforced inside ``Session`` and surfaced as HTTP 409 here.

Error model
-----------

Every non-2xx response uses :class:`ic_api.schemas.ErrorResponse`:
``{"detail": "...", "code": "..."}``. The ``code`` values are a
finite enum so the frontend can dispatch on them without parsing
free-form ``detail`` strings.

What's deliberately missing in v1
---------------------------------

* **Auth.** Single-user / local-tool target — see migration plan
  §"Auth".
* **WebSocket progress events.** The numpy classifier is fast
  enough on the dataset sizes we care about that synchronous JSON
  responses are fine. Add a streaming endpoint when an operation
  starts feeling slow.
* **Auto-grouping endpoint.** Deferred at the algorithm layer —
  this endpoint returns HTTP 501.
* **Persistent storage.** The default store is in-memory only;
  swap :mod:`ic_api.store` for a SQLite-backed implementation when
  sessions need to outlive a process restart.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ic_api.schemas import (
    ClassifyRequest,
    ErrorResponse,
    GlyphDTO,
    GroupRequest,
    RenameClassRequest,
    SessionDTO,
    UpdateGlyphRequest,
    glyph_to_dto,
    session_to_dto,
)
from ic_api.store import InMemorySessionStore, default_store
from ic_core.ingest import AnnotationFormat, ingest_page
from ic_core.io_xml import dumps_glyphs, load_glyphs
from ic_core.state import Session, StateTransitionError


# ---------------------------------------------------------------------------
# App & dependency wiring
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Interactive Classifier API",
    version="0.1.0",
    description=(
        "Phase-2 HTTP layer for the Interactive Classifier rewrite. "
        "Wraps ic_core.state.Session with REST endpoints."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_store() -> InMemorySessionStore:
    """Dependency-injection point for the session store.

    The default returns the module-level :data:`ic_api.store.default_store`;
    tests override this with a fresh store via ``app.dependency_overrides``.
    """
    return default_store


Store = Annotated[InMemorySessionStore, Depends(get_store)]


# ---------------------------------------------------------------------------
# Built-in training sets
# ---------------------------------------------------------------------------
#
# Pre-built GameraXML training databases live under ``core/data/derived``
# (e.g. ``Hufnagel_training_data.xml``). The frontend offers them as a
# dropdown on the upload screen; picking one seeds the session's training
# pool so the first classify round applies that vocabulary directly.
#
# The directory is resolved relative to the repo root and may be
# overridden via ``IC_DERIVED_DIR`` to stay consistent with
# ``core/scripts/paths.py``. The API only ever opens files it has itself
# enumerated in this directory — a client-supplied name is validated
# against that listing before any disk access, so path traversal
# (``../secrets.xml``) cannot escape the directory.


def derived_dir() -> Path:
    """Directory holding the pre-built training-set XML databases."""
    override = os.environ.get("IC_DERIVED_DIR")
    if override:
        return Path(override)
    # main.py → ic_api → src → api → <repo root>
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "core" / "data" / "derived"


def list_training_sets() -> list[str]:
    """Return the sorted filenames of every ``*.xml`` in :func:`derived_dir`."""
    root = derived_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.glob("*.xml") if p.is_file())


def resolve_training_set(name: str) -> Path:
    """Map a client-supplied training-set filename to a safe on-disk path.

    Raises:
        ValueError: If ``name`` is not one of the files enumerated by
            :func:`list_training_sets` (guards against path traversal and
            typos alike).
    """
    if name not in list_training_sets():
        available = ", ".join(list_training_sets()) or "(none)"
        raise ValueError(
            f"Unknown training set {name!r}. Available: {available}"
        )
    return derived_dir() / name


# ---------------------------------------------------------------------------
# Vocabulary files
# ---------------------------------------------------------------------------
#
# A "vocabulary" is the set of class names the user wants available for a
# session, independent of any training database. They live as CSV files
# under ``core/data/train`` (e.g. ``csv-hufnagel_neume_level_newest.csv``)
# and the class names are the distinct values of the ``classification``
# column. The frontend offers them as a second dropdown on the upload
# screen and previews the resulting class list; the chosen file's classes
# seed the session's autocomplete vocabulary.
#
# The directory may be overridden via ``IC_TRAIN_DIR``. As with training
# sets, a client-supplied name is validated against the enumerated listing
# before any disk access, so path traversal cannot escape the directory.

# Column whose distinct values make up a vocabulary's class names.
VOCABULARY_CLASS_COLUMN = "classification"


def train_dir() -> Path:
    """Directory holding the vocabulary CSV files."""
    override = os.environ.get("IC_TRAIN_DIR")
    if override:
        return Path(override)
    # main.py → ic_api → src → api → <repo root>
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "core" / "data" / "train"


def _has_classification_column(path: Path) -> bool:
    """True if ``path`` is a CSV whose header includes the class column.

    This is what separates a vocabulary file from the other CSVs in the
    directory (VIA annotation exports, etc.), which have no
    ``classification`` column.
    """
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
    except (OSError, UnicodeDecodeError):
        return False
    return VOCABULARY_CLASS_COLUMN in header


def list_vocabularies() -> list[str]:
    """Return the sorted filenames of every vocabulary CSV in :func:`train_dir`."""
    root = train_dir()
    if not root.is_dir():
        return []
    return sorted(
        p.name
        for p in root.glob("*.csv")
        if p.is_file() and _has_classification_column(p)
    )


def resolve_vocabulary(name: str) -> Path:
    """Map a client-supplied vocabulary filename to a safe on-disk path.

    Raises:
        ValueError: If ``name`` is not one of the files enumerated by
            :func:`list_vocabularies` (guards against path traversal and
            typos alike).
    """
    if name not in list_vocabularies():
        available = ", ".join(list_vocabularies()) or "(none)"
        raise ValueError(
            f"Unknown vocabulary {name!r}. Available: {available}"
        )
    return train_dir() / name


def vocabulary_classes(name: str) -> list[str]:
    """Return the sorted distinct class names in a vocabulary CSV.

    The class names are the non-empty values of the
    :data:`VOCABULARY_CLASS_COLUMN` column. ``name`` is validated via
    :func:`resolve_vocabulary` first.
    """
    path = resolve_vocabulary(name)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        names = {
            (row.get(VOCABULARY_CLASS_COLUMN) or "").strip()
            for row in reader
        }
    return sorted(n for n in names if n)


# Why every handler goes through ``store.session(...)``:
# The store's registry lock keeps the dict thread-safe, but a
# retrieved :class:`Session` is a plain mutable object. Two requests
# that hit the same session id (browser double-click, async UI calls,
# retry) would otherwise interleave their mutations and corrupt
# state. ``store.session(id)`` yields the session under a per-session
# lock so each handler's read-mutate-serialise sequence is atomic.
# A missing id raises :class:`KeyError`, which :func:`_key_error_handler`
# maps to a 404 with ``code: "not_found"``.


# ---------------------------------------------------------------------------
# Exception handlers — translate domain errors into HTTP shapes
# ---------------------------------------------------------------------------


@app.exception_handler(StateTransitionError)
async def _state_transition_handler(_request, exc: StateTransitionError) -> JSONResponse:
    # 409 Conflict is the right code for "operation valid in some
    # other state but not the current one" — the resource exists,
    # the request is well-formed, just not allowed right now.
    return JSONResponse(
        status_code=409,
        content=ErrorResponse(detail=str(exc), code="state_conflict").model_dump(),
    )


@app.exception_handler(KeyError)
async def _key_error_handler(_request, exc: KeyError) -> JSONResponse:
    # KeyError comes from Session.find / store.get; both map to 404.
    detail = exc.args[0] if exc.args else str(exc)
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(detail=str(detail), code="not_found").model_dump(),
    )


@app.exception_handler(ValueError)
async def _value_error_handler(_request, exc: ValueError) -> JSONResponse:
    # Most ValueErrors from ic_core are input-validation: empty
    # training pool, rename-to-UNCLASSIFIED, etc.
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(detail=str(exc), code="validation_error").model_dump(),
    )


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


@app.get("/training-sets", response_model=list[str])
def get_training_sets() -> list[str]:
    """List the pre-built training-set filenames available for selection.

    These are the ``*.xml`` GameraXML databases under
    ``core/data/derived``. The frontend renders them as a dropdown on the
    upload screen; the chosen filename is passed back as the
    ``training_xml`` field of :func:`create_session`.
    """
    return list_training_sets()


@app.get("/vocabularies", response_model=list[str])
def get_vocabularies() -> list[str]:
    """List the vocabulary CSV filenames available for selection.

    These are the CSVs under ``core/data/train`` that carry a
    ``classification`` column. The frontend renders them as a dropdown on
    the upload screen; the chosen filename is passed back as the
    ``vocabulary`` field of :func:`create_session`.
    """
    return list_vocabularies()


@app.get("/vocabularies/{name}/classes", response_model=list[str])
def get_vocabulary_classes(name: str) -> list[str]:
    """Return the sorted distinct class names in a vocabulary CSV.

    The frontend fetches this when a vocabulary is selected to preview the
    available class names before the session starts.
    """
    return vocabulary_classes(name)


@app.post("/sessions", response_model=SessionDTO, status_code=201)
async def create_session(
    page_image: Annotated[UploadFile, File(description="Full-page image.")],
    annotations: Annotated[
        UploadFile,
        File(description="MOTHRA JSON or YOLO TXT bbox document."),
    ],
    annotations_format: Annotated[
        AnnotationFormat,
        Form(description="Which annotation parser to use: 'json' or 'yolo'."),
    ],
    # NOTE on parameter ordering and types:
    # * Using ``Depends(get_store)`` directly (rather than the
    #   ``Store`` Annotated alias) — FastAPI mis-classifies the body
    #   when an ``Annotated[..., Depends(...)]`` alias precedes
    #   File/Form parameters in the same signature.
    # * ``class_names`` is a JSON-encoded string, not ``list[str]``
    #   — FastAPI 0.136 treats any ``list[X]`` Form parameter sharing
    #   an endpoint with ``UploadFile`` as a JSON body, which then
    #   makes every multipart field look 'missing'. The JSON-string
    #   shape is a workaround for that bug.
    store: InMemorySessionStore = Depends(get_store),
    class_names: Annotated[
        str | None,
        Form(description="Optional JSON-encoded list[str] of class names."),
    ] = None,
    training_xml: Annotated[
        str | None,
        Form(
            description=(
                "Optional filename of a pre-built training set under "
                "core/data/derived (see GET /training-sets). When given, its "
                "glyphs seed the training pool and a classify round runs "
                "automatically so the working set is labelled with that "
                "training vocabulary before the session is returned."
            ),
        ),
    ] = None,
    vocabulary: Annotated[
        str | None,
        Form(
            description=(
                "Optional filename of a vocabulary CSV under core/data/train "
                "(see GET /vocabularies). When given, the distinct values of "
                "its 'classification' column seed the session's class-name "
                "list (autocomplete vocabulary)."
            ),
        ),
    ] = None,
) -> SessionDTO:
    """Create a session and ingest a page + bbox upload.

    The endpoint accepts ``multipart/form-data`` with two file
    parts (the page image and the bbox document) plus an
    ``annotations_format`` field telling us which parser to use.
    Server-side paths are intentionally *not* accepted — the API
    never opens a file chosen by the client.

    Returns the freshly-ingested session in ``CLASSIFYING`` state.
    The user can immediately call ``POST /sessions/{id}/classify``
    (once they have at least one manual or training glyph) or start
    labelling glyphs via :func:`update_glyph`.

    When ``training_xml`` names a pre-built training set, its glyphs are
    loaded into the training pool and a classify round runs before the
    response is sent, so the returned session is already labelled with
    that training vocabulary.
    """
    parsed_names: list[str] | None = None
    if class_names is not None:
        try:
            parsed_names = json.loads(class_names)
        except json.JSONDecodeError as e:
            raise ValueError(f"class_names is not valid JSON: {e}") from e
        if not isinstance(parsed_names, list) or not all(
            isinstance(n, str) for n in parsed_names
        ):
            raise ValueError("class_names must be a JSON list of strings.")

    # A selected vocabulary contributes its class names to the same
    # autocomplete pool as an explicit ``class_names`` list. Resolving it
    # here (before uploads) also fails fast on a bad filename with a 400.
    if vocabulary:
        vocab_names = vocabulary_classes(vocabulary)
        parsed_names = sorted(set(parsed_names or []) | set(vocab_names))

    # Resolve the optional training set *before* touching uploads so a
    # bad filename fails fast with a 400 rather than after the work.
    training_glyphs: list | None = None
    if training_xml:
        training_glyphs = load_glyphs(resolve_training_set(training_xml))

    page_bytes = await page_image.read()
    annotations_bytes = await annotations.read()
    glyphs = ingest_page(
        page_bytes,
        annotations_bytes,
        format=annotations_format,
    )
    session = Session()
    session.ingest(
        glyphs,
        training_glyphs=training_glyphs,
        class_names=parsed_names,
    )
    # A selected training set means "label this page with that vocabulary
    # now" — run the first classify round server-side so the frontend
    # lands on an already-classified session.
    if training_glyphs:
        session.classify()
    store.create(session)
    return session_to_dto(session)


@app.get("/sessions/{session_id}", response_model=SessionDTO)
def get_session(session_id: str, store: Store) -> SessionDTO:
    """Fetch the full current state of a session."""
    with store.session(session_id) as session:
        return session_to_dto(session)


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, store: Store) -> Response:
    """Discard a session and free its memory."""
    store.delete(session_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Classification & editing
# ---------------------------------------------------------------------------


@app.post("/sessions/{session_id}/classify", response_model=SessionDTO)
def classify(session_id: str, body: ClassifyRequest, store: Store) -> SessionDTO:
    """Re-train and re-classify every non-manual glyph in one round."""
    with store.session(session_id) as session:
        session.classify(k=body.k)
        return session_to_dto(session)


@app.post(
    "/sessions/{session_id}/glyphs/{glyph_id}",
    response_model=GlyphDTO,
)
def update_glyph(
    session_id: str,
    glyph_id: str,
    body: UpdateGlyphRequest,
    store: Store,
) -> GlyphDTO:
    """Partial update of a single glyph (class label, manual flag)."""
    with store.session(session_id) as session:
        new = session.update_glyph(
            glyph_id,
            class_name=body.class_name,
            id_state_manual=body.id_state_manual,
            category=body.category,
        )
        return glyph_to_dto(new)


@app.delete("/sessions/{session_id}/glyphs/{glyph_id}", status_code=204)
def delete_glyph(session_id: str, glyph_id: str, store: Store) -> Response:
    """Drop a glyph from the working set."""
    with store.session(session_id) as session:
        session.delete_glyph(glyph_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


@app.post("/sessions/{session_id}/group", response_model=GlyphDTO)
def manual_group(
    session_id: str,
    body: GroupRequest,
    store: Store,
) -> GlyphDTO:
    """Union the selected glyphs into one new manual glyph."""
    with store.session(session_id) as session:
        grouped = session.manual_group(body.glyph_ids, body.class_name)
        return glyph_to_dto(grouped)


@app.post("/sessions/{session_id}/auto-group", status_code=501)
def auto_group(session_id: str, store: Store) -> JSONResponse:
    """Deferred — see migration plan §'Risks and gotchas' (4).

    Spatial auto-grouping needs a per-glyph page coordinate frame.
    Our ingest path *does* now provide that (the page+bbox flow),
    so this endpoint can be wired up once
    :func:`ic_core.grouping.auto_group_shaped` is implemented. For
    v1 we return 501 explicitly rather than 404 so the frontend can
    show a meaningful 'feature not available yet' message.
    """
    # Touch the session lookup so a request for a nonexistent
    # session still 404s rather than 501.
    store.get(session_id)
    return JSONResponse(
        status_code=501,
        content=ErrorResponse(
            detail=(
                "Auto-grouping is not implemented in v1. See "
                "docs/migration_plan.md §'Risks and gotchas' (4)."
            ),
            code="deferred",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Class-name management
# ---------------------------------------------------------------------------


@app.post(
    "/sessions/{session_id}/classes/{class_name}/rename",
    response_model=SessionDTO,
)
def rename_class(
    session_id: str,
    class_name: str,
    body: RenameClassRequest,
    store: Store,
) -> SessionDTO:
    """Rename a class across the working set, training set, and autocomplete."""
    with store.session(session_id) as session:
        session.rename_class(class_name, body.new_name)
        return session_to_dto(session)


@app.delete(
    "/sessions/{session_id}/classes/{class_name}",
    response_model=SessionDTO,
)
def delete_class(session_id: str, class_name: str, store: Store) -> SessionDTO:
    """Drop a class (and dotted-namespace subclasses) from the autocomplete list."""
    with store.session(session_id) as session:
        session.delete_class(class_name)
        return session_to_dto(session)


# ---------------------------------------------------------------------------
# Persistence checkpoints
# ---------------------------------------------------------------------------


@app.post("/sessions/{session_id}/save", response_model=SessionDTO)
def save_session(session_id: str, store: Store) -> SessionDTO:
    """No-op for the in-memory store; returns the current state.

    Exposed so the frontend can call it on a 'save' button without
    branching on the storage backend. Once :mod:`ic_api.store` is
    replaced with a SQLite/Postgres implementation this will flush.
    """
    with store.session(session_id) as session:
        return session_to_dto(session)


@app.post("/sessions/{session_id}/complete")
def complete_session(session_id: str, store: Store) -> Response:
    """Finalise the session and stream back the GameraXML export.

    The session transitions to ``EXPORT`` (terminal). The frontend
    should treat the returned XML as the canonical artefact for
    downstream MEI pipelines.

    Response body is ``application/xml``, not JSON, because the XML
    *is* the deliverable. The session remains in the store so the
    caller can ``DELETE`` it explicitly once they've saved the file.
    """
    with store.session(session_id) as session:
        session.complete()
        payload = dumps_glyphs(session.glyphs)
        filename = f'attachment; filename="ic-session-{session.id}.xml"'
    return Response(
        content=payload,
        media_type="application/xml",
        headers={"Content-Disposition": filename},
    )


# ---------------------------------------------------------------------------
# Entry point for `uv run ic-api`
# ---------------------------------------------------------------------------


def run() -> None:
    """Launch the dev server. Used by the ``ic-api`` console script."""
    import uvicorn

    uvicorn.run(
        "ic_api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
