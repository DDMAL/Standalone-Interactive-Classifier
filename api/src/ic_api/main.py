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
import re
import threading
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from lxml import etree

from ic_api.schemas import (
    ClassifyRequest,
    ErrorResponse,
    GlyphDTO,
    GroupRequest,
    RebinarizeRequest,
    RenameClassRequest,
    SessionDTO,
    SessionSummaryDTO,
    SplitRequest,
    UpdateGlyphRequest,
    glyph_to_dto,
    session_to_dto,
)
from ic_api.store import SessionStore, default_store, store_backend_info
from ic_core.ingest import (
    AnnotationFormat,
    BinarizationMethod,
    binarize_page,
    ingest_page,
)
from ic_core.classifier import UNCLASSIFIED, filter_parts
from ic_core.glyph import CATEGORY_NEUMES
from ic_core.io_xml import dumps_glyphs, load_glyphs_bytes
from ic_core.state import Session, StateTransitionError

# ---------------------------------------------------------------------------
# Starlette 1.x caps each multipart part at 1 MB by default, which is too
# small for high-res page scans.  FastAPI calls request.form() without a
# max_part_size argument, so we raise the default here to avoid spurious 413s.
# Override with the MAX_UPLOAD_BYTES env var when needed.
# ---------------------------------------------------------------------------
_MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

from starlette import requests as _starlette_requests  # noqa: E402

_orig_get_form = _starlette_requests.Request._get_form


async def _patched_get_form(
    self: _starlette_requests.Request,
    *,
    max_files: int | float = 1000,
    max_fields: int | float = 1000,
    max_part_size: int = _MAX_UPLOAD_BYTES,
) -> _starlette_requests.FormData:
    return await _orig_get_form(
        self,
        max_files=max_files,
        max_fields=max_fields,
        max_part_size=max_part_size,
    )


_starlette_requests.Request._get_form = _patched_get_form  # type: ignore[method-assign]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_stem(filename: str | None) -> str:
    """Reduce an uploaded filename to a safe stem for the export name.

    Strips any client-supplied directory components and the file
    extension, then keeps only filename-safe characters so the value
    can be dropped into a ``Content-Disposition`` header and saved to
    disk verbatim. Returns ``""`` when nothing usable remains, in which
    case the caller falls back to the session id.
    """
    if not filename:
        return ""
    stem = Path(filename).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return safe


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


def get_store() -> SessionStore:
    """Dependency-injection point for the session store.

    The default returns the module-level :data:`ic_api.store.default_store`
    — an in-memory or Postgres-backed store depending on the environment
    (see :func:`ic_api.store.build_default_store`). Tests override this with
    a fresh store via ``app.dependency_overrides``.
    """
    return default_store


Store = Annotated[SessionStore, Depends(get_store)]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz(store: Store) -> dict[str, object]:
    """Liveness probe that also reports which session store is live.

    The store backend is the difference between "a restart costs a hiccup"
    and "a restart costs every session in flight": on the in-memory store,
    an OOM kill or a redeploy drops the whole registry and the frontend's
    next call fails with ``Unknown session id``. Deployments select the
    backend purely by whether ``DATABASE_URL`` / ``IC_DATABASE_URL`` is in
    the environment, which is easy to omit and, until now, invisible from
    outside the process. Probing this endpoint answers "is this deployment
    actually persisting sessions?" without shell or log access.

    ``backend`` / ``persistent`` report what the environment *asked for*.
    Holding a Postgres store proves nothing on its own — it connects
    lazily, so a typo'd DSN or an unreachable database looks identical to
    a working one at construction time. ``reachable`` closes that gap by
    round-tripping the database (``SELECT 1``, bounded by
    ``db_store.CONNECT_TIMEOUT_SECONDS``): ``true`` means sessions really
    are being persisted, ``false`` means the deployment believes it
    configured persistence but hasn't got it, and ``null`` means the
    backend has nothing to reach (the in-memory store).

    ``status`` stays ``"ok"`` even when the database is unreachable, so
    wiring this up as a liveness probe can't turn a DB hiccup into a
    restart loop — the diagnosis belongs in the payload, not the status
    code. ``sessions`` counts what this process holds in its registry /
    hot cache.
    """
    info = store_backend_info()
    # Only the Postgres store defines ping(); the in-memory store has no
    # database to be unreachable, so `reachable` stays null for it.
    ping = getattr(store, "ping", None)
    if ping is None:
        info["reachable"] = None
    else:
        try:
            ping()
            info["reachable"] = True
        except Exception as exc:
            info["reachable"] = False
            info["error"] = str(exc).strip().splitlines()[0][:200]
    try:
        n_sessions: int | None = len(store)  # type: ignore[arg-type]
    except TypeError:  # a store without __len__ (e.g. a test double)
        n_sessions = None
    return {"status": "ok", "store": info, "sessions": n_sessions}


# ---------------------------------------------------------------------------
# Built-in training-set presets
# ---------------------------------------------------------------------------
#
# Pre-built GameraXML training databases live under ``core/data/presets``
# (e.g. ``Hufnagel.xml``, ``Square.xml``). The frontend offers them as
# checkboxes on the upload screen; the glyphs of every picked preset are
# concatenated into the session's training pool — alongside any uploaded
# training sets — so the first classify round applies that vocabulary
# directly.
#
# The directory is resolved relative to the repo root and may be overridden
# via ``IC_PRESETS_DIR``. As with vocabularies, the API only ever opens
# files it has itself enumerated here — a client-supplied name is validated
# against that listing before any disk access, so path traversal
# (``../secrets.xml``) cannot escape the directory.


def presets_dir() -> Path:
    """Directory holding the built-in GameraXML training-set presets."""
    override = os.environ.get("IC_PRESETS_DIR")
    if override:
        return Path(override)
    # main.py → ic_api → src → api → <repo root>
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "core" / "data" / "presets"


def list_presets() -> list[str]:
    """Return the sorted filenames of every ``.xml`` preset in :func:`presets_dir`."""
    root = presets_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.glob("*.xml") if p.is_file())


def resolve_preset(name: str) -> Path:
    """Map a client-supplied preset filename to a safe on-disk path.

    Raises:
        ValueError: If ``name`` is not one of the files enumerated by
            :func:`list_presets` (guards against path traversal and
            typos alike).
    """
    if name not in list_presets():
        available = ", ".join(list_presets()) or "(none)"
        raise ValueError(
            f"Unknown training preset {name!r}. Available: {available}"
        )
    return presets_dir() / name


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


# ---------------------------------------------------------------------------
# Session-building helpers (shared by /sessions and /sessions/from-staging)
# ---------------------------------------------------------------------------


def _resolve_class_names(
    class_names: str | None, vocabulary: str | None
) -> list[str] | None:
    """Merge an explicit JSON class-name list with a vocabulary's classes.

    ``class_names`` is a JSON-encoded ``list[str]`` (see the multipart
    workaround note on :func:`create_session`); ``vocabulary`` is a CSV
    filename whose distinct classes are unioned in. Either may be ``None``.
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
    if vocabulary:
        vocab_names = vocabulary_classes(vocabulary)
        parsed_names = sorted(set(parsed_names or []) | set(vocab_names))
    return parsed_names


async def _parse_training_files(training_files) -> list:
    """Concatenate glyphs from uploaded GameraXML (.xml) training sets.

    Returns an empty list when no files were given. Raises ``ValueError``
    (→ 400) on a non-``.xml`` name or invalid XML, before any page work so
    a bad file fails fast.
    """
    if not training_files:
        return []
    training_glyphs: list = []
    for tf in training_files:
        name = tf.filename or ""
        if not name.lower().endswith(".xml"):
            raise ValueError(f"{name!r} is not a .xml file.")
        try:
            training_glyphs.extend(load_glyphs_bytes(await tf.read()))
        except etree.XMLSyntaxError as e:
            raise ValueError(f"{name!r} is not valid XML: {e}") from e
    return training_glyphs


def _parse_training_presets(training_presets: str | None) -> list:
    """Concatenate glyphs from built-in training-set presets.

    ``training_presets`` is a JSON-encoded ``list[str]`` of preset filenames
    (see the multipart workaround note on :func:`create_session`). Each name
    is validated against :func:`list_presets` before any disk access, so a
    bad name fails fast (→ 400). Returns an empty list when none were given.
    """
    if not training_presets:
        return []
    try:
        names = json.loads(training_presets)
    except json.JSONDecodeError as e:
        raise ValueError(f"training_presets is not valid JSON: {e}") from e
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("training_presets must be a JSON list of strings.")
    preset_glyphs: list = []
    for name in names:
        path = resolve_preset(name)
        try:
            preset_glyphs.extend(load_glyphs_bytes(path.read_bytes()))
        except OSError as e:
             raise ValueError(f"Could not read preset {name!r}: {e}") from e
        except etree.XMLSyntaxError as e:
            raise ValueError(f"Preset {name!r} is not valid XML: {e}") from e
    return preset_glyphs


async def _training_glyphs_by_source(
    training_files, training_presets
) -> tuple[list, list]:
    """Load preset + uploaded training glyphs, kept separate by source.

    Returns ``(preset_glyphs, uploaded_glyphs)`` — each an independent list
    (either may be empty). :func:`_finalize_session` concatenates them for
    the classifier but records which is which (by glyph id) so the export
    screen can toggle the two pools independently.
    """
    return (
        _parse_training_presets(training_presets),
        await _parse_training_files(training_files),
    )


def _finalize_session(
    store: SessionStore,
    *,
    page_bytes: bytes,
    page_media_type: str | None,
    source_filename: str | None,
    annotations_bytes: bytes,
    annotations_format: AnnotationFormat,
    binarization_method: BinarizationMethod,
    parsed_names: list[str] | None,
    preset_glyphs: list,
    uploaded_glyphs: list,
    project_id: int | None = None,
    image_id: str | None = None,
) -> SessionDTO:
    """Ingest a page + bboxes into a stored session and return its DTO.

    Shared by :func:`create_session` (direct upload) and
    :func:`create_session_from_staging` (mothra-staged page + bboxes).

    ``preset_glyphs`` and ``uploaded_glyphs`` are the two training-set
    sources; they are concatenated (presets first) into the session's
    training pool, and their glyph ids are recorded so the export screen can
    later toggle the two pools independently.
    """
    glyphs = ingest_page(
        page_bytes, annotations_bytes,
        format=annotations_format, method=binarization_method,
    )
    training_glyphs = [*preset_glyphs, *uploaded_glyphs]
    # Keep the full-page mask so manual grouping can recover ink that falls
    # in the gap between child glyph bboxes (never copied into any crop).
    # It must use the SAME method as the glyphs, or it disagrees with them.
    page_mask = binarize_page(page_bytes, method=binarization_method)
    session = Session()
    session.ingest(
        glyphs,
        training_glyphs=training_glyphs,
        class_names=parsed_names,
        page_mask=page_mask,
        source_name=_source_stem(source_filename),
    )
    session.preset_training_ids = {g.id for g in preset_glyphs}
    session.uploaded_training_ids = {g.id for g in uploaded_glyphs}
    # Retain the original page bytes so GET /sessions/{id}/page can serve
    # them back to a frontend that did not perform the upload itself.
    session.page_bytes = page_bytes
    session.page_media_type = page_media_type or "application/octet-stream"
    # Retain the bboxes + method too, so POST /sessions/{id}/binarization can
    # re-run ingest under a different method without a fresh upload.
    session.annotations_bytes = annotations_bytes
    session.annotations_format = annotations_format
    session.binarization_method = binarization_method
    # A training set means "label this page with that vocabulary now" — run
    # the first classify round so the frontend lands already-classified.
    if training_glyphs:
        session.classify()
    # project_id/image_id (mothra's owning page) let a persistent store key
    # the session for resume; None for IC's own upload path.
    store.create(session, project_id=project_id, image_id=image_id)
    return session_to_dto(session)


# ---------------------------------------------------------------------------
# Staging — a page + bboxes pushed by an embedding host (mothra) before the
# user has chosen training data / vocabulary
# ---------------------------------------------------------------------------
#
# mothra owns the page image and generates the bboxes, but we still want the
# user to see IC's real create-session screen so they can add training sets
# and pick a vocabulary. mothra therefore *stages* the page + bboxes here and
# deep-links the SPA to ``/?staged=<id>``; the UploadView pre-fills the staged
# page + bboxes (locked) and leaves only training/vocabulary to the user. On
# submit the SPA calls POST /sessions/from-staging, which pairs the staged
# page + bboxes with the user's choices. Staging entries are single-use and
# live only in memory — the same single-user-tool tradeoff as the session
# store.


class _Staged:
    __slots__ = (
        "page_bytes",
        "page_media_type",
        "page_name",
        "annotations_bytes",
        "annotations_format",
        # The mothra project + image this page belongs to, carried through
        # to the created session so a persistent store can key it for resume.
        "project_id",
        "image_id",
    )

    def __init__(
        self,
        page_bytes: bytes,
        page_media_type: str,
        page_name: str,
        annotations_bytes: bytes,
        annotations_format: AnnotationFormat,
        project_id: int | None = None,
        image_id: str | None = None,
    ) -> None:
        self.page_bytes = page_bytes
        self.page_media_type = page_media_type
        self.page_name = page_name
        self.annotations_bytes = annotations_bytes
        self.annotations_format = annotations_format
        self.project_id = project_id
        self.image_id = image_id


_staging: dict[str, _Staged] = {}
_staging_lock = threading.Lock()


def _annotation_count(annotations_bytes: bytes, fmt: AnnotationFormat) -> int:
    """Best-effort count of boxes in a staged annotation document."""
    try:
        if fmt == "json":
            doc = json.loads(annotations_bytes)
            if isinstance(doc, list):
                doc = doc[0] if doc else {}
            return len(doc.get("annotations", []))
        # YOLO: one non-empty, non-comment line per box.
        return sum(
            1
            for line in annotations_bytes.decode("utf-8", "ignore").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except Exception:
        return 0


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


@app.get("/training-presets", response_model=list[str])
def get_training_presets() -> list[str]:
    """List the built-in training-set preset filenames available for selection.

    These are the GameraXML (.xml) files under ``core/data/presets``. The
    frontend renders them as checkboxes on the upload screen; the chosen
    filenames are passed back as the ``training_presets`` field of
    :func:`create_session` and concatenated into the training pool.
    """
    return list_presets()


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
    binarization_method: Annotated[
        BinarizationMethod,
        Form(
            description=(
                "How to binarise the page: 'global' (fixed ≤127 cutoff), "
                "'otsu' (histogram-derived global cutoff), or 'sauvola' "
                "(local adaptive — recovers faint staff lines on uneven "
                "parchment). Baked into each glyph's mask at ingest."
            ),
        ),
    ] = "global",
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
    store: SessionStore = Depends(get_store),
    class_names: Annotated[
        str | None,
        Form(description="Optional JSON-encoded list[str] of class names."),
    ] = None,
    training_files: Annotated[
        list[UploadFile] | None,
        File(
            description=(
                "Optional GameraXML (.xml) training-set uploads. When given, "
                "the glyphs from every file are concatenated to seed the "
                "training pool and a classify round runs automatically so the "
                "working set is labelled with that training vocabulary before "
                "the session is returned."
            ),
        ),
    ] = None,
    training_presets: Annotated[
        str | None,
        Form(
            description=(
                "Optional JSON-encoded list[str] of built-in preset filenames "
                "under core/data/presets (see GET /training-presets). Their "
                "glyphs are concatenated into the training pool ahead of any "
                "uploaded training_files. Encoded as a JSON string for the "
                "same multipart reason as class_names."
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

    When ``training_presets`` and/or ``training_files`` carry GameraXML
    training sets, the glyphs from every source are concatenated into the
    training pool (presets first, then uploads) and a classify round runs
    before the response is sent, so the returned session is already
    labelled with that training vocabulary.
    """
    parsed_names = _resolve_class_names(class_names, vocabulary)
    preset_glyphs, uploaded_glyphs = await _training_glyphs_by_source(
        training_files, training_presets
    )

    page_bytes = await page_image.read()
    annotations_bytes = await annotations.read()
    return _finalize_session(
        store,
        page_bytes=page_bytes,
        page_media_type=page_image.content_type,
        source_filename=annotations.filename,
        annotations_bytes=annotations_bytes,
        annotations_format=annotations_format,
        binarization_method=binarization_method,
        parsed_names=parsed_names,
        preset_glyphs=preset_glyphs,
        uploaded_glyphs=uploaded_glyphs,
    )


@app.post("/staging", status_code=201)
async def stage_page(
    page_image: Annotated[UploadFile, File(description="Full-page image.")],
    annotations: Annotated[
        UploadFile, File(description="MOTHRA JSON or YOLO TXT bbox document.")
    ],
    annotations_format: Annotated[
        AnnotationFormat, Form(description="'json' or 'yolo'.")
    ],
    project_id: Annotated[
        int | None,
        Form(description="Owning mothra project id (for session resume)."),
    ] = None,
    image_id: Annotated[
        str | None,
        Form(description="Owning mothra image id (for session resume)."),
    ] = None,
):
    """Stage a page + bboxes for a later from-staging session creation.

    Used by an embedding host (mothra) that owns the page and generates the
    bboxes but wants the user to complete IC's own create-session screen
    (training data + vocabulary). Returns a single-use ``staging_id`` the
    SPA references via ``/?staged=<id>``.

    ``project_id`` / ``image_id`` identify the mothra page this staging
    belongs to; they ride through to the created session so a persistent
    store can key it and resume it when the user returns to the page.
    """
    page_bytes = await page_image.read()
    annotations_bytes = await annotations.read()
    staging_id = uuid.uuid4().hex
    staged = _Staged(
        page_bytes=page_bytes,
        page_media_type=page_image.content_type or "application/octet-stream",
        page_name=page_image.filename or "page",
        annotations_bytes=annotations_bytes,
        annotations_format=annotations_format,
        project_id=project_id,
        image_id=image_id,
    )
    with _staging_lock:
        _staging[staging_id] = staged
    return {
        "staging_id": staging_id,
        "page_name": staged.page_name,
        "annotations_format": annotations_format,
        "annotation_count": _annotation_count(annotations_bytes, annotations_format),
    }


def _get_staged_or_404(staging_id: str) -> "_Staged | JSONResponse":
    with _staging_lock:
        staged = _staging.get(staging_id)
    if staged is None:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                detail=f"Unknown staging id: {staging_id!r}", code="not_found"
            ).model_dump(),
        )
    return staged


@app.get("/staging/{staging_id}")
def get_staging(staging_id: str):
    """Return metadata for a staged page + bboxes (for the UploadView)."""
    staged = _get_staged_or_404(staging_id)
    if isinstance(staged, JSONResponse):
        return staged
    return {
        "staging_id": staging_id,
        "page_name": staged.page_name,
        "annotations_format": staged.annotations_format,
        "annotation_count": _annotation_count(
            staged.annotations_bytes, staged.annotations_format
        ),
    }


@app.get("/staging/{staging_id}/page")
def get_staging_page(staging_id: str) -> Response:
    """Serve a staged page image (lets the UploadView preview it)."""
    staged = _get_staged_or_404(staging_id)
    if isinstance(staged, JSONResponse):
        return staged
    return Response(content=staged.page_bytes, media_type=staged.page_media_type)


@app.post("/sessions/from-staging", response_model=SessionDTO, status_code=201)
async def create_session_from_staging(
    staging_id: Annotated[str, Form(description="Id returned by POST /staging.")],
    # See create_session for why Depends(get_store) is used inline (not the
    # Store alias) and why class_names is a JSON string.
    store: SessionStore = Depends(get_store),
    binarization_method: Annotated[
        BinarizationMethod,
        Form(description="'global', 'otsu', or 'sauvola'; see POST /sessions."),
    ] = "global",
    class_names: Annotated[
        str | None, Form(description="Optional JSON-encoded list[str].")
    ] = None,
    training_files: Annotated[
        list[UploadFile] | None,
        File(description="Optional GameraXML (.xml) training-set uploads."),
    ] = None,
    training_presets: Annotated[
        str | None,
        Form(
            description=(
                "Optional JSON-encoded list[str] of built-in preset filenames "
                "under core/data/presets (see GET /training-presets); "
                "concatenated into the training pool ahead of training_files."
            ),
        ),
    ] = None,
    vocabulary: Annotated[
        str | None, Form(description="Optional vocabulary CSV filename.")
    ] = None,
) -> SessionDTO:
    """Create a session from staged page + bboxes plus the user's choices.

    Pairs the page + bboxes staged by :func:`stage_page` with the
    training sets / vocabulary the user picked on IC's create-session
    screen. The staging entry is consumed (single-use).
    """
    with _staging_lock:
        staged = _staging.pop(staging_id, None)
    if staged is None:
        # Maps to 404 via _key_error_handler.
        raise KeyError(f"Unknown staging id: {staging_id!r}")
    parsed_names = _resolve_class_names(class_names, vocabulary)
    preset_glyphs, uploaded_glyphs = await _training_glyphs_by_source(
        training_files, training_presets
    )
    return _finalize_session(
        store,
        page_bytes=staged.page_bytes,
        page_media_type=staged.page_media_type,
        source_filename=staged.page_name,
        annotations_bytes=staged.annotations_bytes,
        annotations_format=staged.annotations_format,
        binarization_method=binarization_method,
        parsed_names=parsed_names,
        preset_glyphs=preset_glyphs,
        uploaded_glyphs=uploaded_glyphs,
        project_id=staged.project_id,
        image_id=staged.image_id,
    )


@app.get("/sessions", response_model=list[SessionSummaryDTO])
def list_sessions(
    store: Store, project_id: int | None = None
) -> list[SessionSummaryDTO]:
    """List stored sessions as lightweight summaries, most-recent first.

    Powers the standalone frontend's "resume a saved session" list: the
    embedding-host (mothra) resume path goes through
    :func:`lookup_session` keyed by project + page, but IC's own upload
    screen has no such key, so it enumerates everything here and lets the
    user pick. Summaries omit glyph masks and the page image; the client
    fetches the full session via :func:`get_session` on open.

    When ``project_id`` is given, only sessions staged for that project are
    returned — this is how mothra's per-project "saved sessions" management
    view scopes the otherwise-global list so it never surfaces (or lets a
    user delete) another project's sessions.

    Against the in-memory store this reflects only sessions created since
    the last restart; against the persistent store it spans restarts and
    carries an ``updated_at`` timestamp.
    """
    summaries = store.list_sessions()
    if project_id is not None:
        summaries = [s for s in summaries if s.project_id == project_id]
    return [
        SessionSummaryDTO(
            id=s.id,
            state=s.state,
            source_name=s.source_name,
            n_glyphs=s.n_glyphs,
            updated_at=s.updated_at,
            project_id=s.project_id,
            image_id=s.image_id,
        )
        for s in summaries
    ]


@app.delete("/sessions")
def clear_sessions(store: Store) -> dict:
    """Discard *every* stored session and free its memory.

    Backs the resume list's "clear all" action. Wipes the whole store — the
    in-memory registry, or every row of ``ic_sessions`` in the persistent
    backend — and returns how many sessions were removed. There is no undo;
    the client confirms before calling this. Distinct from
    :func:`delete_session`, which drops one session by id.
    """
    return {"deleted": store.clear()}


# Declared before GET /sessions/{session_id} so "lookup" isn't captured as a
# session id by the path parameter route.
@app.get("/sessions/lookup")
def lookup_session(project_id: int, image_id: str, store: Store):
    """Return the resumable session id for a mothra project + page, if any.

    The embedding host calls this before staging a page: a hit means the
    user has a saved (still-editable) session for that page to resume; a
    404 means there's nothing to resume and the host should stage fresh.
    Only meaningful against a persistent store — the in-memory store only
    knows sessions created in the current process.
    """
    session_id = store.lookup(project_id, image_id)
    if session_id is None:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                detail="No resumable session for this page.", code="not_found"
            ).model_dump(),
        )
    return {"session_id": session_id}


@app.get("/sessions/{session_id}", response_model=SessionDTO)
def get_session(session_id: str, store: Store) -> SessionDTO:
    """Fetch the full current state of a session."""
    with store.session(session_id) as session:
        return session_to_dto(session)


@app.get("/sessions/{session_id}/page")
def get_session_page(session_id: str, store: Store) -> Response:
    """Serve the original uploaded page image for a session.

    Lets a frontend that did not perform the upload itself (an
    embedding host that created the session via :func:`create_session`
    and deep-linked into the SPA) render the page. Returns 404 when the
    session was created without a page image.
    """
    with store.session(session_id) as session:
        if not session.page_bytes:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    detail="Session has no page image.",
                    code="not_found",
                ).model_dump(),
            )
        return Response(
            content=session.page_bytes,
            media_type=session.page_media_type or "application/octet-stream",
        )


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


@app.post("/sessions/{session_id}/binarization", response_model=SessionDTO)
def rebinarize(
    session_id: str, body: RebinarizeRequest, store: Store
) -> SessionDTO:
    """Switch the page's binarisation method and rebuild every glyph mask.

    Re-binarises the retained page and hands the new full-page mask to
    :meth:`ic_core.state.Session.rebinarize`, which re-slices every glyph's
    own bbox out of it. Everything the user built survives — labels, manual
    flags, categories, and manual splits and groups — because a glyph's mask
    is by construction a slice of the page mask at its bbox. Auto labels
    carry over but are stale under the new pixels, so callers normally chain
    a classify round (the frontend's toolbar does).

    Only the page image is needed: the bbox document never changes, so
    re-running ingest could only reproduce the boxes the session already
    holds — and doing so used to drop split children and grouped glyphs,
    whose ids no ingest can produce.
    """
    with store.session(session_id) as session:
        if session.page_bytes is None:
            # Sessions created without a page upload (legacy XML import)
            # have nothing to re-binarise from.
            raise ValueError(
                "This session has no retained page image to re-binarise; "
                "the method can only be changed on sessions created from "
                "a page upload."
            )
        page_mask = binarize_page(session.page_bytes, method=body.method)
        session.rebinarize(page_mask=page_mask, method=body.method)
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


@app.delete(
    "/sessions/{session_id}/training-glyphs/{glyph_id}",
    response_model=SessionDTO,
)
def delete_training_glyph(
    session_id: str, glyph_id: str, store: Store
) -> SessionDTO:
    """Drop a single glyph from the session's external training pool.

    Returns the updated session (rather than 204) so the frontend can
    refresh the training-set count and the training-data panel from one
    response without a follow-up fetch. Does not re-classify — the caller
    re-runs classify explicitly if it wants the working set re-scored
    against the smaller pool.
    """
    with store.session(session_id) as session:
        session.delete_training_glyph(glyph_id)
        return session_to_dto(session)


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


@app.post(
    "/sessions/{session_id}/glyphs/{glyph_id}/split",
    response_model=list[GlyphDTO],
)
def manual_split(
    session_id: str,
    glyph_id: str,
    body: SplitRequest,
    store: Store,
) -> list[GlyphDTO]:
    """Slice a glyph into N children along user-drawn rectangles.

    The parent glyph is removed from the working set and replaced
    (at its original index) by N ``UNCLASSIFIED`` children, one per
    rectangle. Rectangles are ``[ulx, uly, ncols, nrows]`` in page
    coordinates; the same frame the glyph's bbox is in.

    Returns the list of new children in insertion order. The next
    classify round will label them.
    """
    with store.session(session_id) as session:
        children = session.manual_split(glyph_id, body.regions)
        return [glyph_to_dto(c) for c in children]


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
    """Persist the current session state and return it.

    With the persistent (Postgres) store this flushes the session on the
    :meth:`session` context exit — the same write-through that already
    happens after every mutating endpoint, so 'Save' is an explicit
    checkpoint rather than the only persistence point. With the in-memory
    store it's a no-op that simply echoes the current state.
    """
    with store.session(session_id) as session:
        return session_to_dto(session)


@app.post("/sessions/{session_id}/complete")
def complete_session(
    session_id: str,
    store: Store,
    page: bool = False,
    manual_neumes: bool = False,
    preset_training: bool = False,
    uploaded_training: bool = False,
    finalize: bool = True,
) -> Response:
    """Stream back the GameraXML export for the session.

    The first call transitions the session from ``CLASSIFYING`` to
    ``EXPORT`` (idempotent: subsequent calls are no-ops on the state
    machine). Once in ``EXPORT``, further mutations return 409 but
    re-export is always allowed. The returned XML is a snapshot of the
    current working set, the canonical artefact for downstream MEI
    pipelines.

    ``finalize=false`` exports *without* that transition: the session
    stays in ``CLASSIFYING``, editable and resumable, and the
    export-time hygiene (strip transient ``_group``/``_delete`` parts,
    drop ``UNCLASSIFIED`` training entries) is applied to the exported
    copy only, leaving the live session untouched. An embedding host
    that treats the XML as an intermediate artefact rather than an
    end-of-life one needs this: mothra hands the export to its MEI
    encoder but still lets the user reopen the page and correct it
    afterwards, which a terminal ``EXPORT`` session forbids (see
    :func:`lookup_session` — a completed session is not resumable, so
    finalising on export silently discards every correction behind it).

    The caller picks which sections to fold into a single GameraXML
    document via independent boolean flags (the export screen's
    checkboxes):

    * ``page`` — every working glyph on the annotated page.
    * ``manual_neumes`` — only the working neumes the user labelled by
      hand (a subset of ``page``).
    * ``preset_training`` — the training glyphs that came from a
      built-in preset.
    * ``uploaded_training`` — the training glyphs the user uploaded.

    Sections are concatenated in that order and de-duplicated by glyph
    id (so selecting both ``page`` and ``manual_neumes`` never emits a
    glyph twice). At least one flag must be set, else 400.

    Response body is ``application/xml``, not JSON, because the XML
    *is* the deliverable. The session remains in the store, still
    editable, so the caller can re-export or ``DELETE`` it explicitly
    once they've saved the file.
    """
    if not (page or manual_neumes or preset_training or uploaded_training):
        raise ValueError(
            "Select at least one section to include in the export."
        )
    with store.session(session_id) as session:
        if finalize:
            # Finalise the session (CLASSIFYING → EXPORT) on the first
            # export. Idempotent: already-EXPORT sessions are a no-op, so
            # repeated exports work fine. After this call, mutations raise
            # 409. Session.complete() does the hygiene pass in place.
            session.complete()
            page_glyphs = session.glyphs
            training_glyphs = [
                g
                for g in session.training_glyphs
                if g.class_name != UNCLASSIFIED
            ]
        else:
            # Same hygiene, applied to the *exported* glyphs only, so the
            # live session is left in CLASSIFYING and fully re-editable.
            page_glyphs = filter_parts(session.glyphs)
            training_glyphs = [
                g
                for g in filter_parts(session.training_glyphs)
                if g.class_name != UNCLASSIFIED
            ]
        selected: list = []
        seen: set[str] = set()

        def add(glyphs) -> None:
            for g in glyphs:
                if g.id not in seen:
                    seen.add(g.id)
                    selected.append(g)

        if page:
            add(page_glyphs)
        if manual_neumes:
            add(
                g
                for g in page_glyphs
                if g.category == CATEGORY_NEUMES and g.id_state_manual
            )
        if preset_training:
            add(
                g
                for g in training_glyphs
                if g.id in session.preset_training_ids
            )
        if uploaded_training:
            add(
                g
                for g in training_glyphs
                if g.id in session.uploaded_training_ids
            )
        payload = dumps_glyphs(selected)
        # Tag the filename with the chosen sections so a user exporting
        # several variants from one session gets self-describing files.
        tags = [
            name
            for flag, name in (
                (page, "page"),
                (manual_neumes, "manual-neumes"),
                (preset_training, "preset"),
                (uploaded_training, "uploaded"),
            )
            if flag
        ]
        # Prefer the original bbox document's name so a user exporting
        # several pages gets self-describing files; fall back to the
        # opaque session id when no usable source name was captured.
        stem = session.source_name or session.id
        filename = (
            f'attachment; filename="ic-session-{stem}-{"-".join(tags)}.xml"'
        )
    return Response(
        content=payload,
        media_type="application/xml",
        headers={"Content-Disposition": filename},
    )


# ---------------------------------------------------------------------------
# Static frontend (production single-origin deploy)
# ---------------------------------------------------------------------------
#
# When the built frontend has been copied to ``static/`` next to this module
# (see the Dockerfile), serve it from the same origin as the API. This mount
# is registered *after* every API route above, so ``/sessions`` etc. still
# resolve to their handlers; everything else falls through to the SPA.
# ``html=True`` serves ``index.html`` for unknown paths so client-side routing
# works. In local dev the directory is absent and this is simply skipped.
_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="ui")


# ---------------------------------------------------------------------------
# Entry point for `uv run ic-api`
# ---------------------------------------------------------------------------


def run() -> None:
    """Launch the dev server. Used by the ``ic-api`` console script."""
    import uvicorn

    uvicorn.run(
        "ic_api.main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
