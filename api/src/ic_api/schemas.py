"""Pydantic request/response schemas for the HTTP API.

The DTOs are intentionally kept separate from the domain
:class:`ic_core.glyph.Glyph` dataclass:

* Glyph is a *frozen* in-memory record — fast to construct, slots,
  no validation overhead.
* The DTOs here add JSON-friendly conveniences (base64 PNG preview,
  string-keyed enum for state) and absorb breaking schema changes
  without forcing the domain model to twitch.

Conversion is one-way and explicit: :func:`glyph_to_dto` walks a
domain glyph into the response model. We do not reconstruct domain
glyphs from inbound JSON — the API never accepts a full glyph
payload, only field-level updates (``class_name``, ``id_state_manual``).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ic_core.glyph import Glyph
from ic_core.ingest import BinarizationMethod
from ic_core.state import ClassifierState, Session


# ---------------------------------------------------------------------------
# Response — read paths
# ---------------------------------------------------------------------------


class GlyphDTO(BaseModel):
    """JSON shape of a single glyph sent to the frontend."""

    id: str = Field(..., description="Stable UUID4 hex (32 chars).")
    class_name: str
    confidence: float
    id_state_manual: bool

    # Coarse MOTHRA category: "Text" | "Neumes" | "Staves". The frontend
    # groups the glyph grid by this; only Neumes carry a meaningful class_name.
    category: Literal["Text", "Neumes", "Staves"]
    # Page-coordinate frame inherited from the bbox annotation file.
    ulx: int
    uly: int
    ncols: int
    nrows: int

    # Base64-encoded PNG preview for the frontend `<img>` tag. The
    # raw RLE is not sent — the frontend doesn't decode it.
    image_b64: str = Field(..., description="Base64 PNG, ASCII.")


class SessionSummaryDTO(BaseModel):
    """Lightweight session metadata for the resume list (GET /sessions).

    Deliberately omits glyphs, training glyphs, and the page image so the
    list stays cheap to build and small on the wire — the frontend hydrates
    the full :class:`SessionDTO` only once the user opens a session.
    """

    id: str
    state: ClassifierState
    source_name: str = Field(
        "", description="Human-facing label (the uploaded bbox filename stem)."
    )
    n_glyphs: int = Field(..., description="Working-set glyph count.")
    updated_at: str | None = Field(
        None,
        description=(
            "ISO-8601 last-modified time; null on the in-memory store, "
            "which keeps no timestamp."
        ),
    )
    project_id: int | None = Field(
        None, description="Owning mothra project id, if the session is keyed."
    )
    image_id: str | None = Field(
        None, description="Owning mothra image id, if the session is keyed."
    )


class SessionDTO(BaseModel):
    """JSON shape of an entire session."""

    id: str
    state: ClassifierState
    glyphs: list[GlyphDTO]
    training_glyphs: list[GlyphDTO]
    class_names: list[str] = Field(
        ..., description="Sorted union of all known class names."
    )
    binarization_method: BinarizationMethod = Field(
        ..., description="Method that produced the current glyph masks."
    )
    preset_training_count: int = Field(
        ...,
        description="How many training glyphs came from a built-in preset.",
    )
    uploaded_training_count: int = Field(
        ...,
        description="How many training glyphs came from an uploaded file.",
    )


# ---------------------------------------------------------------------------
# Request bodies — write paths
# ---------------------------------------------------------------------------


class ClassifyRequest(BaseModel):
    """POST /sessions/{id}/classify body."""

    k: int = Field(default=3, ge=1, description="Neighbour count; default 3. Ignored when backend='ssl_fusion'.")
    backend: Literal["knn", "ssl_fusion"] = Field(
        default="knn",
        description=(
            "'knn' (default): handcrafted-feature k-nearest-neighbours "
            "classifier. 'ssl_fusion': optional SSL+handcrafted fused "
            "logistic-regression classifier; requires the server to have "
            "the ssl extra installed and IC_SSL_CHECKPOINT configured."
        ),
    )


class RebinarizeRequest(BaseModel):
    """POST /sessions/{id}/binarization body.

    Switches the binarisation method and rebuilds every glyph mask from
    the session's retained page + bboxes. See :meth:`Session.rebinarize`.
    """

    method: BinarizationMethod = Field(
        ..., description="'global', 'otsu', or 'sauvola'."
    )


class UpdateGlyphRequest(BaseModel):
    """POST /sessions/{id}/glyphs/{gid} body — partial update."""

    class_name: str | None = None
    id_state_manual: bool | None = None
    # Move the glyph to another MOTHRA category (Text / Neumes / Staves).
    category: Literal["Text", "Neumes", "Staves"] | None = None


class GroupRequest(BaseModel):
    """POST /sessions/{id}/group body."""

    glyph_ids: list[str] = Field(..., min_length=1)
    class_name: str


class SplitRequest(BaseModel):
    """POST /sessions/{id}/glyphs/{gid}/split body.

    ``regions`` is a list of ``[ulx, uly, ncols, nrows]`` rectangles
    in **page coordinates** (the same frame the glyph's bbox uses),
    matching what the frontend draws on the page-image canvas.
    Tuples — rather than named-field objects — keep the wire shape
    compact and avoid a positional/keyword mismatch in JS clients
    that already think of bboxes as ``[x, y, w, h]`` arrays.
    """

    regions: list[tuple[int, int, int, int]] = Field(
        ...,
        min_length=1,
        description=(
            "One or more rectangles as [ulx, uly, ncols, nrows] in page "
            "coordinates. Each becomes one UNCLASSIFIED child glyph."
        ),
    )


class RenameClassRequest(BaseModel):
    """POST /sessions/{id}/classes/{name}/rename body."""

    new_name: str


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def glyph_to_dto(glyph: Glyph) -> GlyphDTO:
    """Domain :class:`Glyph` → wire :class:`GlyphDTO`."""
    return GlyphDTO(
        id=glyph.id,
        class_name=glyph.class_name,
        confidence=glyph.confidence,
        id_state_manual=glyph.id_state_manual,
        category=glyph.category,
        ulx=glyph.ulx,
        uly=glyph.uly,
        ncols=glyph.ncols,
        nrows=glyph.nrows,
        image_b64=glyph.to_base64_png(),
    )


def session_to_dto(session: Session) -> SessionDTO:
    """Domain :class:`Session` → wire :class:`SessionDTO`."""
    return SessionDTO(
        id=session.id,
        state=session.state,
        glyphs=[glyph_to_dto(g) for g in session.glyphs],
        training_glyphs=[glyph_to_dto(g) for g in session.training_glyphs],
        class_names=sorted(session.class_names),
        binarization_method=session.binarization_method,
        preset_training_count=sum(
            1
            for g in session.training_glyphs
            if g.id in session.preset_training_ids
        ),
        uploaded_training_count=sum(
            1
            for g in session.training_glyphs
            if g.id in session.uploaded_training_ids
        ),
    )


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Uniform error body used by every non-2xx response."""

    detail: str
    code: Literal[
        "not_found",
        "state_conflict",
        "validation_error",
        "deferred",
        "internal_error",
    ]
