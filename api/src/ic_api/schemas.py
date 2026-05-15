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

    # Page-coordinate frame inherited from the bbox annotation file.
    ulx: int
    uly: int
    ncols: int
    nrows: int

    # Base64-encoded PNG preview for the frontend `<img>` tag. The
    # raw RLE is not sent — the frontend doesn't decode it.
    image_b64: str = Field(..., description="Base64 PNG, ASCII.")


class SessionDTO(BaseModel):
    """JSON shape of an entire session."""

    id: str
    state: ClassifierState
    glyphs: list[GlyphDTO]
    training_glyphs: list[GlyphDTO]
    class_names: list[str] = Field(
        ..., description="Sorted union of all known class names."
    )


# ---------------------------------------------------------------------------
# Request bodies — write paths
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """POST /sessions body."""

    page_image: str = Field(..., description="Server-side path to the page image.")
    annotations: str = Field(
        ...,
        description="Server-side path to a MOTHRA .json or YOLO .txt bbox file.",
    )
    class_names: list[str] | None = Field(
        default=None,
        description="Optional list of class names to seed the autocomplete.",
    )


class ClassifyRequest(BaseModel):
    """POST /sessions/{id}/classify body."""

    k: int = Field(default=1, ge=1, description="Neighbour count; default 1.")


class UpdateGlyphRequest(BaseModel):
    """POST /sessions/{id}/glyphs/{gid} body — partial update."""

    class_name: str | None = None
    id_state_manual: bool | None = None


class GroupRequest(BaseModel):
    """POST /sessions/{id}/group body."""

    glyph_ids: list[str] = Field(..., min_length=1)
    class_name: str


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
