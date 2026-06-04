"""Glyph dataclass.

Replaces ``intermediary/gamera_glyph.py``. Defines the in-memory record for a
single classified (or unclassified) glyph: bounding box, binary image data,
class name, confidence, ``id_state_manual`` flag, and stable UUID. UUIDs are
generated on construction and preserved across round-trips; new glyphs from
manual group/split operations receive fresh ones.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ic_core.image import array_to_png_base64, rle_to_array

# ---------------------------------------------------------------------------
# MOTHRA detector categories
# ---------------------------------------------------------------------------
# The upstream MOTHRA detector tags every bbox with a coarse ``classId``
# (1/2/3) describing *what kind* of mark it is. IC only classifies neumes;
# Text and Staves are carried through so the UI can group and hide them, but
# they never enter the kNN classifier. These string constants are the
# canonical category labels used end-to-end (ingest → state → API → UI).
CATEGORY_TEXT = "Text"
CATEGORY_NEUMES = "Neumes"
CATEGORY_STAVES = "Staves"

#: All valid categories, in display order (Text, Neumes, Staves).
CATEGORIES: tuple[str, ...] = (CATEGORY_TEXT, CATEGORY_NEUMES, CATEGORY_STAVES)


@dataclass(slots=True, frozen=True)
class Glyph:
    id: str
    class_name: str
    image_rle: str
    ncols: int
    nrows: int
    ulx: int
    uly: int
    id_state_manual: bool
    confidence: float
    # Coarse MOTHRA detector category (Text / Neumes / Staves). Only
    # ``CATEGORY_NEUMES`` glyphs are classified; the others pass through
    # untouched. Defaults to Neumes so glyphs born outside the JSON ingest
    # path (YOLO, manual grouping) land in the category IC actually works on.
    category: str = CATEGORY_NEUMES
    is_training: bool = False
    # Optional per-glyph feature cache. Populated by the classifier
    # (and read by the XML exporter) so the "full re-train every
    # round" loop doesn't recompute features for stable glyphs.
    # ``feature_version`` records which ``FEATURE_VERSION`` produced
    # the vector so a version bump invalidates stale caches; the
    # vector itself is excluded from ``__eq__`` / ``__hash__`` /
    # ``repr`` because ndarray equality doesn't fit the dataclass
    # default and printing a 29-element array adds noise to logs.
    feature_vector: np.ndarray | None = field(
        default=None, compare=False, repr=False
    )
    feature_version: str | None = field(default=None, compare=False, repr=False)

    @classmethod
    def new(
        cls,
        *,
        class_name: str,
        image_rle: str,
        ncols: int,
        nrows: int,
        ulx: int,
        uly: int,
        id_state_manual: bool,
        confidence: float,
        category: str = CATEGORY_NEUMES,
        is_training: bool = False,
        id: str | None = None,
        feature_vector: np.ndarray | None = None,
        feature_version: str | None = None,
    ) -> "Glyph":
        return cls(
            id=id if id is not None else uuid.uuid4().hex,
            class_name=class_name,
            image_rle=image_rle,
            ncols=ncols,
            nrows=nrows,
            ulx=ulx,
            uly=uly,
            id_state_manual=id_state_manual,
            confidence=confidence,
            category=category,
            is_training=is_training,
            feature_vector=feature_vector,
            feature_version=feature_version,
        )

    def is_manual_id(self) -> bool:
        return self.id_state_manual

    def is_id(self, other: str) -> bool:
        return self.id == other

    def classify_manual(self, class_name: str) -> "Glyph":
        return replace(
            self,
            class_name=str(class_name),
            confidence=1.0,
            id_state_manual=True,
        )

    def classify_automatic(self, class_name: str, confidence: float) -> "Glyph":
        return replace(
            self,
            class_name=str(class_name),
            confidence=confidence,
            id_state_manual=False,
        )

    def to_array(self) -> np.ndarray:
        return rle_to_array(self.image_rle, self.ncols, self.nrows)

    def to_base64_png(self) -> str:
        return array_to_png_base64(self.to_array())

    def to_dict(self) -> dict[str, Any]:
        # Built explicitly (rather than via ``asdict``) so the ndarray-valued
        # ``feature_vector`` cache isn't deep-copied just to be discarded.
        return {
            "id": self.id,
            "class_name": self.class_name,
            "image": self.image_rle,
            "ncols": self.ncols,
            "nrows": self.nrows,
            "ulx": self.ulx,
            "uly": self.uly,
            "id_state_manual": self.id_state_manual,
            "confidence": self.confidence,
            "category": self.category,
            "is_training": self.is_training,
            "image_b64": self.to_base64_png(),
        }
