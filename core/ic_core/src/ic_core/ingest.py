"""Ingest a page image + bbox annotations into :class:`Glyph` objects.

The pipeline's primary input is **one full page image** plus a
companion **bounding-box document** describing where each neume sits
on that page. We crop on the fly rather than asking the caller to
pre-slice the page into per-neume PNGs.

Inputs are passed as **raw bytes**, not filesystem paths. The HTTP
layer above this hands us multipart upload payloads directly, and
tests read fixtures via :func:`Path.read_bytes`. Keeping ingest off
the filesystem means the API layer can never be tricked into
reading server-side files chosen by the client.

Two annotation formats are supported, both produced by the upstream
detector (MOTHRA / YOLO). The caller picks via the ``format``
argument — we no longer guess from a file suffix:

1. **MOTHRA JSON** (``format="json"``) — pixel coordinates plus a
   stable per-annotation UUID. Structure:

   .. code-block:: json

       {
         "imageName": "...",
         "imageWidth": 804,
         "imageHeight": 1135,
         "annotations": [
           {"id": "8cffd2b0-...", "classId": 1,
            "bbox": [ulx, uly, w, h], "timestamp": "..."}
         ]
       }

   The ``id`` is preserved as the resulting :class:`Glyph` UUID so
   that re-ingesting the same JSON produces the same glyph IDs
   (algorithm semantic #6: existing glyphs preserve their UUIDs
   across round-trips).

2. **YOLO text** (``format="yolo"``) — one bbox per line, normalised
   to the image dimensions:

   .. code-block:: text

       <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>

   YOLO files carry no stable id, so glyphs receive fresh UUIDs.

Why page + bboxes (and not pre-cropped PNGs)
--------------------------------------------

* **Page coordinates come for free.** The :class:`Glyph` ``ulx`` /
  ``uly`` get the bbox origin in page-pixel space, which is exactly
  what auto-grouping needs (migration plan gotcha #4). The
  pre-cropped-PNG alternative would have required a sidecar JSON
  per file to recover this.
* **One file pair, not hundreds.** Easier to manage, easier to
  diff, easier to send over the API.
* **Cropping logic lives in one place.** No question about whose
  PIL convention was used to slice the originals.

Class labels
------------

The upstream detector's class id (``classId`` in the JSON,
``class_id`` in the YOLO line) is **ignored** — we mark every
ingested glyph as :data:`UNCLASSIFIED`. The user (or the
classifier) assigns real labels through the API. Mapping the
detector classes onto neume classes is a separate concern not in
Phase 1 scope.
"""
from __future__ import annotations

import io
import json
import uuid
from typing import Iterator, Literal

import numpy as np
from PIL import Image as PILImage

from ic_core.classifier import UNCLASSIFIED
from ic_core.glyph import Glyph
from ic_core.image import array_to_rle

#: Discriminator for which annotation parser :func:`ingest_page` picks.
AnnotationFormat = Literal["json", "yolo"]

#: Pixel-intensity cutoff: values ≤ this become foreground (True).
#: 127 corresponds to "everything darker than mid-grey is ink",
#: which works on both pre-binarised neume crops and lightly
#: noisy ones. Override per-call with the ``threshold`` argument
#: to :func:`ingest_page` if a specific dataset needs it.
DEFAULT_THRESHOLD: int = 127


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def ingest_page(
    page_image: bytes,
    annotations: bytes,
    *,
    format: AnnotationFormat,
    threshold: int = DEFAULT_THRESHOLD,
) -> list[Glyph]:
    """Crop a page into glyphs using a bbox annotation document.

    Args:
        page_image: Raw bytes of the full-page image (any format
            PIL can open; typically PNG).
        annotations: Raw bytes of the bbox document.
        format: Which annotation parser to use — ``"json"`` for the
            MOTHRA JSON format, ``"yolo"`` for the YOLO ``.txt``
            format. Explicit because the bytes alone don't always
            disambiguate, and because letting callers (HTTP clients)
            choose a parser by guessing file extensions is the same
            anti-pattern that motivated this byte-based API.
        threshold: Foreground/background cutoff used when binarising
            each cropped region.

    Returns:
        One :class:`Glyph` per bounding box, in the order the
        annotation document lists them.

    Raises:
        ValueError: If ``format`` is not one of ``"json"`` /
            ``"yolo"``.
    """
    if format == "json":
        return ingest_page_json(page_image, annotations, threshold=threshold)
    if format == "yolo":
        return ingest_page_yolo(page_image, annotations, threshold=threshold)
    raise ValueError(
        f"Unrecognised annotation format {format!r}; expected 'json' or 'yolo'"
    )


def ingest_page_json(
    page_image: bytes,
    annotations_json: bytes,
    *,
    threshold: int = DEFAULT_THRESHOLD,
) -> list[Glyph]:
    """Crop using a MOTHRA JSON annotation document.

    The JSON's ``annotations[i].id`` becomes the glyph's UUID (with
    dashes stripped to match :class:`Glyph`'s 32-hex-char
    convention). This is what makes re-ingestion idempotent in id
    space.

    Args:
        page_image: Raw bytes of the page image.
        annotations_json: Raw bytes of the MOTHRA JSON document.
        threshold: Binarisation cutoff.

    Returns:
        One :class:`Glyph` per annotation.
    """
    doc = json.loads(annotations_json)
    annotations = doc.get("annotations", [])

    # Open the page once; reuse the array across crops. Much cheaper
    # than re-opening per glyph (136 glyphs × tens of KB of decode
    # work each adds up).
    page = _load_page(page_image)

    return [
        _crop_to_glyph(
            page,
            ulx=int(round(a["bbox"][0])),
            uly=int(round(a["bbox"][1])),
            width=int(round(a["bbox"][2])),
            height=int(round(a["bbox"][3])),
            threshold=threshold,
            glyph_id=_normalise_uuid(a["id"]),
        )
        for a in annotations
    ]


def ingest_page_yolo(
    page_image: bytes,
    annotations_yolo: bytes,
    *,
    threshold: int = DEFAULT_THRESHOLD,
) -> list[Glyph]:
    """Crop using a YOLO ``.txt`` annotation document.

    YOLO carries no stable ids, so each glyph receives a fresh UUID.

    Args:
        page_image: Raw bytes of the page image.
        annotations_yolo: Raw bytes of the YOLO ``.txt`` document.
        threshold: Binarisation cutoff.

    Returns:
        One :class:`Glyph` per non-empty, non-comment line.
    """
    page = _load_page(page_image)
    img_h, img_w = page.shape

    glyphs: list[Glyph] = []
    for ulx, uly, width, height in _iter_yolo_bboxes(annotations_yolo, img_w, img_h):
        glyphs.append(
            _crop_to_glyph(
                page,
                ulx=ulx,
                uly=uly,
                width=width,
                height=height,
                threshold=threshold,
                glyph_id=None,  # fresh UUID — YOLO has none to inherit
            )
        )
    return glyphs


# ---------------------------------------------------------------------------
# Internals — page loading, cropping, format parsing
# ---------------------------------------------------------------------------


def _load_page(page_image: bytes) -> np.ndarray:
    """Load the page image once as an 8-bit greyscale ``numpy.ndarray``.

    Returns:
        Array of shape ``(height, width)`` and dtype ``uint8``.
        Foreground/background discrimination is deferred to crop
        time so the threshold can be configured per call without
        having to re-open the page.
    """
    with PILImage.open(io.BytesIO(page_image)) as im:
        grey = im.convert("L")
        return np.asarray(grey)


def _crop_to_glyph(
    page: np.ndarray,
    *,
    ulx: int,
    uly: int,
    width: int,
    height: int,
    threshold: int,
    glyph_id: str | None,
) -> Glyph:
    """Slice ``page[uly:uly+h, ulx:ulx+w]``, binarise, wrap as a Glyph.

    Out-of-bounds bboxes are clamped to the page rectangle — a bbox
    that runs a pixel past the edge stays as a glyph (the upstream
    detector occasionally rounds outward), but its actual footprint
    is whatever fell inside the page.
    """
    img_h, img_w = page.shape

    # Clamp to the page rectangle. We keep the *declared* ulx/uly so
    # downstream auto-grouping still places the glyph at the
    # detector's reported origin, even if the clamped crop is
    # slightly smaller than requested.
    x0 = max(0, ulx)
    y0 = max(0, uly)
    x1 = min(img_w, ulx + width)
    y1 = min(img_h, uly + height)

    if x1 <= x0 or y1 <= y0:
        # Pathological: bbox falls entirely outside the page.
        # Return a 1×1 blank glyph rather than crashing — the user
        # can delete it in the UI.
        mask = np.zeros((1, 1), dtype=bool)
        nrows = ncols = 1
    else:
        crop = page[y0:y1, x0:x1]
        mask = crop <= threshold
        nrows, ncols = mask.shape

    return Glyph.new(
        id=glyph_id,
        class_name=UNCLASSIFIED,
        image_rle=array_to_rle(mask),
        ncols=int(ncols),
        nrows=int(nrows),
        ulx=int(ulx),
        uly=int(uly),
        id_state_manual=False,
        confidence=0.0,
        is_training=False,
    )


def _iter_yolo_bboxes(
    yolo_bytes: bytes,
    img_width: int,
    img_height: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(ulx, uly, width, height)`` in pixel coords for each YOLO line.

    The YOLO format normalises coordinates to ``[0, 1]`` over the
    image; we de-normalise once here using the page's pixel
    dimensions. The first token on each line is the class id, which
    we discard (see module docstring).
    """
    text = yolo_bytes.decode("utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Format: <class_id> <cx> <cy> <w> <h>, all floats except class_id.
        # We tolerate either 5 tokens (no confidence) or 6 tokens
        # (some YOLO variants append a detection confidence).
        if len(parts) < 5:
            raise ValueError(f"Malformed YOLO line: {line!r}")
        _, cx, cy, w, h = parts[:5]
        cx_f, cy_f, w_f, h_f = float(cx), float(cy), float(w), float(h)

        # Centre-normalised → top-left pixel coords.
        ulx = int(round((cx_f - w_f / 2.0) * img_width))
        uly = int(round((cy_f - h_f / 2.0) * img_height))
        width = int(round(w_f * img_width))
        height = int(round(h_f * img_height))
        yield ulx, uly, width, height


def _normalise_uuid(raw: str) -> str:
    """Convert a UUID string into the 32-hex-char form used by :class:`Glyph`.

    Accepts both dashed (``8cffd2b0-134e-4018-b6d4-99f8fcc36a37``)
    and undashed input. Invalid input falls back to a fresh UUID
    rather than raising — the ingest path should be tolerant of
    occasional detector quirks.
    """
    try:
        return uuid.UUID(raw).hex
    except (ValueError, AttributeError, TypeError):
        return uuid.uuid4().hex
