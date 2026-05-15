"""Ingest a page image + bbox annotations into :class:`Glyph` objects.

The new pipeline's primary input is **one full page image** plus a
companion **bounding-box file** describing where each neume sits on
that page. We crop on the fly rather than asking the caller to
pre-slice the page into per-neume PNGs.

Two annotation formats are supported, both produced by the upstream
detector (MOTHRA / YOLO):

1. **MOTHRA JSON** (``*.json``) — pixel coordinates plus a stable
   per-annotation UUID. Structure:

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

2. **YOLO text** (``*.txt``) — one bbox per line, normalised to
   the image dimensions:

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

import json
import uuid
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image as PILImage

from ic_core.classifier import UNCLASSIFIED
from ic_core.glyph import Glyph
from ic_core.image import array_to_rle

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
    page_image: Path | str,
    annotations: Path | str,
    *,
    threshold: int = DEFAULT_THRESHOLD,
) -> list[Glyph]:
    """Crop a page into glyphs using a bbox annotation file.

    Dispatches on the annotation file's suffix:

    * ``.json`` → :func:`ingest_page_json` (MOTHRA format)
    * ``.txt`` → :func:`ingest_page_yolo` (YOLO format)

    Args:
        page_image: Path to the full-page image (any format PIL can
            open; typically PNG).
        annotations: Path to the bbox file.
        threshold: Foreground/background cutoff used when binarising
            each cropped region.

    Returns:
        One :class:`Glyph` per bounding box, in the order the
        annotation file lists them.

    Raises:
        ValueError: If the annotation file's suffix is unrecognised.
        FileNotFoundError: If either input file does not exist.
    """
    annotations_path = Path(annotations)
    suffix = annotations_path.suffix.lower()
    if suffix == ".json":
        return ingest_page_json(page_image, annotations_path, threshold=threshold)
    if suffix == ".txt":
        return ingest_page_yolo(page_image, annotations_path, threshold=threshold)
    raise ValueError(
        f"Unrecognised annotation format {suffix!r}; expected .json or .txt"
    )


def ingest_page_json(
    page_image: Path | str,
    json_path: Path | str,
    *,
    threshold: int = DEFAULT_THRESHOLD,
) -> list[Glyph]:
    """Crop using a MOTHRA JSON annotation file.

    The JSON's ``annotations[i].id`` becomes the glyph's UUID (with
    dashes stripped to match :class:`Glyph`'s 32-hex-char
    convention). This is what makes re-ingestion idempotent in id
    space.

    Args:
        page_image: Path to the page image.
        json_path: Path to the MOTHRA JSON file.
        threshold: Binarisation cutoff.

    Returns:
        One :class:`Glyph` per annotation.
    """
    json_path = Path(json_path)
    with json_path.open() as f:
        doc = json.load(f)

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
    page_image: Path | str,
    yolo_path: Path | str,
    *,
    threshold: int = DEFAULT_THRESHOLD,
) -> list[Glyph]:
    """Crop using a YOLO ``.txt`` annotation file.

    YOLO carries no stable ids, so each glyph receives a fresh UUID.

    Args:
        page_image: Path to the page image.
        yolo_path: Path to the YOLO bbox file.
        threshold: Binarisation cutoff.

    Returns:
        One :class:`Glyph` per non-empty, non-comment line.
    """
    yolo_path = Path(yolo_path)
    page = _load_page(page_image)
    img_h, img_w = page.shape

    glyphs: list[Glyph] = []
    for ulx, uly, width, height in _iter_yolo_bboxes(yolo_path, img_w, img_h):
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


def _load_page(page_image: Path | str) -> np.ndarray:
    """Load the page image once as an 8-bit greyscale ``numpy.ndarray``.

    Returns:
        Array of shape ``(height, width)`` and dtype ``uint8``.
        Foreground/background discrimination is deferred to crop
        time so the threshold can be configured per call without
        having to re-open the page.
    """
    page_image = Path(page_image)
    with PILImage.open(page_image) as im:
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
    yolo_path: Path,
    img_width: int,
    img_height: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(ulx, uly, width, height)`` in pixel coords for each YOLO line.

    The YOLO format normalises coordinates to ``[0, 1]`` over the
    image; we de-normalise once here using the page's pixel
    dimensions. The first token on each line is the class id, which
    we discard (see module docstring).
    """
    with yolo_path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Format: <class_id> <cx> <cy> <w> <h>, all floats except class_id.
            # We tolerate either 5 tokens (no confidence) or 6 tokens
            # (some YOLO variants append a detection confidence).
            if len(parts) < 5:
                raise ValueError(
                    f"Malformed YOLO line in {yolo_path.name!r}: {line!r}"
                )
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
