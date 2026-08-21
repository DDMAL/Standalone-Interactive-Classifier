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

Every ingested glyph starts as :data:`UNCLASSIFIED`; the user (or
the classifier) assigns real neume labels through the API.

The detector's class id *is* preserved as a coarse **category**:
``classId`` 1/2/3 in the JSON map to ``Text`` / ``Neumes`` /
``Staves`` (see :data:`_MOTHRA_CLASS_TO_CATEGORY`). Only ``Neumes``
glyphs are classified — Text and Staves are carried through so the
UI can group and hide them. The YOLO ``class_id`` is still ignored
(those glyphs default to ``Neumes``).
"""
from __future__ import annotations

import io
import json
import uuid
from typing import Iterator, Literal

import numpy as np
from PIL import Image as PILImage
from skimage.filters import threshold_otsu, threshold_sauvola

from ic_core.classifier import UNCLASSIFIED
from ic_core.glyph import (
    CATEGORY_NEUMES,
    CATEGORY_STAVES,
    CATEGORY_TEXT,
    Glyph,
)
from ic_core.image import array_to_rle

#: MOTHRA ``classId`` → IC category. The detector tags each bbox 1/2/3;
#: we carry that through so the UI can group Text/Neumes/Staves. An
#: unrecognised id falls back to Neumes so it still surfaces for review.
_MOTHRA_CLASS_TO_CATEGORY: dict[int, str] = {
    1: CATEGORY_TEXT,
    2: CATEGORY_NEUMES,
    3: CATEGORY_STAVES,
}

#: Discriminator for which annotation parser :func:`ingest_page` picks.
AnnotationFormat = Literal["json", "yolo"]

#: Which binarisation algorithm turns the greyscale page into a
#: foreground/background mask. ``"global"`` is the historical fixed
#: cutoff; ``"otsu"`` derives one global cutoff from the page
#: histogram; ``"sauvola"`` is a local adaptive threshold that
#: recovers faint ink (e.g. staff lines) on unevenly-lit parchment.
#: See :func:`binarize_array`.
BinarizationMethod = Literal["global", "otsu", "sauvola"]

#: Default binarisation method. Kept at ``"global"`` so existing
#: callers (scripts, tests) and re-ingests are byte-for-byte
#: unchanged; the frontend opts into ``"sauvola"`` explicitly.
DEFAULT_METHOD: BinarizationMethod = "global"

#: Pixel-intensity cutoff for ``method="global"``: values ≤ this
#: become foreground (True). 127 corresponds to "everything darker
#: than mid-grey is ink", which works on both pre-binarised neume
#: crops and lightly noisy ones. Override per-call with the
#: ``threshold`` argument to :func:`ingest_page` if a specific
#: dataset needs it.
DEFAULT_THRESHOLD: int = 127

#: Defaults for ``method="sauvola"``. The window should be a few×
#: the stroke width — large enough not to eat the interior of thick
#: neumes, small enough not to revert toward a global threshold. ``k``
#: weights the local std-dev: higher is more conservative (less ink).
#: 25 / 0.2 won an A/B on the ``core/data/test`` folios.
SAUVOLA_WINDOW_SIZE: int = 25
SAUVOLA_K: float = 0.2

#: Working-set budget for one Sauvola strip, in bytes.
#:
#: ``threshold_sauvola`` holds roughly six page-sized float64 arrays at once
#: (a float64 copy of the page, two float64 integral images, then the local
#: mean, mean-of-squares and std-dev), so its peak cost is ~48 bytes per
#: *pixel* — over 1.5 GB on a 32 MP folio, which OOM-kills a 1 GiB container.
#: :func:`_sauvola_mask` splits the page into horizontal strips sized to this
#: budget instead, making peak memory a function of the budget rather than of
#: the page. 96 MiB keeps the whole binarisation comfortably inside a few
#: hundred MB for any page size, at the cost of recomputing a ~26-row halo
#: per strip (a few percent more arithmetic).
SAUVOLA_STRIP_BUDGET_BYTES: int = 96 * 1024 * 1024

#: How many bytes per pixel per strip row to assume when sizing a strip —
#: the ~6 float64 page-sized arrays described above.
_SAUVOLA_BYTES_PER_PIXEL: int = 8 * 6


# ---------------------------------------------------------------------------
# Memory-bounded threshold helpers
# ---------------------------------------------------------------------------
#
# Both helpers exist purely to keep peak memory off the page-size curve.
# skimage's thresholding routines are written for images that fit in RAM
# several times over; a 50 MP folio does not, and IC runs in a 1 GiB
# container. Neither helper changes the threshold that comes out — the
# equality is covered by tests in ``core/tests/test_ingest.py``.


def _otsu_threshold(page: np.ndarray) -> float:
    """Otsu's cutoff, without materialising a float copy of the page.

    ``threshold_otsu(image)`` routes the page through ``np.histogram``,
    which promotes the whole array to float64 — ~400 MB on a 50 MP folio,
    for a statistic that only needs 256 bin counts. For the 8-bit pages
    every ingest path produces (:func:`_load_page` returns ``uint8``),
    ``np.bincount`` gets those counts with no large temporary, and skimage
    accepts a precomputed histogram via ``hist=``, so the arithmetic that
    picks the threshold stays skimage's.

    Falls back to passing the array when the dtype isn't ``uint8``, so
    callers outside the ingest paths keep working.
    """
    if page.dtype != np.uint8:
        return float(threshold_otsu(page))
    counts = np.bincount(page.reshape(-1), minlength=256)
    # Bin centres are the intensities themselves, which is what
    # threshold_otsu assumes when handed (counts, centres).
    centres = np.arange(256)
    return float(threshold_otsu(hist=(counts, centres)))


def _sauvola_mask(
    page: np.ndarray,
    *,
    window_size: int,
    k: float,
    budget_bytes: int = SAUVOLA_STRIP_BUDGET_BYTES,
) -> np.ndarray:
    """Sauvola foreground mask computed in horizontal strips.

    Sauvola is a *local* threshold: the value at a pixel depends only on
    the ``window_size`` neighbourhood around it. So a strip that carries a
    halo of at least ``window_size // 2 + 1`` rows above and below sees
    exactly the neighbourhood the full-page pass would have seen, and its
    interior rows come out bit-identical. Strips at the top and bottom of
    the page include the real page edge, so skimage's ``reflect`` padding
    applies there just as it would full-page. ``r`` (the dynamic-range
    term) is derived from the *dtype*, not the data, so it too is constant
    across strips.

    The result is the same mask ``threshold_sauvola`` over the whole page
    would produce, with peak memory set by ``budget_bytes`` instead of by
    the page area.
    """
    height, width = page.shape
    # +1 covers skimage's asymmetric pad (``w // 2 + 1`` leading rows in
    # ``_mean_std``); +1 again is cheap insurance against an off-by-one in
    # a future skimage. test_sauvola_strips_match_whole_page pins the
    # equality this margin protects.
    halo = window_size // 2 + 2
    rows_per_strip = max(1, budget_bytes // (width * _SAUVOLA_BYTES_PER_PIXEL))

    if rows_per_strip + 2 * halo >= height:
        # Page already fits the budget — take the simple path so small
        # pages pay nothing for the strip bookkeeping.
        return page < threshold_sauvola(page, window_size=window_size, k=k)

    mask = np.empty((height, width), dtype=bool)
    for y0 in range(0, height, rows_per_strip):
        y1 = min(height, y0 + rows_per_strip)
        # Widen by the halo, clipped to the page.
        h0, h1 = max(0, y0 - halo), min(height, y1 + halo)
        block = page[h0:h1]
        thresh = threshold_sauvola(block, window_size=window_size, k=k)
        # Keep only the rows this strip owns; the halo rows exist solely to
        # give the window real pixels to look at.
        lo, hi = y0 - h0, y1 - h0
        mask[y0:y1] = block[lo:hi] < thresh[lo:hi]
    return mask


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def binarize_array(
    page: np.ndarray,
    *,
    method: BinarizationMethod = DEFAULT_METHOD,
    threshold: int = DEFAULT_THRESHOLD,
    window_size: int = SAUVOLA_WINDOW_SIZE,
    k: float = SAUVOLA_K,
) -> np.ndarray:
    """Binarise a greyscale page array to a foreground mask.

    The single source of truth for foreground/background discrimination;
    both :func:`binarize_page` and the per-glyph crops in
    :func:`ingest_page` go through here so a session's per-glyph masks
    and its full-page grouping mask always agree.

    Args:
        page: 8-bit greyscale page, shape ``(height, width)``.
        method: ``"global"`` (pixels ≤ ``threshold``), ``"otsu"`` (one
            histogram-derived cutoff), or ``"sauvola"`` (local adaptive
            threshold over a ``window_size`` window weighted by ``k``).
        threshold: Cutoff for ``method="global"`` (also the fallback for
            ``"otsu"`` on a uniform page).
        window_size: Sauvola window edge in pixels (odd).
        k: Sauvola std-dev weight.

    Returns:
        Boolean array of shape ``(height, width)``; ``True`` where the
        page has foreground ink.

    Note:
        Sauvola's per-pixel threshold depends on the surrounding window,
        so it does *not* commute with cropping — it must be computed on
        the whole page and then sliced, never per crop. ``"global"`` and
        ``"otsu"`` do commute, so all three are handled whole-page here.

    Note:
        Peak memory is bounded and roughly independent of page area:
        ``"otsu"`` goes through :func:`_otsu_threshold` (a 256-bin
        histogram, no float copy of the page) and ``"sauvola"`` through
        :func:`_sauvola_mask` (horizontal strips with a window-sized
        halo). Both return exactly what the naive whole-page skimage call
        returns. This matters because a full-page Sauvola on a 32 MP folio
        peaks over 1.5 GB, which OOM-kills IC's 1 GiB container mid-request
        and — on the in-memory session store — takes every live session
        with it.
    """
    if method == "global":
        return page <= threshold
    if method == "otsu":
        # threshold_otsu raises on a single-valued image; a blank/uniform
        # crop has no bimodal histogram, so fall back to the global cutoff.
        if page.min() == page.max():
            return page <= threshold
        return page <= _otsu_threshold(page)
    if method == "sauvola":
        return _sauvola_mask(page, window_size=window_size, k=k)
    raise ValueError(
        f"Unrecognised binarization method {method!r}; "
        "expected 'global', 'otsu', or 'sauvola'"
    )


def binarize_page(
    page_image: bytes,
    *,
    method: BinarizationMethod = DEFAULT_METHOD,
    threshold: int = DEFAULT_THRESHOLD,
    window_size: int = SAUVOLA_WINDOW_SIZE,
    k: float = SAUVOLA_K,
) -> np.ndarray:
    """Decode a page image and binarise it to a full-page foreground mask.

    Same method/parameter convention as :func:`ingest_page`; this is the
    full-page mask manual grouping relies on to recover ink falling
    *between* child glyph bboxes (never copied into any crop). It MUST be
    produced with the same ``method`` as the page's glyphs were, or the
    grouping mask and the per-glyph masks disagree.

    Returns:
        Boolean array of shape ``(height, width)``; ``True`` where the
        page has foreground ink.
    """
    return binarize_array(
        _load_page(page_image),
        method=method,
        threshold=threshold,
        window_size=window_size,
        k=k,
    )


def ingest_page(
    page_image: bytes,
    annotations: bytes,
    *,
    format: AnnotationFormat,
    method: BinarizationMethod = DEFAULT_METHOD,
    threshold: int = DEFAULT_THRESHOLD,
    window_size: int = SAUVOLA_WINDOW_SIZE,
    k: float = SAUVOLA_K,
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
        method: Binarisation algorithm — see :func:`binarize_array`.
        threshold: Foreground/background cutoff for ``method="global"``.
        window_size: Sauvola window edge in pixels.
        k: Sauvola std-dev weight.

    Returns:
        One :class:`Glyph` per bounding box, in the order the
        annotation document lists them.

    Raises:
        ValueError: If ``format`` is not one of ``"json"`` /
            ``"yolo"``.
    """
    bin_kwargs = dict(method=method, threshold=threshold, window_size=window_size, k=k)
    if format == "json":
        return ingest_page_json(page_image, annotations, **bin_kwargs)
    if format == "yolo":
        return ingest_page_yolo(page_image, annotations, **bin_kwargs)
    raise ValueError(
        f"Unrecognised annotation format {format!r}; expected 'json' or 'yolo'"
    )


def ingest_page_json(
    page_image: bytes,
    annotations_json: bytes,
    *,
    method: BinarizationMethod = DEFAULT_METHOD,
    threshold: int = DEFAULT_THRESHOLD,
    window_size: int = SAUVOLA_WINDOW_SIZE,
    k: float = SAUVOLA_K,
) -> list[Glyph]:
    """Crop using a MOTHRA JSON annotation document.

    The JSON's ``annotations[i].id`` becomes the glyph's UUID (with
    dashes stripped to match :class:`Glyph`'s 32-hex-char
    convention). This is what makes re-ingestion idempotent in id
    space.

    The document may be either a single page object (``{"imageName":
    ..., "annotations": [...]}``) or a one-element list wrapping it
    (``[{...}]``) — MOTHRA emits both. The list form is unwrapped to
    its single page; a multi-page list is rejected, since this ingest
    path binarises one ``page_image``.

    Args:
        page_image: Raw bytes of the page image.
        annotations_json: Raw bytes of the MOTHRA JSON document.
        method: Binarisation algorithm — see :func:`binarize_array`.
        threshold: Cutoff for ``method="global"``.
        window_size: Sauvola window edge in pixels.
        k: Sauvola std-dev weight.

    Returns:
        One :class:`Glyph` per annotation.

    Raises:
        ValueError: If the document is a list with anything other than
            exactly one page.
    """
    doc = _unwrap_page(json.loads(annotations_json))
    annotations = doc.get("annotations", [])

    # Binarise the whole page once, then slice each glyph out of the
    # resulting mask. Opening the page per glyph would be wasteful, and
    # binarising per crop would break Sauvola (its window must see the
    # real page, not a crop's mirror-padded edge).
    mask = binarize_array(
        _load_page(page_image),
        method=method,
        threshold=threshold,
        window_size=window_size,
        k=k,
    )

    return [
        _crop_to_glyph(
            mask,
            ulx=int(round(a["bbox"][0])),
            uly=int(round(a["bbox"][1])),
            width=int(round(a["bbox"][2])),
            height=int(round(a["bbox"][3])),
            glyph_id=_normalise_uuid(a["id"]),
            category=_MOTHRA_CLASS_TO_CATEGORY.get(a.get("classId"), CATEGORY_NEUMES),
        )
        for a in annotations
    ]


def ingest_page_yolo(
    page_image: bytes,
    annotations_yolo: bytes,
    *,
    method: BinarizationMethod = DEFAULT_METHOD,
    threshold: int = DEFAULT_THRESHOLD,
    window_size: int = SAUVOLA_WINDOW_SIZE,
    k: float = SAUVOLA_K,
) -> list[Glyph]:
    """Crop using a YOLO ``.txt`` annotation document.

    YOLO carries no stable ids, so each glyph receives a fresh UUID.

    Args:
        page_image: Raw bytes of the page image.
        annotations_yolo: Raw bytes of the YOLO ``.txt`` document.
        method: Binarisation algorithm — see :func:`binarize_array`.
        threshold: Cutoff for ``method="global"``.
        window_size: Sauvola window edge in pixels.
        k: Sauvola std-dev weight.

    Returns:
        One :class:`Glyph` per non-empty, non-comment line.
    """
    # Whole-page binarise once (see ingest_page_json for why), then slice.
    mask = binarize_array(
        _load_page(page_image),
        method=method,
        threshold=threshold,
        window_size=window_size,
        k=k,
    )
    img_h, img_w = mask.shape

    glyphs: list[Glyph] = []
    for _class, ulx, uly, width, height in _iter_yolo_bboxes(annotations_yolo, img_w, img_h):
        glyphs.append(
            _crop_to_glyph(
                mask,
                ulx=ulx,
                uly=uly,
                width=width,
                height=height,
                glyph_id=None,  # fresh UUID — YOLO has none to inherit
                category=_MOTHRA_CLASS_TO_CATEGORY.get(int(_class)+1, CATEGORY_NEUMES),
            )
        )
    return glyphs


# ---------------------------------------------------------------------------
# Internals — page loading, cropping, format parsing
# ---------------------------------------------------------------------------


def _unwrap_page(doc: object) -> dict:
    """Normalise a MOTHRA document to a single page dict.

    Accepts either a page dict or a one-element list wrapping one.
    A multi-page list can't be ingested against a single page image,
    so it's rejected rather than silently dropping pages.
    """
    if isinstance(doc, list):
        if len(doc) != 1:
            raise ValueError(
                f"Expected a single-page JSON list, got {len(doc)} pages; "
                "this ingest path binarises one page image at a time."
            )
        doc = doc[0]
    if not isinstance(doc, dict):
        raise ValueError(
            f"Unrecognised MOTHRA JSON: expected an object or one-element "
            f"list, got {type(doc).__name__}."
        )
    return doc


def _load_page(page_image: bytes) -> np.ndarray:
    """Load the page image once as an 8-bit greyscale ``numpy.ndarray``.

    Returns:
        Array of shape ``(height, width)`` and dtype ``uint8``.
        Foreground/background discrimination is left to
        :func:`binarize_array`, which runs once over the whole page.
    """
    with PILImage.open(io.BytesIO(page_image)) as im:
        grey = im.convert("L")
        return np.asarray(grey)


def _crop_to_glyph(
    page_mask: np.ndarray,
    *,
    ulx: int,
    uly: int,
    width: int,
    height: int,
    glyph_id: str | None,
    category: str = CATEGORY_NEUMES,
) -> Glyph:
    """Slice ``page_mask[uly:uly+h, ulx:ulx+w]`` and wrap it as a Glyph.

    ``page_mask`` is the whole-page boolean foreground mask (already
    binarised by :func:`binarize_array`); this just slices it. Binarising
    whole-page-then-slice — rather than per-crop — is what keeps Sauvola
    correct (its window sees the real page) and keeps every glyph mask
    consistent with the full-page grouping mask.

    Out-of-bounds bboxes are clamped to the page rectangle — a bbox
    that runs a pixel past the edge stays as a glyph (the upstream
    detector occasionally rounds outward), but its actual footprint
    is whatever fell inside the page.
    """
    img_h, img_w = page_mask.shape

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
        mask = page_mask[y0:y1, x0:x1]
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
        category=category,
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
        _class, cx, cy, w, h = parts[:5]
        cx_f, cy_f, w_f, h_f = float(cx), float(cy), float(w), float(h)

        # Centre-normalised → top-left pixel coords.
        ulx = int(round((cx_f - w_f / 2.0) * img_width))
        uly = int(round((cy_f - h_f / 2.0) * img_height))
        width = int(round(w_f * img_width))
        height = int(round(h_f * img_height))
        yield _class, ulx, uly, width, height


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
