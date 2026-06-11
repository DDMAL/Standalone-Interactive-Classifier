"""Spatial grouping.

Phase-1 replacement for ``group_and_correct`` and Gamera's grouping
primitives from
``../Rodan-lite/backend/django/code/jobs/interactive_classifier/interactive_classifier.py``.

Two flavours of grouping exist in the legacy system:

1. **Manual group** — the user selects 2+ glyphs and assigns a class
   name. The backend bitwise-ORs the binary masks (Gamera's
   ``gamera.plugins.image_utilities.union_images``) into a single new
   glyph whose bounding box encompasses every input. The resulting
   glyph is marked ``id_state_manual=True, confidence=1.0`` and joins
   the training pool *immediately* (algorithm semantic #7 in
   ``docs/migration_plan.md``).

2. **Auto-group** — Gamera builds a spatial-adjacency graph over the
   working glyphs using either ``ShapedGroupingFunction`` (per-pixel
   distance via ``distance_transform_edt``) or
   ``BoundingBoxGroupingFunction`` (bounding-box overlap/distance),
   then groups candidates up to ``max_parts_per_group`` and
   reclassifies the merged shapes.

Phase 1 scope:

* **Manual group is implemented here.** It is foundational (the
  grouped glyph is immediately training data) and works without any
  page-coordinate metadata beyond what every :class:`Glyph` already
  carries (``ulx``, ``uly``, ``ncols``, ``nrows``).

* **Auto-group is deferred** pending the ingestion-format decision
  flagged in ``docs/migration_plan.md`` §"Risks and gotchas" (4):
  with per-neume cropped input, glyphs no longer share a single page
  coordinate frame, so spatial auto-grouping only works if a sidecar
  carries per-crop ``(page_id, x, y, w, h)``. The stub functions
  below preserve the public API surface so callers can wire UI
  affordances without conditional imports, and they raise a clear
  :class:`NotImplementedError` if invoked before the design call is
  made.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ic_core.glyph import Glyph
from ic_core.image import array_to_rle

# ---------------------------------------------------------------------------
# Manual group
# ---------------------------------------------------------------------------


def manual_group(
    glyphs: Sequence[Glyph],
    class_name: str,
    *,
    page_mask: np.ndarray | None = None,
) -> Glyph:
    """Bitwise-OR a set of glyphs into a single new manual glyph.

    This is the Phase-1 replacement for
    ``gamera.plugins.image_utilities.union_images``. Each input
    glyph's binary mask is painted into a shared canvas spanning the
    union of all inputs' bounding boxes; the result becomes a new
    :class:`Glyph` with:

    * a **fresh UUID** — newly created glyphs always get a new id
      (algorithm semantic #6 in ``docs/migration_plan.md``);
    * ``id_state_manual = True`` and ``confidence = 1.0`` — the
      grouped glyph is *immediately* training data, not a candidate
      for the next auto-classify round (semantic #7);
    * a bounding box ``(new_ulx, new_uly, new_ncols, new_nrows)``
      that encompasses every input.

    The page-coordinate frame is assumed to be shared across the
    inputs. With per-neume cropped input that lacks page coordinates,
    every glyph defaults to ``ulx=uly=0`` and the union collapses to
    the largest input's footprint with all masks stacked at the
    origin — still correct, still useful as a training example, just
    not spatially meaningful.

    **Gap pixels.** When the children's tight bboxes are separated by
    a gap, ink in that gap was never captured by any child at ingest
    time, so OR'ing the children's masks alone leaves the gap blank.
    Pass ``page_mask`` (the full-page binarised foreground, in the
    same coordinate frame as the glyphs' ``ulx`` / ``uly``) and the
    gap is filled from the page. Pixels *inside* any child's bbox
    still come from the children — this preserves prior manual edits
    like an earlier split that explicitly dropped some pixels — so
    only pixels in the between-bboxes gap are recovered.

    Args:
        glyphs: Two or more :class:`Glyph` objects (the legacy UX
            requires ≥ 2; we accept ≥ 1 because the math works fine
            for a single glyph — a one-element call is just a
            relabel-as-manual operation).
        class_name: The class label to assign to the grouped glyph.
        page_mask: Optional full-page binarised foreground mask. When
            provided, foreground pixels falling in the gap between
            child bboxes are pulled in from the page. When ``None``,
            falls back to the children-only OR (gap stays empty).

    Returns:
        A new :class:`Glyph` with manual state, full confidence, and
        the union mask.

    Raises:
        ValueError: If ``glyphs`` is empty.
    """
    if not glyphs:
        raise ValueError("manual_group requires at least one glyph")

    # 1. Compute the bounding box that encompasses every input.
    #    All four corners are taken in the inputs' shared coordinate
    #    frame; we use lower-right corner = ulx + ncols (exclusive)
    #    to make slicing arithmetic line up cleanly.
    new_ulx = min(g.ulx for g in glyphs)
    new_uly = min(g.uly for g in glyphs)
    new_lrx = max(g.ulx + g.ncols for g in glyphs)
    new_lry = max(g.uly + g.nrows for g in glyphs)
    new_ncols = new_lrx - new_ulx
    new_nrows = new_lry - new_uly

    # 2. Paint each input's mask into the shared canvas at its
    #    offset position. ``|=`` is the per-pixel OR. Bool dtype
    #    keeps memory low and matches the ONEBIT convention used
    #    throughout the package. Track which pixels fall inside any
    #    child bbox so step 3 can fill only the gap pixels.
    canvas = np.zeros((new_nrows, new_ncols), dtype=bool)
    covered = np.zeros((new_nrows, new_ncols), dtype=bool)
    for g in glyphs:
        dy = g.uly - new_uly
        dx = g.ulx - new_ulx
        canvas[dy : dy + g.nrows, dx : dx + g.ncols] |= g.to_array()
        covered[dy : dy + g.nrows, dx : dx + g.ncols] = True

    # 3. Fill gap pixels (in the union bbox but outside every child
    #    bbox) from the page mask, when one was supplied. Clip the
    #    page slice to the page rectangle so a bbox that runs past
    #    the page edge doesn't crash — that case can only happen if
    #    upstream produced a bbox larger than the page, but we'd
    #    rather degrade gracefully than raise here.
    if page_mask is not None:
        page_h, page_w = page_mask.shape
        x0 = max(0, new_ulx)
        y0 = max(0, new_uly)
        x1 = min(page_w, new_ulx + new_ncols)
        y1 = min(page_h, new_uly + new_nrows)
        if x1 > x0 and y1 > y0:
            cy0 = y0 - new_uly
            cx0 = x0 - new_ulx
            cy1 = cy0 + (y1 - y0)
            cx1 = cx0 + (x1 - x0)
            gap = page_mask[y0:y1, x0:x1] & ~covered[cy0:cy1, cx0:cx1]
            canvas[cy0:cy1, cx0:cx1] |= gap

    # 4. Wrap as a Glyph. Glyph.new() generates a fresh UUID by
    #    default, which is exactly what we want — the grouped glyph
    #    is a *new* entity, not a relabel of an existing one.
    return Glyph.new(
        class_name=str(class_name),
        image_rle=array_to_rle(canvas),
        ncols=new_ncols,
        nrows=new_nrows,
        ulx=new_ulx,
        uly=new_uly,
        id_state_manual=True,
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# Auto-group — deferred stubs
# ---------------------------------------------------------------------------


_AUTO_GROUP_DEFERRED_MSG = (
    "Auto-grouping is deferred in Phase 1. See "
    "docs/migration_plan.md §'Risks and gotchas' (4): with per-neume "
    "cropped input the glyphs do not share a page-coordinate frame, "
    "so spatial grouping only works once the ingestion format carries "
    "per-crop (page_id, x, y, w, h) metadata. Reintroduce this function "
    "with scipy.ndimage.distance_transform_edt (Shaped) or pure-numpy "
    "bbox distance (BoundingBox) once that decision is made."
)


def auto_group_shaped(
    glyphs: Sequence[Glyph],
    *,
    distance: int,
    max_parts_per_group: int,
    max_graph_size: int,
    criterion: str,
) -> tuple[list[Glyph], list[Glyph]]:
    """Deferred — pixel-shape-based auto-grouping.

    Will replace Gamera's ``ShapedGroupingFunction`` /
    ``cknn.group_list_automatic`` once the page-coordinate ingestion
    decision is made. Intended return contract (for forward
    compatibility): ``(add, remove)`` where ``remove`` are the
    original glyphs absorbed into groups and ``add`` are the new
    grouped glyphs with updated class names and confidences. The
    ``max_graph_size`` parameter caps graph size to prevent blow-up
    on dense layouts (gotcha #5 in the migration plan).
    """
    raise NotImplementedError(_AUTO_GROUP_DEFERRED_MSG)


def auto_group_bounding_box(
    glyphs: Sequence[Glyph],
    *,
    distance: int,
    max_parts_per_group: int,
    max_graph_size: int,
    criterion: str,
) -> tuple[list[Glyph], list[Glyph]]:
    """Deferred — bounding-box-distance auto-grouping.

    Will replace Gamera's ``BoundingBoxGroupingFunction``. Same
    return contract as :func:`auto_group_shaped`; same dependency on
    a shared page-coordinate frame.
    """
    raise NotImplementedError(_AUTO_GROUP_DEFERRED_MSG)
