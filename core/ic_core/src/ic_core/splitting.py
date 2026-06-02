"""Manual splitting via user-drawn rectangles.

Phase-1 replacement for the legacy split UX, which ran Gamera's
``segmentation.cc_analysis`` on a single glyph to break it into
connected components. We deliberately do **not** port the algorithmic
CCA path — real-world neume crops have touching strokes, ligatures,
and binarisation artefacts that bridge marks, which defeat
connected-components analysis and produce confidently-wrong outputs
on the cases that matter most. See ``docs/migration_plan.md``
§"Scope decision — manual splitting kept, algorithmic splitting
dropped" for the full rationale.

What we keep is the *outcome* of the legacy split action: a parent
glyph is replaced in the working set by N children, each marked
``UNCLASSIFIED`` / ``confidence=0`` / ``id_state_manual=False`` so the
next classify round re-labels them, each with a fresh UUID
(algorithm semantic #6 + #8 in the migration plan).

The boundary between children comes from the user, not an algorithm:
the frontend collects N axis-aligned rectangles drawn on the parent
glyph and posts them to ``manual_split``. Each rectangle becomes one
child; pixels outside every rectangle are discarded.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ic_core.classifier import UNCLASSIFIED
from ic_core.glyph import Glyph
from ic_core.image import array_to_rle

#: Class name assigned to every child of a manual split. Matches the
#: legacy split UX, which also emitted ``UNCLASSIFIED`` rather than a
#: ``_split``-prefixed class name (see migration plan §"Algorithm
#: semantics to preserve verbatim" #4).
_UNCLASSIFIED = UNCLASSIFIED


def manual_split(
    glyph: Glyph,
    regions: Sequence[tuple[int, int, int, int]],
) -> list[Glyph]:
    """Slice a parent glyph into N children along user-drawn rectangles.

    Each region is an axis-aligned rectangle in the **page coordinate
    frame** — the same frame the parent's ``ulx`` / ``uly`` lives in,
    so the frontend can hand off what it draws without translating.
    Coordinates are ``(ulx, uly, ncols, nrows)``; lower-right corner
    is exclusive (``ulx + ncols``, ``uly + nrows``).

    For each region:

    1. **Clip** the rectangle to the parent's bbox. Regions that miss
       the parent entirely (zero-area intersection) are dropped — no
       empty :class:`Glyph` is emitted.
    2. **Slice** the parent's binary mask over the clipped rectangle.
       The child's mask is the parent's mask restricted to the
       rectangle; pixels in the rectangle but background in the
       parent stay background.
    3. **Wrap** the slice as a new :class:`Glyph` with the clipped
       rectangle as its bbox, ``class_name="UNCLASSIFIED"``,
       ``confidence=0``, ``id_state_manual=False``, and a fresh UUID.

    Overlapping regions are allowed: overlap pixels appear in *every*
    child whose region covers them. This is intentional — the legacy
    UX never enforced disjoint cuts, and pixel duplication is the
    obvious behaviour for "I drew two boxes over the same area."
    Pixels outside every region are discarded; the caller (the
    session handler) is expected to remove the parent glyph from the
    working set when it inserts the children.

    Args:
        glyph: The parent :class:`Glyph` to split.
        regions: One or more ``(ulx, uly, ncols, nrows)`` tuples in
            page coordinates. Empty input is rejected — splitting a
            glyph into zero children is almost certainly a UI bug.

    Returns:
        A list of new :class:`Glyph` objects, one per region whose
        clip intersects the parent's bbox. May be shorter than
        ``regions`` if some regions miss entirely; may be empty if
        every region misses (the caller should treat that the same as
        any other "nothing to insert" outcome).

    Raises:
        ValueError: If ``regions`` is empty, or if any region has
            non-positive width/height.
    """
    if not regions:
        raise ValueError("manual_split requires at least one region")

    # Validate up-front so a bad region in the middle of the list
    # doesn't produce a half-built result.
    for i, (_ulx, _uly, ncols, nrows) in enumerate(regions):
        if ncols <= 0 or nrows <= 0:
            raise ValueError(
                f"manual_split region {i} has non-positive size: "
                f"ncols={ncols}, nrows={nrows}"
            )

    parent_mask = glyph.to_array()
    parent_ulx, parent_uly = glyph.ulx, glyph.uly
    parent_lrx = parent_ulx + glyph.ncols
    parent_lry = parent_uly + glyph.nrows

    children: list[Glyph] = []
    for region_ulx, region_uly, region_ncols, region_nrows in regions:
        region_lrx = region_ulx + region_ncols
        region_lry = region_uly + region_nrows

        # Clip the region to the parent's bbox (page coordinates).
        clip_ulx = max(region_ulx, parent_ulx)
        clip_uly = max(region_uly, parent_uly)
        clip_lrx = min(region_lrx, parent_lrx)
        clip_lry = min(region_lry, parent_lry)

        if clip_lrx <= clip_ulx or clip_lry <= clip_uly:
            # Region misses the parent entirely; drop it silently.
            continue

        # Translate the clip into parent-local mask coordinates and
        # slice. ``np.ascontiguousarray`` because the slice may be a
        # view; ``array_to_rle`` expects a contiguous buffer.
        local_y0 = clip_uly - parent_uly
        local_x0 = clip_ulx - parent_ulx
        local_y1 = clip_lry - parent_uly
        local_x1 = clip_lrx - parent_ulx
        child_mask = np.ascontiguousarray(
            parent_mask[local_y0:local_y1, local_x0:local_x1]
        )

        child_nrows, child_ncols = child_mask.shape
        children.append(
            Glyph.new(
                class_name=_UNCLASSIFIED,
                image_rle=array_to_rle(child_mask),
                ncols=child_ncols,
                nrows=child_nrows,
                ulx=clip_ulx,
                uly=clip_uly,
                id_state_manual=False,
                confidence=0.0,
                category=glyph.category,
            )
        )

    return children
