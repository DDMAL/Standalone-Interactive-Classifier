"""Unit tests for :mod:`ic_core.splitting`.

``manual_split`` slices a parent glyph along user-drawn rectangles in
page coordinates. The semantics under test mirror the migration
plan's algorithm semantic #8: each child is ``UNCLASSIFIED`` /
``confidence=0`` / ``id_state_manual=False`` with a fresh UUID, and
geometry is the intersection of the user's rectangle with the
parent's bbox.
"""
from __future__ import annotations

import numpy as np
import pytest

from ic_core.glyph import CATEGORY_NEUMES, CATEGORY_TEXT, Glyph
from ic_core.image import array_to_rle
from ic_core.splitting import manual_split


def _make_glyph(
    arr: np.ndarray,
    *,
    class_name: str = "neume.punctum",
    id_state_manual: bool = True,
    confidence: float = 1.0,
    ulx: int = 0,
    uly: int = 0,
    category: str = CATEGORY_NEUMES,
) -> Glyph:
    """Build a Glyph from a 2-D boolean array placed at (ulx, uly)."""
    arr = np.asarray(arr, dtype=bool)
    nrows, ncols = arr.shape
    return Glyph.new(
        class_name=class_name,
        image_rle=array_to_rle(arr),
        ncols=ncols,
        nrows=nrows,
        ulx=ulx,
        uly=uly,
        id_state_manual=id_state_manual,
        confidence=confidence,
        category=category,
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_manual_split_empty_regions_raises():
    glyph = _make_glyph(np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="at least one region"):
        manual_split(glyph, [])


def test_manual_split_zero_size_region_raises():
    glyph = _make_glyph(np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="non-positive"):
        manual_split(glyph, [(0, 0, 0, 4)])
    with pytest.raises(ValueError, match="non-positive"):
        manual_split(glyph, [(0, 0, 4, 0)])


def test_manual_split_negative_size_region_raises():
    glyph = _make_glyph(np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="non-positive"):
        manual_split(glyph, [(0, 0, -1, 4)])


def test_manual_split_validation_runs_before_any_output():
    # Algorithm guarantee: a bad region in the middle of the list
    # produces no half-built result. The valid first region must not
    # leak through as a side effect.
    glyph = _make_glyph(np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="non-positive"):
        manual_split(glyph, [(0, 0, 2, 2), (0, 0, 0, 2)])


# ---------------------------------------------------------------------------
# Output state — algorithm semantic #8
# ---------------------------------------------------------------------------


def test_manual_split_children_are_unclassified():
    # The whole point of the split: children re-enter classification,
    # so they must NOT carry the parent's label or training flag.
    glyph = _make_glyph(
        np.ones((4, 4), dtype=bool),
        class_name="neume.compound",
        id_state_manual=True,
        confidence=1.0,
    )
    children = manual_split(glyph, [(0, 0, 2, 4), (2, 0, 2, 4)])

    assert len(children) == 2
    for child in children:
        assert child.class_name == "UNCLASSIFIED"
        assert child.confidence == 0.0
        assert child.id_state_manual is False


def test_manual_split_children_get_fresh_uuids():
    # Algorithm semantic #6: newly created glyphs always get fresh
    # UUIDs. None of the children may collide with the parent or
    # with each other.
    glyph = _make_glyph(np.ones((4, 4), dtype=bool))
    children = manual_split(glyph, [(0, 0, 2, 4), (2, 0, 2, 4)])

    ids = {c.id for c in children}
    assert glyph.id not in ids
    assert len(ids) == len(children)  # no duplicates
    for child in children:
        assert len(child.id) == 32  # uuid4.hex


def test_manual_split_preserves_parent_category():
    # MOTHRA category (Neumes / Text / Staves) is a property of the
    # mark, not the segmentation. Children inherit it.
    glyph = _make_glyph(np.ones((4, 4), dtype=bool), category=CATEGORY_TEXT)
    children = manual_split(glyph, [(0, 0, 4, 4)])
    assert children[0].category == CATEGORY_TEXT


# ---------------------------------------------------------------------------
# Geometry — page-coordinate clipping
# ---------------------------------------------------------------------------


def test_manual_split_child_bbox_matches_region_when_fully_inside():
    # Region is strictly inside the parent — the child's bbox is the
    # region itself, no clipping.
    glyph = _make_glyph(np.ones((10, 10), dtype=bool), ulx=100, uly=200)
    children = manual_split(glyph, [(102, 203, 4, 5)])
    assert (children[0].ulx, children[0].uly) == (102, 203)
    assert (children[0].ncols, children[0].nrows) == (4, 5)


def test_manual_split_clips_region_to_parent_bbox():
    # Region overhangs the parent on the right and bottom. The child
    # bbox is the intersection.
    glyph = _make_glyph(np.ones((10, 10), dtype=bool), ulx=0, uly=0)
    # Region: (5, 5) → (15, 15). Parent: (0, 0) → (10, 10). Clip → (5, 5) → (10, 10).
    children = manual_split(glyph, [(5, 5, 10, 10)])
    assert (children[0].ulx, children[0].uly) == (5, 5)
    assert (children[0].ncols, children[0].nrows) == (5, 5)


def test_manual_split_drops_regions_that_miss_parent():
    # Region is entirely outside the parent's bbox — no child emitted.
    glyph = _make_glyph(np.ones((4, 4), dtype=bool), ulx=0, uly=0)
    children = manual_split(glyph, [(100, 100, 5, 5)])
    assert children == []


def test_manual_split_drops_missing_regions_but_keeps_others():
    # Mixed list: one region misses, one hits. The result has length 1.
    glyph = _make_glyph(np.ones((4, 4), dtype=bool), ulx=0, uly=0)
    children = manual_split(glyph, [(100, 100, 5, 5), (0, 0, 4, 4)])
    assert len(children) == 1
    assert (children[0].ulx, children[0].uly) == (0, 0)


# ---------------------------------------------------------------------------
# Mask slicing
# ---------------------------------------------------------------------------


def test_manual_split_child_mask_is_parent_slice():
    # Parent has a recognisable pattern; the child mask should be
    # exactly the parent's slice at the region's local offset.
    parent_arr = np.array(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    glyph = _make_glyph(parent_arr, ulx=10, uly=20)

    # Region covers the right half of the parent in page coords.
    children = manual_split(glyph, [(12, 20, 2, 4)])
    child_arr = children[0].to_array()

    # Right half of the original pattern.
    np.testing.assert_array_equal(child_arr, parent_arr[:, 2:4])


def test_manual_split_full_cover_round_trips_parent_pattern():
    # One region covering the whole parent → child mask equals the
    # parent's mask (state aside). Coverage check that the slicing
    # arithmetic is correct in the trivial case.
    parent_arr = np.array(
        [[True, False, True], [False, True, False]],
        dtype=bool,
    )
    glyph = _make_glyph(parent_arr, ulx=7, uly=3)
    children = manual_split(glyph, [(7, 3, 3, 2)])
    np.testing.assert_array_equal(children[0].to_array(), parent_arr)


def test_manual_split_respects_parent_background_pixels():
    # The child mask should be the parent's mask restricted to the
    # region, NOT a filled rectangle. Pixels that are background in
    # the parent stay background in the child.
    parent_arr = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    glyph = _make_glyph(parent_arr, ulx=0, uly=0)
    # Region covers the whole parent — the empty bottom-right
    # quadrant must remain empty.
    children = manual_split(glyph, [(0, 0, 4, 4)])
    np.testing.assert_array_equal(children[0].to_array(), parent_arr)


# ---------------------------------------------------------------------------
# Overlapping regions
# ---------------------------------------------------------------------------


def test_manual_split_overlapping_regions_both_produce_children():
    # Overlapping regions are allowed; overlap pixels appear in BOTH
    # children. This matches the docstring's documented behaviour.
    glyph = _make_glyph(np.ones((4, 4), dtype=bool), ulx=0, uly=0)
    children = manual_split(glyph, [(0, 0, 3, 4), (1, 0, 3, 4)])

    assert len(children) == 2
    # Both children include column x=1 and x=2 of the parent.
    assert children[0].ncols == 3 and children[0].ulx == 0
    assert children[1].ncols == 3 and children[1].ulx == 1
    # The two children together cover MORE pixels than the parent
    # (overlap is duplicated).
    total = int(children[0].to_array().sum()) + int(children[1].to_array().sum())
    assert total == 4 * 3 + 4 * 3  # 24, vs. 16 parent pixels


# ---------------------------------------------------------------------------
# Non-trivial parent offset (page-coord frame)
# ---------------------------------------------------------------------------


def test_manual_split_works_with_nonzero_parent_offset():
    # The page-coord frame is the whole point of accepting region
    # coords in page space. With a parent at (50, 70), the user's
    # region at (52, 71, 3, 2) must land at parent-local (2, 1) for
    # the slice.
    parent_arr = np.zeros((5, 6), dtype=bool)
    parent_arr[1:3, 2:5] = True  # known foreground rectangle
    glyph = _make_glyph(parent_arr, ulx=50, uly=70)

    children = manual_split(glyph, [(52, 71, 3, 2)])
    assert (children[0].ulx, children[0].uly) == (52, 71)
    np.testing.assert_array_equal(children[0].to_array(), parent_arr[1:3, 2:5])
