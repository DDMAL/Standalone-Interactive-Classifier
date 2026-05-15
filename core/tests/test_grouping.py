"""Unit tests for :mod:`ic_core.grouping`.

Manual group is exercised end-to-end against synthetic glyphs. The
auto-group stubs are tested only for their deferred-error contract —
they will get real tests once the page-coordinate ingestion decision
is made (see ``docs/migration_plan.md`` §"Risks and gotchas" (4)).
"""
from __future__ import annotations

import numpy as np
import pytest

from ic_core.glyph import Glyph
from ic_core.grouping import (
    auto_group_bounding_box,
    auto_group_shaped,
    manual_group,
)
from ic_core.image import array_to_rle


def _make_glyph(
    arr: np.ndarray,
    *,
    class_name: str = "X",
    id_state_manual: bool = False,
    confidence: float = 0.0,
    ulx: int = 0,
    uly: int = 0,
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
    )


# ---------------------------------------------------------------------------
# manual_group — basic invariants
# ---------------------------------------------------------------------------


def test_manual_group_empty_input_raises():
    with pytest.raises(ValueError, match="at least one glyph"):
        manual_group([], "A")


def test_manual_group_sets_manual_flag_and_full_confidence():
    # Algorithm semantic #7: union_images result is id_state_manual=True,
    # confidence=1 — it must be immediately usable as training data.
    a = _make_glyph(np.ones((4, 4), dtype=bool), ulx=0, uly=0)
    b = _make_glyph(np.ones((4, 4), dtype=bool), ulx=10, uly=0)

    grouped = manual_group([a, b], "neume.compound")

    assert grouped.id_state_manual is True
    assert grouped.confidence == 1.0
    assert grouped.class_name == "neume.compound"


def test_manual_group_assigns_fresh_uuid():
    # Algorithm semantic #6: newly created glyphs get fresh UUIDs;
    # the grouped glyph must not collide with any input id.
    a = _make_glyph(np.ones((4, 4), dtype=bool), ulx=0, uly=0)
    b = _make_glyph(np.ones((4, 4), dtype=bool), ulx=10, uly=0)

    grouped = manual_group([a, b], "A")

    assert grouped.id not in {a.id, b.id}
    assert len(grouped.id) == 32  # uuid4.hex


def test_manual_group_class_name_is_stringified():
    # The legacy code stringified class_name via str(); mirror that
    # so callers passing e.g. a path-like or bytes don't break the
    # XML export downstream.
    a = _make_glyph(np.ones((2, 2), dtype=bool))
    grouped = manual_group([a], 12345)  # type: ignore[arg-type]
    assert grouped.class_name == "12345"


# ---------------------------------------------------------------------------
# manual_group — bounding-box geometry
# ---------------------------------------------------------------------------


def test_manual_group_bbox_encompasses_all_inputs():
    # Three glyphs at varied positions; the union bbox should hug
    # the outermost edges in both x and y.
    a = _make_glyph(np.ones((3, 3), dtype=bool), ulx=10, uly=20)
    b = _make_glyph(np.ones((4, 5), dtype=bool), ulx=100, uly=50)
    c = _make_glyph(np.ones((2, 2), dtype=bool), ulx=30, uly=200)

    grouped = manual_group([a, b, c], "A")

    # ulx/uly = mins across inputs.
    assert grouped.ulx == 10
    assert grouped.uly == 20
    # Lower-right = max(ulx + ncols) and max(uly + nrows).
    assert grouped.ulx + grouped.ncols == 100 + 5  # 105
    assert grouped.uly + grouped.nrows == 200 + 2  # 202


def test_manual_group_single_glyph_returns_identical_bbox():
    # Degenerate case — one input. Geometry should be unchanged.
    a = _make_glyph(np.ones((5, 7), dtype=bool), ulx=42, uly=11)
    grouped = manual_group([a], "A")
    assert (grouped.ulx, grouped.uly, grouped.ncols, grouped.nrows) == (42, 11, 7, 5)


# ---------------------------------------------------------------------------
# manual_group — mask union (bitwise OR)
# ---------------------------------------------------------------------------


def test_manual_group_disjoint_masks_or_correctly():
    # Two non-overlapping single-pixel glyphs placed 5 pixels apart.
    # The union should contain exactly those two foreground pixels.
    left = _make_glyph(np.array([[True]], dtype=bool), ulx=0, uly=0)
    right = _make_glyph(np.array([[True]], dtype=bool), ulx=5, uly=0)

    grouped = manual_group([left, right], "A")
    arr = grouped.to_array()

    assert arr.shape == (1, 6)
    assert arr[0, 0] is np.True_
    assert arr[0, 5] is np.True_
    # All interior pixels are background.
    assert not arr[0, 1:5].any()


def test_manual_group_overlapping_masks_or_correctly():
    # Two overlapping glyphs: the OR should fill the union of their
    # foreground pixels — no double-counting, no clipping.
    a = _make_glyph(np.ones((2, 2), dtype=bool), ulx=0, uly=0)
    b = _make_glyph(np.ones((2, 2), dtype=bool), ulx=1, uly=1)

    grouped = manual_group([a, b], "A")
    arr = grouped.to_array()

    # Combined bbox is 3x3; foreground = {(0..1, 0..1) ∪ (1..2, 1..2)}.
    assert arr.shape == (3, 3)
    expected = np.array(
        [
            [True, True, False],
            [True, True, True],
            [False, True, True],
        ]
    )
    np.testing.assert_array_equal(arr, expected)


def test_manual_group_total_black_pixels_at_least_each_input():
    # Pixel count of the union must be ≥ each input's pixel count
    # (≤ sum, with equality iff inputs are disjoint).
    a = _make_glyph(np.ones((3, 3), dtype=bool), ulx=0, uly=0)
    b = _make_glyph(np.ones((3, 3), dtype=bool), ulx=10, uly=10)

    grouped = manual_group([a, b], "A")
    arr = grouped.to_array()

    a_pixels = int(a.to_array().sum())
    b_pixels = int(b.to_array().sum())
    assert int(arr.sum()) == a_pixels + b_pixels  # disjoint case → exact sum


# ---------------------------------------------------------------------------
# Auto-group stubs — deferred error contract
# ---------------------------------------------------------------------------


def test_auto_group_shaped_is_deferred():
    with pytest.raises(NotImplementedError, match="deferred"):
        auto_group_shaped(
            [_make_glyph(np.ones((2, 2), dtype=bool))],
            distance=4,
            max_parts_per_group=4,
            max_graph_size=64,
            criterion="min",
        )


def test_auto_group_bounding_box_is_deferred():
    with pytest.raises(NotImplementedError, match="deferred"):
        auto_group_bounding_box(
            [_make_glyph(np.ones((2, 2), dtype=bool))],
            distance=4,
            max_parts_per_group=4,
            max_graph_size=64,
            criterion="min",
        )
