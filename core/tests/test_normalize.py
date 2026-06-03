"""Unit tests for the opt-in feature-space normalization transforms."""
from __future__ import annotations

import numpy as np

from ic_core.normalize import (
    NormalizeConfig,
    center_by_mass,
    despeckle,
    normalize_mask,
    pad_mask,
)


# ---------------------------------------------------------------------------
# NormalizeConfig
# ---------------------------------------------------------------------------


def test_default_config_is_noop():
    cfg = NormalizeConfig()
    assert cfg.is_noop
    mask = np.array([[True, False], [False, True]])
    # A no-op config returns the very same array object — no copy.
    assert normalize_mask(mask, cfg) is mask


def test_any_enabled_step_makes_config_active():
    assert not NormalizeConfig(despeckle_min_size=3).is_noop
    assert not NormalizeConfig(center=True).is_noop
    assert not NormalizeConfig(pad=1).is_noop


# ---------------------------------------------------------------------------
# despeckle
# ---------------------------------------------------------------------------


def test_despeckle_removes_small_components_keeps_large():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:6, 1:6] = True  # 25-px blob
    mask[9, 9] = True       # 1-px speck
    out = despeckle(mask, min_size=3)
    assert out[1:6, 1:6].all()
    assert not out[9, 9]


def test_despeckle_min_size_one_is_noop():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    assert despeckle(mask, min_size=1) is mask


def test_despeckle_empty_mask_unchanged():
    mask = np.zeros((3, 3), dtype=bool)
    assert despeckle(mask, min_size=5) is mask


def test_despeckle_all_small_keeps_largest():
    # Two specks, neither meeting the threshold; the bigger survives.
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = True            # 1 px
    mask[3:5, 3:5] = True        # 4 px
    out = despeckle(mask, min_size=100)
    assert out.sum() == 4
    assert out[3:5, 3:5].all()
    assert not out[0, 0]


def test_despeckle_diagonal_pixels_are_one_component():
    # 8-connectivity: the diagonal touch keeps them as one 2-px blob,
    # which survives min_size=2.
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    out = despeckle(mask, min_size=2)
    assert out[0, 0] and out[1, 1]


# ---------------------------------------------------------------------------
# center_by_mass
# ---------------------------------------------------------------------------


def test_center_puts_centroid_at_array_centre():
    # Single pixel hard against the top-left of a big frame.
    mask = np.zeros((9, 9), dtype=bool)
    mask[0, 0] = True
    out = center_by_mass(mask)
    ys, xs = np.nonzero(out)
    cy, cx = ys.mean(), xs.mean()
    # Centroid should sit at the geometric centre (within rounding).
    assert abs(cy - (out.shape[0] - 1) / 2) <= 0.5
    assert abs(cx - (out.shape[1] - 1) / 2) <= 0.5


def test_center_already_centred_is_unchanged():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    assert center_by_mass(mask) is mask


def test_center_preserves_foreground_count():
    mask = np.zeros((7, 7), dtype=bool)
    mask[0:3, 0:2] = True
    out = center_by_mass(mask)
    assert out.sum() == mask.sum()


def test_center_empty_mask_unchanged():
    mask = np.zeros((4, 4), dtype=bool)
    assert center_by_mass(mask) is mask


# ---------------------------------------------------------------------------
# pad_mask
# ---------------------------------------------------------------------------


def test_pad_adds_uniform_border():
    mask = np.ones((2, 3), dtype=bool)
    out = pad_mask(mask, pad=2)
    assert out.shape == (6, 7)
    assert out[2:4, 2:5].all()
    assert out.sum() == mask.sum()  # only background added


def test_pad_zero_is_noop():
    mask = np.ones((2, 2), dtype=bool)
    assert pad_mask(mask, pad=0) is mask


# ---------------------------------------------------------------------------
# normalize_mask orchestration
# ---------------------------------------------------------------------------


def test_normalize_pipeline_order_and_shape():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:5, 1:5] = True   # 16-px blob
    mask[9, 9] = True        # speck to be despeckled
    cfg = NormalizeConfig(despeckle_min_size=3, center=True, pad=2)
    out = normalize_mask(mask, cfg)
    # Speck gone → only the blob's 16 px remain.
    assert out.sum() == 16
    # Padding added a 2-px ring on each side after centring.
    ys, xs = np.nonzero(out)
    assert ys.min() >= 2 and xs.min() >= 2
    assert (out.shape[0] - 1 - ys.max()) >= 2
    assert (out.shape[1] - 1 - xs.max()) >= 2


def test_normalize_mask_dtype_stays_bool():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    out = normalize_mask(mask, NormalizeConfig(center=True, pad=1))
    assert out.dtype == np.bool_
