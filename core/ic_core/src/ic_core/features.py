"""Feature extraction.

Replaces the Gamera-internal feature vector computation. Reimplements a
minimal Phase-1 subset of features used by the classifier on top of
``numpy``, ``scikit-image`` and ``scipy``. The feature vector layout is
versioned by ``FEATURE_VERSION`` — files produced under one version are
not feature-compatible with another.

Phase 1: 7 features, 29 dimensions. See docs/migration_plan.md §190-192
for the clean-break rationale.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.ndimage import label
from skimage.measure import moments_central, moments_hu, moments_normalized, perimeter

from ic_core.glyph import Glyph

FEATURE_VERSION = "ic-core/v1"

FEATURE_NAMES: list[str] = [
    "aspect_ratio",
    "volume",
    "nrows_feature",
    "ncols_feature",
    "compactness",
    "nholes",
    *[f"volume16regions_{i}" for i in range(16)],
    *[f"hu_moment_{i}" for i in range(7)],
]


def compute_features(glyph: Glyph) -> np.ndarray:
    """Return a (29,) float64 feature vector matching ``FEATURE_NAMES``."""
    arr = glyph.to_array()
    parts = [
        np.array([_aspect_ratio(arr)], dtype=np.float64),
        np.array([_volume(arr)], dtype=np.float64),
        np.array([float(arr.shape[0])], dtype=np.float64),
        np.array([float(arr.shape[1])], dtype=np.float64),
        np.array([_compactness(arr)], dtype=np.float64),
        np.array([_nholes(arr)], dtype=np.float64),
        _volume16regions(arr),
        _hu_moments(arr),
    ]
    return np.concatenate(parts).astype(np.float64, copy=False)


def compute_features_batch(glyphs: Iterable[Glyph]) -> np.ndarray:
    """Return an (N, 29) float64 matrix of features for a batch of glyphs."""
    vectors = [compute_features(g) for g in glyphs]
    if not vectors:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack(vectors)


def _aspect_ratio(arr: np.ndarray) -> float:
    nrows, ncols = arr.shape
    if nrows == 0:
        return 0.0
    return ncols / nrows


def _volume(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(arr.sum()) / float(arr.size)


def _compactness(arr: np.ndarray) -> float:
    if not arr.any():
        return 0.0
    area = float(arr.sum())
    p = float(perimeter(arr))
    if p <= 0.0:
        return 0.0
    return 4.0 * np.pi * area / (p * p)


def _nholes(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    inverted = ~arr
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    _, n_components = label(inverted, structure=structure)
    return float(max(n_components - 1, 0))


def _volume16regions(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(16, dtype=np.float64)
    if arr.size == 0:
        return out
    row_splits = np.array_split(arr, 4, axis=0)
    idx = 0
    for row_block in row_splits:
        col_splits = np.array_split(row_block, 4, axis=1)
        for cell in col_splits:
            if cell.size == 0:
                out[idx] = 0.0
            else:
                out[idx] = float(cell.sum()) / float(cell.size)
            idx += 1
    return out


def _hu_moments(arr: np.ndarray) -> np.ndarray:
    if not arr.any():
        return np.zeros(7, dtype=np.float64)
    mu = moments_central(arr.astype(np.float64))
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    return np.asarray(hu, dtype=np.float64)
