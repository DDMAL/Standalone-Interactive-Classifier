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

import contextlib
from dataclasses import replace
from typing import Iterable, Iterator

import numpy as np
from scipy.ndimage import label
from skimage.measure import moments_central, moments_hu, moments_normalized, perimeter

from ic_core.glyph import Glyph
from ic_core.normalize import NormalizeConfig, normalize_mask

FEATURE_VERSION = "ic-core/v1"

#: Active feature-space normalization, set only inside the
#: :func:`feature_normalization` context manager. ``None`` means
#: "raw mask" — the default everywhere (production ingest, the
#: Gamera-parity tests, the API). Normalization is opt-in so the
#: baseline behaviour and the legacy feature parity stay intact; the
#: A/B harness flips it on for the treatment arm only.
_ACTIVE_NORMALIZATION: NormalizeConfig | None = None


@contextlib.contextmanager
def feature_normalization(cfg: NormalizeConfig | None) -> Iterator[None]:
    """Apply ``cfg`` to every mask fed to :func:`compute_features` in this block.

    Process-global and not thread-safe — it mutates module state — but
    the only caller is the single-threaded evaluation harness, where
    that's exactly the point: wrap the treatment arm and every
    classifier-internal feature computation picks up the config without
    threading a flag through ``fit`` / ``predict_many``.

    Passing ``None`` (or a no-op :class:`NormalizeConfig`) is a no-op.
    Restores the previous config on exit, so blocks can nest.
    """
    global _ACTIVE_NORMALIZATION
    prev = _ACTIVE_NORMALIZATION
    _ACTIVE_NORMALIZATION = cfg
    try:
        yield
    finally:
        _ACTIVE_NORMALIZATION = prev

#: Logical feature definitions: one entry per *named* feature in the
#: legacy GameraXML ``<feature name="...">`` convention, paired with
#: its dimensionality. The flat :data:`FEATURE_NAMES` list and the
#: 29-d vector returned by :func:`compute_features` are derived from
#: this so the two representations stay in sync. The XML writer uses
#: this directly to emit one ``<feature>`` element per logical
#: feature (multi-dim values space-separated inside the element),
#: matching the Square_notation fixture layout.
LOGICAL_FEATURES: tuple[tuple[str, int], ...] = (
    ("aspect_ratio", 1),
    ("volume", 1),
    ("nrows_feature", 1),
    ("ncols_feature", 1),
    ("compactness", 1),
    ("nholes", 1),
    ("volume16regions", 16),
    ("hu_moment", 7),
)

FEATURE_NAMES: list[str] = [
    f"{name}_{i}" if dim > 1 else name
    for name, dim in LOGICAL_FEATURES
    for i in range(dim)
]


def compute_features(glyph: Glyph) -> np.ndarray:
    """Return a (29,) float64 feature vector matching ``FEATURE_NAMES``.

    When a :func:`feature_normalization` block is active, the glyph's
    binary mask is normalized (despeckle / centre / pad) *before*
    feature extraction. The glyph itself is untouched — only the
    numbers handed to the classifier change.
    """
    arr = glyph.to_array()
    if _ACTIVE_NORMALIZATION is not None:
        arr = normalize_mask(arr, _ACTIVE_NORMALIZATION)
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


def get_features(glyph: Glyph) -> np.ndarray:
    """Return ``glyph``'s feature vector, reusing the cache when valid.

    A cache is "valid" when both ``feature_vector`` and
    ``feature_version`` are set on the glyph and the version matches
    the current :data:`FEATURE_VERSION`. A version mismatch causes a
    fresh computation — old cached vectors silently fall through.

    This is the function every consumer (classifier, XML writer)
    should call instead of :func:`compute_features` directly, so the
    cache stays consulted.
    """
    if (
        glyph.feature_vector is not None
        and glyph.feature_version == FEATURE_VERSION
    ):
        return glyph.feature_vector
    return compute_features(glyph)


def ensure_features(glyph: Glyph) -> Glyph:
    """Return ``glyph`` with its feature cache populated under the current version.

    If the cache is already valid, returns the input unchanged
    (identity). Otherwise computes the vector and returns a new
    :class:`Glyph` via :func:`dataclasses.replace` with
    ``feature_vector`` and ``feature_version`` filled in.

    Callers that want the cache to persist across rounds — most
    notably :meth:`ic_core.state.Session.classify` — should map this
    over their glyph list and write the result back.
    """
    if (
        glyph.feature_vector is not None
        and glyph.feature_version == FEATURE_VERSION
    ):
        return glyph
    return replace(
        glyph,
        feature_vector=compute_features(glyph),
        feature_version=FEATURE_VERSION,
    )


def compute_features_batch(glyphs: Iterable[Glyph]) -> np.ndarray:
    """Return an (N, 29) float64 matrix of features for a batch of glyphs.

    Uses :func:`get_features` per glyph so cached vectors are reused
    rather than recomputed — the "full re-train every round" loop
    asks for features over the same stable training pool repeatedly,
    and that's exactly the case the cache exists to short-circuit.
    """
    vectors = [get_features(g) for g in glyphs]
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
