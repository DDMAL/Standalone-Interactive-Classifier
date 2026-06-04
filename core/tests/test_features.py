from pathlib import Path

import numpy as np
import pytest
from lxml import etree

from ic_core.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    compute_features,
    compute_features_batch,
)
from ic_core.io_xml import load_glyphs

FIXTURE = Path(__file__).parent / "fixtures" / "Square_notation-example_training_data.xml"


def _gamera_features_by_glyph(path: Path) -> list[dict[str, list[float]]]:
    """Parse Gamera's <feature> tags from the fixture, in document order."""
    tree = etree.parse(str(path))
    out = []
    for g in tree.iterfind(".//glyph"):
        feats: dict[str, list[float]] = {}
        for f in g.iterfind("features/feature"):
            name = f.get("name")
            values = [float(v) for v in (f.text or "").split()]
            feats[name] = values
        out.append(feats)
    return out


def test_feature_names_length():
    assert len(FEATURE_NAMES) == 29


def test_feature_version_constant():
    assert FEATURE_VERSION == "ic-core/v1"


def test_compute_features_shape_and_dtype():
    glyphs = load_glyphs(FIXTURE)
    vec = compute_features(glyphs[0])
    assert vec.shape == (29,)
    assert vec.dtype == np.float64


def test_compute_features_batch_shape():
    glyphs = load_glyphs(FIXTURE)
    mat = compute_features_batch(glyphs)
    assert mat.shape == (len(glyphs), 29)
    assert mat.dtype == np.float64


def test_compute_features_batch_empty():
    mat = compute_features_batch([])
    assert mat.shape == (0, 29)


def test_no_nan_or_inf_across_fixture():
    glyphs = load_glyphs(FIXTURE)
    mat = compute_features_batch(glyphs)
    assert np.all(np.isfinite(mat))


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_definitionally_equal_features_match_gamera(idx):
    glyphs = load_glyphs(FIXTURE)
    gamera = _gamera_features_by_glyph(FIXTURE)
    vec = compute_features(glyphs[idx])
    feats = dict(zip(FEATURE_NAMES, vec))

    assert feats["aspect_ratio"] == pytest.approx(gamera[idx]["aspect_ratio"][0], abs=1e-6)
    assert feats["volume"] == pytest.approx(gamera[idx]["volume"][0], abs=1e-6)
    assert feats["nrows_feature"] == pytest.approx(gamera[idx]["nrows_feature"][0], abs=1e-6)
    assert feats["ncols_feature"] == pytest.approx(gamera[idx]["ncols_feature"][0], abs=1e-6)


def test_volume_matches_black_over_total_for_all_glyphs():
    glyphs = load_glyphs(FIXTURE)
    mat = compute_features_batch(glyphs)
    volume_idx = FEATURE_NAMES.index("volume")
    for i, g in enumerate(glyphs):
        arr = g.to_array()
        expected = float(arr.sum()) / float(arr.size)
        assert mat[i, volume_idx] == pytest.approx(expected, abs=1e-12)


def test_aspect_ratio_matches_ncols_over_nrows_for_all_glyphs():
    glyphs = load_glyphs(FIXTURE)
    mat = compute_features_batch(glyphs)
    ar_idx = FEATURE_NAMES.index("aspect_ratio")
    for i, g in enumerate(glyphs):
        assert mat[i, ar_idx] == pytest.approx(g.ncols / g.nrows, abs=1e-12)


def test_compactness_finite_and_nonneg():
    glyphs = load_glyphs(FIXTURE)
    mat = compute_features_batch(glyphs)
    col = mat[:, FEATURE_NAMES.index("compactness")]
    assert np.all(col >= 0.0)
    assert np.all(np.isfinite(col))


def test_nholes_nonneg_integer_values():
    glyphs = load_glyphs(FIXTURE)
    mat = compute_features_batch(glyphs)
    col = mat[:, FEATURE_NAMES.index("nholes")]
    assert np.all(col >= 0.0)
    np.testing.assert_array_equal(col, np.floor(col))


def test_volume16regions_in_unit_interval():
    glyphs = load_glyphs(FIXTURE)
    mat = compute_features_batch(glyphs)
    idxs = [FEATURE_NAMES.index(f"volume16regions_{i}") for i in range(16)]
    block = mat[:, idxs]
    assert np.all(block >= 0.0)
    assert np.all(block <= 1.0)


def test_hu_moments_finite():
    glyphs = load_glyphs(FIXTURE)
    mat = compute_features_batch(glyphs)
    idxs = [FEATURE_NAMES.index(f"hu_moment_{i}") for i in range(7)]
    block = mat[:, idxs]
    assert np.all(np.isfinite(block))
