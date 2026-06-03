"""Tests for :mod:`ic_core.ingest`.

Uses the real sample input under ``core/data/test/`` so the test
exercises the actual MOTHRA JSON and YOLO formats the upstream
detector produces. Ingest now consumes raw bytes (not paths), so
fixtures are read once at module load and passed by value to each
call.
"""
from __future__ import annotations

import json
import uuid

import numpy as np
import pytest
from PIL import Image as PILImage

from ic_core.classifier import UNCLASSIFIED
from ic_core.ingest import (
    _binarize_page,
    _unwrap_page,
    binarize_page,
    ingest_page,
    ingest_page_json,
    ingest_page_yolo,
)
from paths import TEST_JSON, TEST_PAGE, TEST_YOLO

PAGE_BYTES = TEST_PAGE.read_bytes()
JSON_BYTES = TEST_JSON.read_bytes()
YOLO_BYTES = TEST_YOLO.read_bytes()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_ingest_page_dispatches_on_format():
    json_glyphs = ingest_page(PAGE_BYTES, JSON_BYTES, format="json")
    yolo_glyphs = ingest_page(PAGE_BYTES, YOLO_BYTES, format="yolo")
    # Both should produce non-trivial output from the sample page.
    assert len(json_glyphs) > 0
    assert len(yolo_glyphs) > 0


def test_ingest_page_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unrecognised"):
        ingest_page(PAGE_BYTES, JSON_BYTES, format="csv")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# JSON ingest
# ---------------------------------------------------------------------------


def test_json_ingest_count_matches_annotations():
    doc = _unwrap_page(json.loads(JSON_BYTES))
    expected = len(doc["annotations"])
    glyphs = ingest_page_json(PAGE_BYTES, JSON_BYTES)
    assert len(glyphs) == expected


def test_json_ingest_preserves_annotation_ids_as_glyph_uuids():
    doc = _unwrap_page(json.loads(JSON_BYTES))
    glyphs = ingest_page_json(PAGE_BYTES, JSON_BYTES)

    expected_ids = [uuid.UUID(a["id"]).hex for a in doc["annotations"]]
    actual_ids = [g.id for g in glyphs]
    assert actual_ids == expected_ids


def test_json_ingest_bbox_coords_match_annotations():
    doc = _unwrap_page(json.loads(JSON_BYTES))
    glyphs = ingest_page_json(PAGE_BYTES, JSON_BYTES)

    # Spot-check the first three glyphs — full enumeration is the
    # same logic; a sample is enough to catch a mis-rounded coord.
    for ann, g in zip(doc["annotations"][:3], glyphs[:3]):
        ulx, uly, w, h = ann["bbox"]
        assert g.ulx == int(round(ulx))
        assert g.uly == int(round(uly))
        # The cropped glyph may be smaller than the declared bbox if
        # the bbox runs past the page edge, but should match for
        # interior crops.
        assert g.ncols == int(round(w))
        assert g.nrows == int(round(h))


def test_json_ingest_marks_everything_unclassified():
    # The detector's classId becomes a coarse category, but no neume
    # label — the user labels glyphs through the API, not at ingest time.
    glyphs = ingest_page_json(PAGE_BYTES, JSON_BYTES)
    assert all(g.class_name == UNCLASSIFIED for g in glyphs)
    assert all(g.confidence == 0.0 for g in glyphs)
    assert all(g.id_state_manual is False for g in glyphs)


def test_json_ingest_is_idempotent_in_id_space():
    # The whole point of preserving the JSON id field: re-ingesting
    # produces the same glyph ids in the same positions.
    a = ingest_page_json(PAGE_BYTES, JSON_BYTES)
    b = ingest_page_json(PAGE_BYTES, JSON_BYTES)
    assert [g.id for g in a] == [g.id for g in b]


def test_json_ingest_accepts_both_dict_and_single_element_list():
    # MOTHRA emits the page either as a bare object or wrapped in a
    # one-element list; both must yield identical glyphs.
    page = _unwrap_page(json.loads(JSON_BYTES))
    as_dict = json.dumps(page).encode("utf-8")
    as_list = json.dumps([page]).encode("utf-8")
    from_dict = ingest_page_json(PAGE_BYTES, as_dict)
    from_list = ingest_page_json(PAGE_BYTES, as_list)
    assert [g.id for g in from_dict] == [g.id for g in from_list]
    assert len(from_list) == len(page["annotations"])


def test_json_ingest_rejects_multi_page_list():
    page = _unwrap_page(json.loads(JSON_BYTES))
    multi = json.dumps([page, page]).encode("utf-8")
    with pytest.raises(ValueError, match="single-page"):
        ingest_page_json(PAGE_BYTES, multi)


# ---------------------------------------------------------------------------
# YOLO ingest
# ---------------------------------------------------------------------------


def test_yolo_ingest_count_matches_lines():
    lines = [
        line
        for line in YOLO_BYTES.decode("utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    glyphs = ingest_page_yolo(PAGE_BYTES, YOLO_BYTES)
    assert len(glyphs) == len(lines)


def test_yolo_ingest_assigns_fresh_uuids():
    a = ingest_page_yolo(PAGE_BYTES, YOLO_BYTES)
    b = ingest_page_yolo(PAGE_BYTES, YOLO_BYTES)
    # YOLO carries no stable ids — two runs should produce
    # different glyph UUIDs.
    assert all(len(g.id) == 32 for g in a)
    assert {g.id for g in a}.isdisjoint({g.id for g in b})


def test_yolo_ingest_first_glyph_matches_yolo_geometry():
    # First line: 0 0.292332 0.183886 0.051083 0.033496
    import io

    with PILImage.open(io.BytesIO(PAGE_BYTES)) as im:
        img_w, img_h = im.size

    glyphs = ingest_page_yolo(PAGE_BYTES, YOLO_BYTES)
    cx, cy, w, h = 0.292332, 0.183886, 0.051083, 0.033496
    expected_ulx = int(round((cx - w / 2) * img_w))
    expected_uly = int(round((cy - h / 2) * img_h))
    expected_w = int(round(w * img_w))
    expected_h = int(round(h * img_h))

    g = glyphs[0]
    assert g.ulx == expected_ulx
    assert g.uly == expected_uly
    assert g.ncols == expected_w
    assert g.nrows == expected_h


# ---------------------------------------------------------------------------
# Cropping behaviour
# ---------------------------------------------------------------------------


def test_crop_mask_dtype_and_shape():
    glyphs = ingest_page_json(PAGE_BYTES, JSON_BYTES)
    g = glyphs[0]
    arr = g.to_array()
    assert arr.dtype == np.bool_
    assert arr.shape == (g.nrows, g.ncols)


def test_crop_has_some_foreground_pixels():
    # Every annotated bbox should contain at least one ink pixel on
    # a real chant page. If a glyph comes back fully white, either
    # the threshold or the bbox alignment is wrong.
    glyphs = ingest_page_json(PAGE_BYTES, JSON_BYTES)
    total_fg = sum(int(g.to_array().sum()) for g in glyphs)
    assert total_fg > 0


# ---------------------------------------------------------------------------
# Threshold stabilization
# ---------------------------------------------------------------------------


def test_binarize_fixed_matches_raw_comparison():
    # The "fixed" method must stay byte-identical to the historical
    # ``page <= 127`` so existing output never silently shifts.
    from ic_core.ingest import _load_page

    page = _load_page(PAGE_BYTES)
    out = _binarize_page(page, method="fixed", threshold=127, sauvola_window=25)
    np.testing.assert_array_equal(out, page <= 127)


def test_binarize_otsu_picks_global_threshold():
    from ic_core.ingest import _load_page
    from skimage.filters import threshold_otsu

    page = _load_page(PAGE_BYTES)
    out = _binarize_page(page, method="otsu", threshold=127, sauvola_window=25)
    np.testing.assert_array_equal(out, page <= threshold_otsu(page))


def test_binarize_otsu_falls_back_on_uniform_page():
    # A blank/uniform page has no bimodal histogram; Otsu would raise,
    # so we fall back to the fixed cutoff instead of crashing.
    uniform = np.full((8, 8), 200, dtype=np.uint8)
    out = _binarize_page(uniform, method="otsu", threshold=127, sauvola_window=25)
    # 200 > 127 → all background, no crash.
    assert out.shape == (8, 8)
    assert not out.any()


def test_binarize_sauvola_returns_boolean_full_page():
    from ic_core.ingest import _load_page

    page = _load_page(PAGE_BYTES)
    out = _binarize_page(page, method="sauvola", threshold=127, sauvola_window=25)
    assert out.shape == page.shape
    assert out.dtype == np.bool_
    assert out.any()  # a real page has ink under Sauvola too


def test_binarize_even_sauvola_window_is_bumped_odd():
    # window_size must be odd; an even value should not raise.
    page = np.tile(np.arange(0, 256, 32, dtype=np.uint8), (8, 1))
    out = _binarize_page(page, method="sauvola", threshold=127, sauvola_window=8)
    assert out.shape == page.shape


def test_binarize_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unrecognised threshold method"):
        _binarize_page(
            np.zeros((4, 4), dtype=np.uint8),
            method="adaptive",  # type: ignore[arg-type]
            threshold=127,
            sauvola_window=25,
        )


def test_otsu_changes_glyph_count_invariant_but_not_geometry():
    # Switching the method changes *which pixels* are ink, never the
    # glyph count or the page-coordinate frame (ulx/uly/ncols/nrows) —
    # that's what keeps the UI bbox overlay correct.
    fixed = ingest_page_json(PAGE_BYTES, JSON_BYTES)
    otsu = ingest_page_json(PAGE_BYTES, JSON_BYTES, threshold_method="otsu")
    assert len(fixed) == len(otsu)
    for f, o in zip(fixed, otsu):
        assert (f.id, f.ulx, f.uly, f.ncols, f.nrows) == (
            o.id,
            o.ulx,
            o.uly,
            o.ncols,
            o.nrows,
        )


def test_binarize_page_public_honours_method():
    # The grouping-side entry point routes through the same binariser.
    out = binarize_page(PAGE_BYTES, threshold_method="otsu")
    assert out.dtype == np.bool_
    assert out.any()
