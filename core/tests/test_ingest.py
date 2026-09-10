"""Tests for :mod:`ic_core.ingest`.

Uses the real sample input under ``core/data/test/`` so the test
exercises the actual MOTHRA JSON and YOLO formats the upstream
detector produces. Ingest now consumes raw bytes (not paths), so
fixtures are read once at module load and passed by value to each
call.
"""
from __future__ import annotations

import io
import json
import uuid

import numpy as np
import pytest
from PIL import Image as PILImage

from ic_core.classifier import UNCLASSIFIED
from ic_core.ingest import (
    SAUVOLA_K,
    SAUVOLA_WINDOW_SIZE,
    _otsu_threshold,
    _sauvola_mask,
    _unwrap_page,
    binarize_array,
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
# Binarisation methods
# ---------------------------------------------------------------------------


def test_default_method_is_global_backcompat():
    # The default path must stay byte-for-byte what it produced before
    # methods existed, so existing callers / fixtures don't shift.
    default = ingest_page_json(PAGE_BYTES, JSON_BYTES)
    explicit = ingest_page_json(PAGE_BYTES, JSON_BYTES, method="global")
    assert [g.image_rle for g in default] == [g.image_rle for g in explicit]


def test_global_and_threshold_arg_still_apply():
    # The legacy `threshold` knob still drives method="global".
    loose = ingest_page_json(PAGE_BYTES, JSON_BYTES, method="global", threshold=60)
    tight = ingest_page_json(PAGE_BYTES, JSON_BYTES, method="global", threshold=200)
    # A higher cutoff calls more pixels ink, so total foreground grows.
    assert sum(int(g.to_array().sum()) for g in tight) > sum(
        int(g.to_array().sum()) for g in loose
    )


def test_sauvola_differs_from_global():
    g = ingest_page_json(PAGE_BYTES, JSON_BYTES, method="global")
    s = ingest_page_json(PAGE_BYTES, JSON_BYTES, method="sauvola")
    assert [x.image_rle for x in g] != [x.image_rle for x in s]


@pytest.mark.parametrize("method", ["global", "otsu", "sauvola"])
def test_binarize_array_returns_full_page_bool_mask(method):
    from ic_core.ingest import _load_page

    page = _load_page(PAGE_BYTES)
    mask = binarize_array(page, method=method)
    assert mask.dtype == np.bool_
    assert mask.shape == page.shape
    assert mask.any()  # a real chant page is never all-background


def test_binarize_array_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unrecognised binarization method"):
        binarize_array(np.zeros((4, 4), dtype=np.uint8), method="bogus")  # type: ignore[arg-type]


def test_otsu_uniform_image_does_not_raise():
    # threshold_otsu raises on a single-valued image; binarize_array must
    # fall back to the global cutoff instead of blowing up.
    bright = np.full((20, 20), 200, dtype=np.uint8)  # all background (>127)
    dark = np.full((20, 20), 10, dtype=np.uint8)  # all foreground (<=127)
    assert binarize_array(bright, method="otsu").sum() == 0
    assert binarize_array(dark, method="otsu").all()


@pytest.mark.parametrize("method", ["global", "otsu", "sauvola"])
def test_per_glyph_mask_matches_full_page_mask(method):
    # The no-desync invariant: a glyph's mask is exactly the slice of the
    # full-page mask (what manual grouping uses) at its bbox — for every
    # method. This is what whole-page binarisation buys us; per-crop
    # Sauvola would break it.
    page_mask = binarize_page(PAGE_BYTES, method=method)
    h, w = page_mask.shape
    glyphs = ingest_page_json(PAGE_BYTES, JSON_BYTES, method=method)

    checked = 0
    for g in glyphs:
        # Skip edge-clamped glyphs — their declared bbox runs past the
        # page, so a naive slice wouldn't line up.
        if g.ulx < 0 or g.uly < 0 or g.ulx + g.ncols > w or g.uly + g.nrows > h:
            continue
        sub = page_mask[g.uly : g.uly + g.nrows, g.ulx : g.ulx + g.ncols]
        assert np.array_equal(sub, g.to_array())
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# Memory-bounded threshold helpers
# ---------------------------------------------------------------------------
#
# These exist only to keep peak memory off the page-area curve (a full-page
# Sauvola on a 32 MP folio peaks over 1.5 GB and OOM-kills IC's 1 GiB
# container). The whole point is that they change *nothing* about the output,
# so that equality is what these tests pin.


def _naive_sauvola(page):
    """The straightforward whole-page call these helpers must reproduce."""
    from skimage.filters import threshold_sauvola

    return page < threshold_sauvola(
        page, window_size=SAUVOLA_WINDOW_SIZE, k=SAUVOLA_K
    )


def test_sauvola_strips_match_whole_page():
    # A tiny budget forces many strips over the real test page; every one of
    # them must agree with the whole-page result, including the rows either
    # side of each strip seam and the reflect-padded page edges.
    page = np.asarray(PILImage.open(io.BytesIO(PAGE_BYTES)).convert("L"))
    got = _sauvola_mask(
        page,
        window_size=SAUVOLA_WINDOW_SIZE,
        k=SAUVOLA_K,
        budget_bytes=64 * 1024,  # ~1-2 rows per strip: maximum seam exposure
    )
    assert np.array_equal(got, _naive_sauvola(page))


def test_sauvola_single_strip_path_matches_whole_page():
    # The other branch: a page that fits the budget skips the strip
    # bookkeeping entirely and must still match.
    page = np.asarray(PILImage.open(io.BytesIO(PAGE_BYTES)).convert("L"))
    got = _sauvola_mask(
        page, window_size=SAUVOLA_WINDOW_SIZE, k=SAUVOLA_K, budget_bytes=1 << 30
    )
    assert np.array_equal(got, _naive_sauvola(page))


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (1, 500), (500, 1), (60, 40)])
def test_sauvola_strips_handle_degenerate_shapes(shape):
    # Pages smaller than the window, and single-row/column arrays, must not
    # crash or diverge — the halo arithmetic has to clamp, not wrap.
    rng = np.random.default_rng(0)
    page = rng.integers(0, 256, size=shape, dtype=np.uint8)
    got = _sauvola_mask(
        page, window_size=SAUVOLA_WINDOW_SIZE, k=SAUVOLA_K, budget_bytes=4096
    )
    assert got.shape == page.shape
    assert np.array_equal(got, _naive_sauvola(page))


def test_otsu_threshold_matches_skimage():
    from skimage.filters import threshold_otsu

    page = np.asarray(PILImage.open(io.BytesIO(PAGE_BYTES)).convert("L"))
    assert _otsu_threshold(page) == float(threshold_otsu(page))


def test_otsu_threshold_falls_back_for_non_uint8():
    # The bincount shortcut is uint8-only; other dtypes must still work by
    # handing the array to skimage.
    from skimage.filters import threshold_otsu

    rng = np.random.default_rng(1)
    page = rng.random((64, 64))  # float64
    assert _otsu_threshold(page) == float(threshold_otsu(page))
