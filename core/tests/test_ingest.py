"""Tests for :mod:`ic_core.ingest`.

Uses the real sample input under ``tests/sample_input/`` so the
test exercises the actual MOTHRA JSON and YOLO formats the
upstream detector produces.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from ic_core.classifier import UNCLASSIFIED
from ic_core.ingest import ingest_page, ingest_page_json, ingest_page_yolo

SAMPLE_DIR = Path(__file__).parent / "sample_input"
PAGE_IMAGE = SAMPLE_DIR / "NZ-Wt MSR-03 109v.png"
JSON_PATH = SAMPLE_DIR / "MOTHRA_NZ-Wt MSR-03 109v_annotations.json"
YOLO_PATH = SAMPLE_DIR / "NZ-Wt MSR-03 109v.txt"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_ingest_page_dispatches_on_suffix():
    json_glyphs = ingest_page(PAGE_IMAGE, JSON_PATH)
    yolo_glyphs = ingest_page(PAGE_IMAGE, YOLO_PATH)
    # Both should produce non-trivial output from the sample page.
    assert len(json_glyphs) > 0
    assert len(yolo_glyphs) > 0


def test_ingest_page_rejects_unknown_extension(tmp_path: Path):
    bogus = tmp_path / "bboxes.csv"
    bogus.write_text("not a real format")
    with pytest.raises(ValueError, match="Unrecognised"):
        ingest_page(PAGE_IMAGE, bogus)


# ---------------------------------------------------------------------------
# JSON ingest
# ---------------------------------------------------------------------------


def test_json_ingest_count_matches_annotations():
    with JSON_PATH.open() as f:
        doc = json.load(f)
    expected = len(doc["annotations"])
    glyphs = ingest_page_json(PAGE_IMAGE, JSON_PATH)
    assert len(glyphs) == expected


def test_json_ingest_preserves_annotation_ids_as_glyph_uuids():
    with JSON_PATH.open() as f:
        doc = json.load(f)
    glyphs = ingest_page_json(PAGE_IMAGE, JSON_PATH)

    expected_ids = [uuid.UUID(a["id"]).hex for a in doc["annotations"]]
    actual_ids = [g.id for g in glyphs]
    assert actual_ids == expected_ids


def test_json_ingest_bbox_coords_match_annotations():
    with JSON_PATH.open() as f:
        doc = json.load(f)
    glyphs = ingest_page_json(PAGE_IMAGE, JSON_PATH)

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
    # The detector's classId is ignored — the user labels glyphs
    # through the API, not at ingest time.
    glyphs = ingest_page_json(PAGE_IMAGE, JSON_PATH)
    assert all(g.class_name == UNCLASSIFIED for g in glyphs)
    assert all(g.confidence == 0.0 for g in glyphs)
    assert all(g.id_state_manual is False for g in glyphs)


def test_json_ingest_is_idempotent_in_id_space():
    # The whole point of preserving the JSON id field: re-ingesting
    # produces the same glyph ids in the same positions.
    a = ingest_page_json(PAGE_IMAGE, JSON_PATH)
    b = ingest_page_json(PAGE_IMAGE, JSON_PATH)
    assert [g.id for g in a] == [g.id for g in b]


# ---------------------------------------------------------------------------
# YOLO ingest
# ---------------------------------------------------------------------------


def test_yolo_ingest_count_matches_lines():
    lines = [
        line
        for line in YOLO_PATH.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    glyphs = ingest_page_yolo(PAGE_IMAGE, YOLO_PATH)
    assert len(glyphs) == len(lines)


def test_yolo_ingest_assigns_fresh_uuids():
    a = ingest_page_yolo(PAGE_IMAGE, YOLO_PATH)
    b = ingest_page_yolo(PAGE_IMAGE, YOLO_PATH)
    # YOLO carries no stable ids — two runs should produce
    # different glyph UUIDs.
    assert all(len(g.id) == 32 for g in a)
    assert {g.id for g in a}.isdisjoint({g.id for g in b})


def test_yolo_ingest_first_glyph_matches_yolo_geometry():
    # First line: 0 0.292332 0.183886 0.051083 0.033496
    with PILImage.open(PAGE_IMAGE) as im:
        img_w, img_h = im.size

    glyphs = ingest_page_yolo(PAGE_IMAGE, YOLO_PATH)
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
    glyphs = ingest_page_json(PAGE_IMAGE, JSON_PATH)
    g = glyphs[0]
    arr = g.to_array()
    assert arr.dtype == np.bool_
    assert arr.shape == (g.nrows, g.ncols)


def test_crop_has_some_foreground_pixels():
    # Every annotated bbox should contain at least one ink pixel on
    # a real chant page. If a glyph comes back fully white, either
    # the threshold or the bbox alignment is wrong.
    glyphs = ingest_page_json(PAGE_IMAGE, JSON_PATH)
    total_fg = sum(int(g.to_array().sum()) for g in glyphs)
    assert total_fg > 0
