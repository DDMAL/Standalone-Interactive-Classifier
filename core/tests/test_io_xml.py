import base64
from pathlib import Path

import numpy as np

from ic_core.image import array_to_rle, rle_to_array
from ic_core.io_xml import load_glyphs

FIXTURE = Path(__file__).parent / "fixtures" / "Square_notation-example_training_data.xml"


def test_load_training_data_glyph_count():
    glyphs = load_glyphs(FIXTURE)
    assert len(glyphs) > 0


def test_glyph_fields_populated():
    glyphs = load_glyphs(FIXTURE)
    g = glyphs[0]
    assert g.class_name == "neume.oblique3"
    assert g.ulx == 3030
    assert g.uly == 697
    assert g.ncols == 117
    assert g.nrows == 115
    assert g.id_state_manual is True
    assert g.confidence == 1.0


def test_uuid_is_unique_and_hex():
    glyphs = load_glyphs(FIXTURE)
    ids = [g.id for g in glyphs]
    assert all(len(i) == 32 for i in ids)
    hex_chars = set("0123456789abcdef")
    assert all(set(i) <= hex_chars for i in ids)
    assert len(set(ids)) == len(ids)


def test_rle_decodes_to_correct_dimensions():
    glyphs = load_glyphs(FIXTURE)
    g = glyphs[0]
    arr = g.to_array()
    assert arr.shape == (g.nrows, g.ncols)
    assert arr.dtype == np.bool_


def test_rle_round_trip():
    arr = np.array(
        [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=bool,
    )
    rle = array_to_rle(arr)
    back = rle_to_array(rle, 4, 4)
    np.testing.assert_array_equal(arr, back)


def test_rle_round_trip_starts_black():
    arr = np.array([[1, 1, 0, 0]], dtype=bool)
    rle = array_to_rle(arr)
    assert rle.split()[0] == "0"
    back = rle_to_array(rle, 4, 1)
    np.testing.assert_array_equal(arr, back)


def test_to_base64_png_is_ascii_str():
    glyphs = load_glyphs(FIXTURE)
    s = glyphs[0].to_base64_png()
    assert isinstance(s, str)
    raw = base64.b64decode(s)
    assert raw.startswith(b"\x89PNG")
