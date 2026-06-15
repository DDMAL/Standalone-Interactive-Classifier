"""Tests for the :func:`ic_core.io_xml.write_glyphs` / ``dumps_glyphs`` writer.

The reader tests live in :mod:`tests.test_io_xml`; this file focuses
on round-trip semantics and on the schema details the migration plan
calls out as load-bearing (for example ``state`` attribute
discrimination, confidence formatting, input ordering, and
transient-prefix exclusion).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from lxml import etree

from ic_core.classifier import UNCLASSIFIED, filter_parts
from ic_core.glyph import (
    CATEGORY_NEUMES,
    CATEGORY_STAVES,
    CATEGORY_TEXT,
    Glyph,
)
from ic_core.image import array_to_rle
from ic_core.io_xml import dumps_glyphs, load_glyphs, write_glyphs


def _make_glyph(
    *,
    class_name: str = UNCLASSIFIED,
    id_state_manual: bool = False,
    confidence: float = 0.0,
    ulx: int = 0,
    uly: int = 0,
    category: str = CATEGORY_NEUMES,
    shape: tuple[int, int] = (4, 4),
) -> Glyph:
    arr = np.ones(shape, dtype=bool)
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


def test_dumps_returns_xml_with_correct_root():
    blob = dumps_glyphs([_make_glyph(class_name="A", id_state_manual=True, confidence=1.0)])
    root = etree.fromstring(blob)
    assert root.tag == "gamera-database"
    assert root.get("version") == "2.0"
    assert root[0].tag == "glyphs"


def test_state_attribute_reflects_glyph_kind():
    manual = _make_glyph(class_name="A", id_state_manual=True, confidence=1.0)
    auto = _make_glyph(class_name="A", id_state_manual=False, confidence=0.42)
    unclassified = _make_glyph(class_name=UNCLASSIFIED)

    root = etree.fromstring(dumps_glyphs([manual, auto, unclassified]))
    states = [g.find("ids").get("state") for g in root.iterfind(".//glyph")]

    assert states == ["MANUAL", "AUTOMATIC", "UNCLASSIFIED"]


def test_unclassified_id_name_reflects_category():
    """UNCLASSIFIED Text/Staves glyphs surface a concrete ``<id>`` name
    (``text`` / ``staff``); an UNCLASSIFIED Neume keeps the sentinel."""
    text = _make_glyph(class_name=UNCLASSIFIED, category=CATEGORY_TEXT)
    staff = _make_glyph(class_name=UNCLASSIFIED, category=CATEGORY_STAVES)
    neume = _make_glyph(class_name=UNCLASSIFIED, category=CATEGORY_NEUMES)

    root = etree.fromstring(dumps_glyphs([text, staff, neume]))
    names = [g.find(".//id").get("name") for g in root.iterfind(".//glyph")]

    assert names == ["text", "staff", UNCLASSIFIED]


def test_classified_id_name_ignores_category():
    """A real class name wins regardless of the glyph's category."""
    g = _make_glyph(class_name="A", id_state_manual=True, confidence=1.0,
                    category=CATEGORY_TEXT)
    root = etree.fromstring(dumps_glyphs([g]))
    assert root.find(".//id").get("name") == "A"


def test_geometry_attributes_round_trip(tmp_path: Path):
    src = _make_glyph(
        class_name="A",
        id_state_manual=True,
        confidence=1.0,
        ulx=100,
        uly=200,
        shape=(15, 25),
    )
    out_path = tmp_path / "out.xml"
    write_glyphs([src], out_path)

    loaded = load_glyphs(out_path)
    assert len(loaded) == 1
    g = loaded[0]
    assert (g.ulx, g.uly, g.nrows, g.ncols) == (100, 200, 15, 25)
    assert g.class_name == "A"
    assert g.id_state_manual is True
    assert g.confidence == 1.0


def test_rle_payload_round_trips():
    # Heterogeneous mask to make sure the RLE survives serialisation
    # (the writer emits the raw RLE string; the reader splits/parses it).
    arr = np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    g = Glyph.new(
        class_name="A",
        image_rle=array_to_rle(arr),
        ncols=4,
        nrows=3,
        ulx=0,
        uly=0,
        id_state_manual=True,
        confidence=1.0,
    )

    blob = dumps_glyphs([g])
    root = etree.fromstring(blob)
    rle_in_xml = (root.findtext(".//data") or "").strip()
    # Whitespace-normalise both sides for comparison.
    assert re.sub(r"\s+", " ", rle_in_xml) == re.sub(r"\s+", " ", g.image_rle)


def test_filter_parts_then_write_excludes_transients(tmp_path: Path):
    keep = _make_glyph(class_name="A", id_state_manual=True, confidence=1.0)
    drop_group = _make_glyph(class_name="_group.foo")
    drop_delete = _make_glyph(class_name="_delete")

    out_path = tmp_path / "out.xml"
    write_glyphs(filter_parts([keep, drop_group, drop_delete]), out_path)

    names = [g.class_name for g in load_glyphs(out_path)]
    assert names == ["A"]


def test_confidence_serialised_with_six_decimal_places():
    g = _make_glyph(class_name="A", id_state_manual=False, confidence=0.123)
    blob = dumps_glyphs([g])
    root = etree.fromstring(blob)
    conf = root.find(".//id").get("confidence")
    assert conf == "0.123000"


def test_features_block_is_emitted_with_version_and_named_children():
    """The writer embeds a ``<features>`` block per glyph carrying the
    current ``FEATURE_VERSION`` plus one ``<feature name=...>`` per
    logical feature, mirroring the Square_notation fixture shape."""
    from ic_core.features import FEATURE_VERSION, LOGICAL_FEATURES

    g = _make_glyph(class_name="A", id_state_manual=True, confidence=1.0)
    root = etree.fromstring(dumps_glyphs([g]))

    feats = root.find(".//glyph/features")
    assert feats is not None, "expected one <features> block per glyph"
    assert feats.get("version") == FEATURE_VERSION
    assert feats.get("scaling") == "1.0"

    names = [f.get("name") for f in feats.findall("feature")]
    assert names == [name for name, _ in LOGICAL_FEATURES]

    # Multi-dim features have ``dim`` space-separated floats inside
    # a single element — verify with ``volume16regions``.
    vol16 = feats.find('feature[@name="volume16regions"]')
    assert vol16 is not None
    assert len((vol16.text or "").split()) == 16


def test_writer_preserves_input_order(tmp_path: Path):
    glyphs = [
        _make_glyph(class_name="A", id_state_manual=True, confidence=1.0, ulx=0),
        _make_glyph(class_name="B", id_state_manual=True, confidence=1.0, ulx=10),
        _make_glyph(class_name="C", id_state_manual=True, confidence=1.0, ulx=20),
    ]
    out_path = tmp_path / "ordered.xml"
    write_glyphs(glyphs, out_path)

    loaded = load_glyphs(out_path)
    assert [g.class_name for g in loaded] == ["A", "B", "C"]
    assert [g.ulx for g in loaded] == [0, 10, 20]
