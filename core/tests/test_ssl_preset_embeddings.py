"""Tests for the Hufnagel preset's companion SSL embeddings.

Confirms the ``.ssl_embeddings.npz`` file shipped alongside
``Hufnagel.xml`` is in sync (same length, same document order) with the
preset it belongs to, and that ``SSLFusionClassifier`` can train on
preset-sourced glyphs carrying only a precomputed ``ssl_embedding`` (no
``image_gray_b64``, no live model pass) via
``ic_core.ssl_preset_embeddings.attach_ssl_embeddings``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sklearn = pytest.importorskip("sklearn")

from ic_core.classifier import UNCLASSIFIED, run_correction_stage
from ic_core.io_xml import load_glyphs
from ic_core.ssl_classifier import SSLFusionClassifier
from ic_core.ssl_preset_embeddings import (
    attach_ssl_embeddings,
    has_ssl_embeddings,
    load_ssl_embeddings,
    match_glyphs_to_source_pages,
)

PRESET_XML = Path(__file__).parent.parent / "data" / "presets" / "Hufnagel.xml"
TRAIN_DIR = Path(__file__).parent.parent / "data" / "train"
HUFNAGEL_SOURCE_PAGES = [
    TRAIN_DIR / "hufnagel_example_826dd1b4.png",
    TRAIN_DIR / "hufnagel_example_a77ec16f.png",
    TRAIN_DIR / "hufnagel_example_fbed8126.png",
]


def test_hufnagel_preset_has_embeddings():
    assert has_ssl_embeddings(PRESET_XML)


def test_embeddings_length_matches_preset_glyph_count():
    glyphs = load_glyphs(PRESET_XML)
    embeddings = load_ssl_embeddings(PRESET_XML)
    assert embeddings is not None
    assert embeddings.shape[0] == len(glyphs)


def test_attach_ssl_embeddings_sets_field_on_every_glyph():
    glyphs = load_glyphs(PRESET_XML)
    embeddings = load_ssl_embeddings(PRESET_XML)

    attached = attach_ssl_embeddings(glyphs, embeddings)

    assert all(g.ssl_embedding is not None for g in attached)
    assert len(attached[0].ssl_embedding) == embeddings.shape[1]
    # Attaching doesn't touch anything glyph identity/labelling depends on.
    assert [g.id for g in attached] == [g.id for g in glyphs]
    assert [g.class_name for g in attached] == [g.class_name for g in glyphs]


def test_ssl_fusion_classifier_trains_on_preset_embeddings_alone():
    """No image_gray_b64, no live extractor pass -- just precomputed vectors."""
    glyphs = load_glyphs(PRESET_XML)
    embeddings = load_ssl_embeddings(PRESET_XML)
    training_glyphs = attach_ssl_embeddings(glyphs, embeddings)

    assert all(g.image_gray_b64 is None for g in training_glyphs)

    split = int(len(training_glyphs) * 0.8)
    fit_pool = training_glyphs[:split]
    held_out = training_glyphs[split:]
    query_glyphs = [
        g.classify_automatic(UNCLASSIFIED, 0.0) for g in held_out
    ]

    new_glyphs, classifier = run_correction_stage(
        query_glyphs,
        fit_pool,
        classifier_factory=SSLFusionClassifier,
    )

    assert isinstance(classifier, SSLFusionClassifier)
    assert classifier.training_size == len(fit_pool)
    assert len(new_glyphs) == len(query_glyphs)
    for g in new_glyphs:
        assert g.class_name != UNCLASSIFIED


def test_match_glyphs_to_source_pages_recovers_every_hufnagel_glyph():
    """Same recovery this preset's embeddings were generated from (see
    core/scripts/generate_hufnagel_ssl_embeddings.py), exercised as the
    general-purpose path an uploaded (not just preset) GameraXML file
    would go through if its source pages are supplied alongside it.
    """
    glyphs = load_glyphs(PRESET_XML)
    pages = [np.array(Image.open(p).convert("L")) for p in HUFNAGEL_SOURCE_PAGES]

    matched = match_glyphs_to_source_pages(glyphs, pages)

    assert len(matched) == len(glyphs)
    assert all(g.image_gray_b64 is not None for g in matched)
    # Doesn't touch identity/labelling.
    assert [g.id for g in matched] == [g.id for g in glyphs]
    assert [g.class_name for g in matched] == [g.class_name for g in glyphs]


def test_match_glyphs_to_source_pages_leaves_unmatched_glyphs_alone():
    glyphs = load_glyphs(PRESET_XML)
    blank_page = np.full((3000, 3000), 255, dtype=np.uint8)

    matched = match_glyphs_to_source_pages(glyphs, [blank_page])

    assert len(matched) == len(glyphs)
    assert all(g.image_gray_b64 is None for g in matched)
