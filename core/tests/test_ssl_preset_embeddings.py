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

import pytest

sklearn = pytest.importorskip("sklearn")

from ic_core.classifier import UNCLASSIFIED, run_correction_stage
from ic_core.io_xml import load_glyphs
from ic_core.ssl_classifier import SSLFusionClassifier
from ic_core.ssl_preset_embeddings import (
    attach_ssl_embeddings,
    has_ssl_embeddings,
    load_ssl_embeddings,
)

PRESET_XML = Path(__file__).parent.parent / "data" / "presets" / "Hufnagel.xml"


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
