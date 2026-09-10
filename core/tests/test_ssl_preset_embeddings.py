"""Tests for the Hufnagel/Square-notation presets' companion SSL embeddings.

Confirms each ``.ssl_embeddings.npz`` file is in sync (same length, same
document order) with the preset it belongs to, and that
``SSLFusionClassifier`` can train on preset-sourced glyphs carrying only a
precomputed ``ssl_embedding`` (no ``image_gray_b64``, no live model pass)
via ``ic_core.ssl_preset_embeddings.attach_ssl_embeddings``.

Both presets were replaced by their author (Yueqiao Zhang) after this
module was first written -- Hufnagel on 2026-08-05/06 (557 -> 305 glyphs),
Square notation on 2026-07-15 (as ``Square2.xml``, 144 glyphs, to
``Square notation training data 05.08.26.xml``, 249 glyphs) -- and the
embeddings were regenerated to match (see
``core/scripts/generate_hufnagel_ssl_embeddings.py`` /
``generate_square2_ssl_embeddings.py`` for the full per-page provenance).
``PRESET_XML`` / ``SQUARE_XML`` below point at the *current* filenames.

The two "recovers every glyph" tests intentionally check partial, not
100%, coverage: the regenerated embeddings needed several source pages
Yueqiao supplied afterwards, which live only in this maintainer's local
staging area, not in this repo or CI -- so these tests can only exercise
match_glyphs_to_source_pages against the pages that *are* committed under
``core/data/train/`` (three of Hufnagel's ten, one of Square notation's
nine), and assert the exact partial match count those give rather than
full coverage.
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
from ic_core.ssl_extractor import extract_ssl_embeddings
from ic_core.ssl_preset_embeddings import (
    attach_ssl_embeddings,
    has_ssl_embeddings,
    load_ssl_embeddings,
    match_glyphs_to_source_pages,
)

PRESET_XML = (
    Path(__file__).parent.parent
    / "data"
    / "presets"
    / "Hufnagel training data 06.08.26.xml"
)
TRAIN_DIR = Path(__file__).parent.parent / "data" / "train"
HUFNAGEL_SOURCE_PAGES = [
    TRAIN_DIR / "hufnagel_example_826dd1b4.png",
    TRAIN_DIR / "hufnagel_example_a77ec16f.png",
    TRAIN_DIR / "hufnagel_example_fbed8126.png",
]
# Exact match count for the three pages above against the current (305-glyph)
# preset -- see the module docstring for why this isn't all 305.
HUFNAGEL_COMMITTED_PAGES_MATCH_COUNT = 238


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


def test_attach_ssl_embeddings_rejects_non_2d_array():
    """embeddings.npz is user-uploadable -- a malformed file must fail
    clearly here, not surface as an opaque numpy/sklearn error later.
    """
    glyphs = load_glyphs(PRESET_XML)[:3]
    with pytest.raises(ValueError, match="2-D array of floats"):
        attach_ssl_embeddings(glyphs, np.zeros(3 * 768, dtype=np.float32))


def test_attach_ssl_embeddings_rejects_non_finite_values():
    glyphs = load_glyphs(PRESET_XML)[:3]
    embeddings = np.zeros((3, 768), dtype=np.float32)
    embeddings[1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        attach_ssl_embeddings(glyphs, embeddings)


def test_match_glyphs_to_source_pages_recovers_most_hufnagel_glyphs():
    """Partial-coverage check against only the pages committed in this repo
    -- see the module docstring. The full 305/305 recovery (using pages
    supplied outside the repo) is what generate_hufnagel_ssl_embeddings.py
    actually used to build the committed .ssl_embeddings.npz; this test
    exercises the same general-purpose path (an uploaded GameraXML file
    with source pages supplied alongside it) at reduced coverage.
    """
    glyphs = load_glyphs(PRESET_XML)
    pages = [np.array(Image.open(p).convert("L")) for p in HUFNAGEL_SOURCE_PAGES]

    matched = match_glyphs_to_source_pages(glyphs, pages)

    assert len(matched) == len(glyphs)
    n_recovered = sum(1 for g in matched if g.image_gray_b64 is not None)
    assert n_recovered == HUFNAGEL_COMMITTED_PAGES_MATCH_COUNT
    # Doesn't touch identity/labelling, matched or not.
    assert [g.id for g in matched] == [g.id for g in glyphs]
    assert [g.class_name for g in matched] == [g.class_name for g in glyphs]


def test_match_glyphs_to_source_pages_leaves_unmatched_glyphs_alone():
    glyphs = load_glyphs(PRESET_XML)
    blank_page = np.full((3000, 3000), 255, dtype=np.uint8)

    matched = match_glyphs_to_source_pages(glyphs, [blank_page])

    assert len(matched) == len(glyphs)
    assert all(g.image_gray_b64 is None for g in matched)


def test_extract_ssl_embeddings_reuses_precomputed_vectors_without_a_checkpoint():
    """All-precomputed glyphs never need torch/a checkpoint at all."""
    glyphs = load_glyphs(PRESET_XML)[:5]
    embeddings = load_ssl_embeddings(PRESET_XML)[:5]
    attached = attach_ssl_embeddings(glyphs, embeddings)

    result = extract_ssl_embeddings(attached, checkpoint=None)

    assert np.allclose(result, embeddings)


def test_extract_ssl_embeddings_rejects_glyphs_with_no_features_at_all():
    glyphs = load_glyphs(PRESET_XML)[:3]
    assert all(g.ssl_embedding is None and g.image_gray_b64 is None for g in glyphs)

    with pytest.raises(ValueError, match="neither a"):
        extract_ssl_embeddings(glyphs, checkpoint=None)


SQUARE_XML = (
    Path(__file__).parent.parent
    / "data"
    / "presets"
    / "Square notation training data 05.08.26.xml"
)
SQUARE_SOURCE_PAGE = (
    TRAIN_DIR / "Einsiedeln__Stiftsbibliothek__Codex_611_014r.jpg"
)
# Exact match count for the one page above against the current (249-glyph)
# preset, at the module default binarize_threshold (127) -- see the module
# docstring. The old Square2.xml preset needed binarize_threshold=110; this
# replacement preset matches cleanly at the default instead (110 gives 0).
SQUARE_COMMITTED_PAGE_MATCH_COUNT = 52


def test_square_preset_has_embeddings():
    assert has_ssl_embeddings(SQUARE_XML)


def test_square_embeddings_length_matches_preset_glyph_count():
    glyphs = load_glyphs(SQUARE_XML)
    embeddings = load_ssl_embeddings(SQUARE_XML)
    assert embeddings is not None
    assert embeddings.shape[0] == len(glyphs)


def test_match_glyphs_to_source_pages_recovers_some_square_glyphs():
    """Partial-coverage check against only the page committed in this repo
    -- see the module docstring. The full 249/249 recovery (using pages
    supplied outside the repo) is what generate_square2_ssl_embeddings.py
    actually used to build the committed .ssl_embeddings.npz.
    """
    glyphs = load_glyphs(SQUARE_XML)
    page = np.array(Image.open(SQUARE_SOURCE_PAGE).convert("L"))

    matched = match_glyphs_to_source_pages(glyphs, [page])

    assert len(matched) == len(glyphs)
    n_recovered = sum(1 for g in matched if g.image_gray_b64 is not None)
    assert n_recovered == SQUARE_COMMITTED_PAGE_MATCH_COUNT
