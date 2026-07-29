"""Smoke test for the optional SSL+HC fused classifier backend.

Two things are asserted:

1. **Zero regression on the default path.** ``run_correction_stage``
   without ``classifier_factory`` behaves exactly as before this
   backend was added -- covered by the existing ``test_classifier.py``
   / ``test_real_input_knn.py`` suites, which must keep passing
   unchanged (see ``git log`` for this branch: those files were never
   touched).
2. **The new path actually works end-to-end** on a real manuscript
   page: ingest real glyphs from the Hufnagel fixture data (with
   ``image_gray_b64`` populated so the SSL extractor has real pixels
   to work with, not just the binary silhouette), fit
   ``SSLFusionClassifier``, and confirm it produces sane predictions
   through the same ``run_correction_stage`` entry point every other
   classifier goes through.

Requires the ``ssl`` extra (``pip install ic-core[ssl]``); skipped
entirely if scikit-learn/torch/transformers aren't importable.  Also
skipped if no checkpoint is configured via ``IC_SSL_CHECKPOINT`` --
without it, the test would need network access to download a
plain pretrained backbone, which isn't guaranteed in CI.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sklearn = pytest.importorskip("sklearn")
pytest.importorskip("torch")
pytest.importorskip("transformers")

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "scripts"))

from ic_core.classifier import UNCLASSIFIED, run_correction_stage
from ic_core.image import grayscale_array_to_png_base64
from ic_core.ssl_classifier import SSLFusionClassifier

TRAIN_DIR = _HERE.parent / "data" / "train"
CSV_PATH = TRAIN_DIR / "hufnagel_annotations_826dd1b4.csv"
PAGE_PATH = TRAIN_DIR / "hufnagel_example_826dd1b4.png"

_SSL_CHECKPOINT = os.environ.get("IC_SSL_CHECKPOINT")

pytestmark = pytest.mark.skipif(
    not _SSL_CHECKPOINT,
    reason="Set IC_SSL_CHECKPOINT to a fine-tuned checkpoint directory to run this test.",
)


def _attach_real_crops(glyphs, page_color: np.ndarray):
    """Slice the real page at each glyph's bbox and attach image_gray_b64.

    ``page_color`` is RGB, matching what production ingest now stores
    (the DINO SimCLR checkpoint was trained on real colour crops, not
    greyscale-replicated-to-RGB ones -- see ``ic_core.ingest._load_page_color``).
    """
    out = []
    for g in glyphs:
        crop = page_color[g.uly : g.uly + g.nrows, g.ulx : g.ulx + g.ncols]
        out.append(replace(g, image_gray_b64=grayscale_array_to_png_base64(crop)))
    return out


@pytest.fixture(scope="module")
def real_glyphs():
    """Real, labelled Hufnagel glyphs with real-pixel crops attached."""
    from convert_hufnagel_csv import _glyphs_for_pair

    glyphs = _glyphs_for_pair(CSV_PATH, PAGE_PATH)
    assert len(glyphs) > 20, "expected a real, non-trivial glyph set"

    with Image.open(PAGE_PATH) as im:
        page_color = np.asarray(im.convert("RGB"))

    return _attach_real_crops(glyphs, page_color)


def test_ssl_fusion_classifier_end_to_end(real_glyphs):
    """SSLFusionClassifier trains and predicts through run_correction_stage."""
    n = len(real_glyphs)
    split = int(n * 0.8)
    training_glyphs = real_glyphs[:split]  # already id_state_manual=True
    true_labels = {g.id: g.class_name for g in real_glyphs[split:]}

    # Build the "not yet classified" working set from the held-out tail.
    query_glyphs = [
        replace(g, class_name=UNCLASSIFIED, id_state_manual=False, confidence=0.0)
        for g in real_glyphs[split:]
    ]

    new_glyphs, classifier = run_correction_stage(
        query_glyphs,
        training_glyphs,
        classifier_factory=lambda: SSLFusionClassifier(checkpoint=_SSL_CHECKPOINT),
    )

    assert isinstance(classifier, SSLFusionClassifier)
    assert classifier.is_trained
    assert classifier.training_size == len(training_glyphs)
    assert len(new_glyphs) == len(query_glyphs)

    known_vocab = set(classifier.classes)
    correct = 0
    for g in new_glyphs:
        assert g.class_name != UNCLASSIFIED, "every query glyph should get a real prediction"
        assert g.class_name in known_vocab
        assert 0.0 < g.confidence <= 1.0
        if g.class_name == true_labels[g.id]:
            correct += 1

    accuracy = correct / len(new_glyphs)
    chance_level = 1.0 / len(known_vocab)
    print(
        f"\nSSLFusionClassifier smoke-test accuracy: {accuracy:.2%} "
        f"({correct}/{len(new_glyphs)}), chance level {chance_level:.2%} "
        f"over {len(known_vocab)} classes"
    )
    # NOT a rigorous accuracy claim: this is a single small page split
    # 80/20 (~230 training glyphs from one page, vs. the ~1500+
    # multi-manuscript training set the real ~90%+ numbers in the
    # accompanying report come from). This only confirms the pipeline
    # produces *sane*, meaningfully-better-than-chance predictions
    # rather than silently returning garbage.
    assert accuracy > 2 * chance_level


def test_default_backend_unaffected_by_ssl_addition(real_glyphs):
    """run_correction_stage with no classifier_factory still uses kNN."""
    from ic_core.classifier import InteractiveClassifier

    n = len(real_glyphs)
    split = int(n * 0.8)
    training_glyphs = real_glyphs[:split]
    query_glyphs = [
        replace(g, class_name=UNCLASSIFIED, id_state_manual=False, confidence=0.0)
        for g in real_glyphs[split:]
    ]

    new_glyphs, classifier = run_correction_stage(query_glyphs, training_glyphs)

    assert isinstance(classifier, InteractiveClassifier)
    assert len(new_glyphs) == len(query_glyphs)


def test_fits_with_one_example_per_class(real_glyphs):
    """CalibratedClassifierCV needs >=cv examples per class for stratified
    CV -- unlike the LogisticRegression this replaced, which had no such
    floor. A real session can start with exactly one labelled example per
    class, so this must degrade gracefully (uncalibrated SVC, confidence=1.0)
    rather than raise sklearn's "n_splits cannot be greater than the
    number of members in each class" error.
    """
    two_glyphs = [
        replace(real_glyphs[0], class_name="solo.a", id_state_manual=True),
        replace(real_glyphs[1], class_name="solo.b", id_state_manual=True),
    ]
    query = [replace(g, class_name=UNCLASSIFIED, id_state_manual=False, confidence=0.0)
             for g in real_glyphs[2:5]]

    new_glyphs, classifier = run_correction_stage(
        query, two_glyphs, classifier_factory=lambda: SSLFusionClassifier(checkpoint=_SSL_CHECKPOINT)
    )

    assert classifier.is_calibrated is False
    for g in new_glyphs:
        assert g.class_name in ("solo.a", "solo.b")
        assert g.confidence == 1.0


def test_fits_with_few_examples_per_class_shrinks_cv(real_glyphs):
    """3 examples/class is enough to calibrate (cv shrinks to 3), but would
    have hit the same n_splits error at the default cv=5.
    """
    by_label: dict[str, list] = {}
    for g in real_glyphs:
        by_label.setdefault(g.class_name, []).append(g)
    two_classes = [c for c, gs in by_label.items() if len(gs) >= 4][:2]
    assert len(two_classes) == 2, "fixture should have at least 2 classes with >=4 examples"

    training = [g for c in two_classes for g in by_label[c][:3]]
    query = [
        replace(g, class_name=UNCLASSIFIED, id_state_manual=False, confidence=0.0)
        for c in two_classes for g in by_label[c][3:4]
    ]

    new_glyphs, classifier = run_correction_stage(
        query, training, classifier_factory=lambda: SSLFusionClassifier(checkpoint=_SSL_CHECKPOINT)
    )

    assert classifier.is_calibrated is True
    assert len(new_glyphs) == len(query)
