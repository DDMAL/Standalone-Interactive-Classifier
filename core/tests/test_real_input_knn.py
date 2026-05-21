"""End-to-end KNN tests against the real sample manuscript page.

Four scenarios:

* **smoke** — ingest the MOTHRA page (filtered to ``classId == 2``),
  train on the legacy GameraXML database, run a full correction
  stage, and assert basic invariants on the result (no crash, all
  glyphs accounted for, confidences in range, no class collapse,
  predicted labels live in the known vocabulary).
* **determinism** — run the smoke pipeline twice and assert the
  predictions match byte-for-byte.
* **5-fold accuracy** (``-m slow``) — stratified 5-fold cross-validation
  over the GameraXML database. MOTHRA doesn't carry per-glyph neume
  labels, so accuracy has to be measured on the only data source
  that does. Marked slow because it retrains 5 times.
* **LOO accuracy** (env-gated) — leave-one-out on a stratified
  subset of the database. ``IC_RUN_LOO=1`` to enable; the subset
  size defaults to 50 glyphs and is overridable via
  ``IC_LOO_LIMIT``.
  
cd core/ic_core && uv run pytest ../tests/test_real_input_knn.py -v
IC_RUN_LOO=1 IC_LOO_LIMIT=200 uv run pytest ../tests/test_real_input_knn.py::test_xml_db_loo_accuracy -v -s
uv run python ../tests/sample_input/helpers/visualize.py

"""
from __future__ import annotations

import csv
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from ic_core.classifier import InteractiveClassifier
from ic_core.glyph import Glyph
from ic_core.io_xml import load_glyphs
from sample_input.helpers.evaluate import (
    TRAINING_XML_PATH,
    classify_page,
    ingest_glyphs_to_classify,
)

CSV_VOCAB_PATH = (
    Path(__file__).parent / "sample_input" / "csv-square_notation_neume_level_newest.csv"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def training_db() -> list[Glyph]:
    """The legacy GameraXML training database, loaded once per module."""
    return load_glyphs(TRAINING_XML_PATH)


@pytest.fixture(scope="module")
def page_glyphs() -> list[Glyph]:
    """MOTHRA page glyphs filtered to ``classId == 2``, loaded once per module."""
    return ingest_glyphs_to_classify()


@pytest.fixture(scope="module")
def vocab(training_db) -> set[str]:
    """Union of class labels from the CSV vocab + the GameraXML training DB.

    Predictions can only emit labels seen during training, so the
    DB classes alone would suffice for the assertion — but checking
    against the CSV too catches cases where the training DB drifts
    away from the canonical neume vocabulary.
    """
    db_labels = {g.class_name for g in training_db}
    csv_labels: set[str] = set()
    with open(CSV_VOCAB_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("classification") or "").strip()
            if label:
                csv_labels.add(label)
    return db_labels | csv_labels


# ---------------------------------------------------------------------------
# 1. Smoke — real page through the full pipeline
# ---------------------------------------------------------------------------


def test_real_page_smoke(page_glyphs, training_db, vocab):
    classified, classifier = classify_page()

    # Every input glyph is returned (none dropped).
    assert len(classified) == len(page_glyphs)

    # UUIDs preserved across the round trip (algorithm invariant #6).
    assert {g.id for g in classified} == {g.id for g in page_glyphs}

    # Every glyph got a real prediction — none left UNCLASSIFIED.
    assert all(g.class_name != "UNCLASSIFIED" for g in classified)

    # Confidence sits strictly inside the documented (0, 1] interval.
    assert all(0.0 < g.confidence <= 1.0 for g in classified)

    # No class collapse: the classifier didn't assign every glyph the
    # same label. (44/86/6 is the classId split, but inside classId=2
    # we expect punctum/clivis/etc. — at least 2 distinct labels.)
    predicted = {g.class_name for g in classified}
    assert len(predicted) >= 2

    # Every predicted label lives in the known vocabulary.
    assert predicted.issubset(vocab), (
        f"Unexpected labels outside vocabulary: {predicted - vocab}"
    )

    # Classifier reports it was trained on the assembled pool.
    assert classifier.is_trained
    assert classifier.training_size > 0

    # Predicted-class histogram — printed under `-v` for triaging a
    # regression (e.g. "punctum predictions dropped from 60 to 4
    # between commits"). Piggybacks on the smoke run so we don't
    # re-ingest+classify just to log counts.
    counts = Counter(g.class_name for g in classified)
    print("\nPredicted-class histogram:")
    for name, n in counts.most_common():
        print(f"  {n:4d}  {name}")


# ---------------------------------------------------------------------------
# 2. Determinism — two runs produce identical predictions
# ---------------------------------------------------------------------------


def test_real_page_determinism():
    a, _ = classify_page()
    b, _ = classify_page()

    a_by_id = {g.id: (g.class_name, g.confidence) for g in a}
    b_by_id = {g.id: (g.class_name, g.confidence) for g in b}
    assert a_by_id == b_by_id


# ---------------------------------------------------------------------------
# 3. Stratified 5-fold accuracy on the GameraXML training DB
# ---------------------------------------------------------------------------


def _stratified_folds(
    glyphs: list[Glyph], n_splits: int, seed: int = 0
) -> list[list[int]]:
    """Assign each glyph an index in [0, n_splits) round-robin per class.

    Classes with fewer than ``n_splits`` members are dropped — they
    can't be stratified across folds without leaving some empty.
    Returns a list of fold-index lists; ``folds[i]`` is the list of
    glyph indices assigned to fold ``i``.
    """
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(glyphs):
        by_class[g.class_name].append(i)

    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for cls, indices in by_class.items():
        if len(indices) < n_splits:
            continue  # excluded — too rare to stratify
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for offset, idx in enumerate(shuffled):
            folds[offset % n_splits].append(idx)
    return folds


@pytest.mark.slow
def test_xml_db_5fold_accuracy(training_db):
    n_splits = 5
    folds = _stratified_folds(training_db, n_splits)

    # Total covered glyphs (excluding tail classes with < n_splits).
    total = sum(len(f) for f in folds)
    assert total > 0, "stratifier dropped all glyphs — fixture is too small"

    correct = 0
    seen = 0
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = [j for k in range(n_splits) if k != i for j in folds[k]]
        train = [training_db[j] for j in train_idx]
        test = [training_db[j] for j in test_idx]

        clf = InteractiveClassifier(k=1).fit(train)
        preds = clf.predict_many(test)
        for g, p in zip(test, preds):
            seen += 1
            if p.class_name == g.class_name:
                correct += 1

    accuracy = correct / seen
    print(
        f"\n[5-fold] tested {seen} glyphs across {n_splits} folds, "
        f"accuracy={accuracy:.4f}"
    )
    # Floor calibrated against the first green run (~0.95) with a
    # 10-point margin for jitter from the shuffle seed. A real
    # regression will drop accuracy well below this band.
    assert accuracy >= 0.85, f"k=1 5-fold accuracy below 0.85: {accuracy:.4f}"


# ---------------------------------------------------------------------------
# 4. LOO accuracy on a stratified subset (opt-in via env)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("IC_RUN_LOO") != "1",
    reason="LOO is opt-in: set IC_RUN_LOO=1 to run",
)
def test_xml_db_loo_accuracy(training_db):
    # Cap the subset so the test stays runnable: full N=2221 means
    # 2221 refits, which is slow even with cached features. Default
    # 50 is enough to catch a regression; bump via IC_LOO_LIMIT.
    limit = int(os.environ.get("IC_LOO_LIMIT", "50"))

    by_class: dict[str, list[Glyph]] = defaultdict(list)
    for g in training_db:
        by_class[g.class_name].append(g)
    # Drop singletons — leaving one out would leave the class with no
    # representative and the prediction can't possibly be correct.
    eligible = [g for cls, gs in by_class.items() if len(gs) >= 2 for g in gs]

    rng = random.Random(0)
    rng.shuffle(eligible)
    subset = eligible[:limit]
    assert subset

    # Index the full DB by id for O(1) "everyone except this glyph" lookup.
    correct = 0
    for held_out in subset:
        train = [g for g in training_db if g.id != held_out.id]
        clf = InteractiveClassifier(k=1).fit(train)
        pred = clf.predict(held_out)
        if pred.class_name == held_out.class_name:
            correct += 1

    accuracy = correct / len(subset)
    print(f"\n[LOO] tested {len(subset)} glyphs, accuracy={accuracy:.4f}")
    assert accuracy >= 0.70, f"k=1 LOO accuracy below 0.70: {accuracy:.4f}"


