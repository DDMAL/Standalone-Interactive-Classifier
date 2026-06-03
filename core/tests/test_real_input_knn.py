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
uv run python ../scripts/visualize.py

"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path


def _ensure_training_xml() -> None:
    """Materialise ``IC_TRAINING_XML`` from committed CSV+PNG pairs on demand.

    ``paths.TRAINING_XML`` defaults to
    ``core/data/derived/Hufnagel_training_data.xml``, which is gitignored
    and absent on a clean checkout. Regenerate it from the committed
    pairs in ``core/data/train/`` via
    :func:`convert_hufnagel_csv.convert_batch`, write into a session
    tempdir, and point ``IC_TRAINING_XML`` there.

    Must run *before* the ``from evaluate import …`` below: ``evaluate``
    captures ``paths.TRAINING_XML`` as a function default at import
    time, so a later override wouldn't reach
    :func:`evaluate.classify_page`.

    Respects existing setup: an explicit ``IC_TRAINING_XML`` env var or
    a pre-generated default path on disk both short-circuit the regen.
    """
    if os.environ.get("IC_TRAINING_XML"):
        return

    import paths

    if paths.TRAINING_XML.exists():
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="ic-training-"))
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    out_xml = tmp_dir / paths.TRAINING_XML.name

    print(
        f"[test_real_input_knn] Generating {out_xml.name} from "
        f"{paths.TRAIN_DIR} (set IC_TRAINING_XML to skip)"
    )
    from convert_hufnagel_csv import convert_batch

    convert_batch(paths.TRAIN_DIR, out_xml)

    os.environ["IC_TRAINING_XML"] = str(out_xml)
    # paths was loaded above; patch the cached module so the
    # ``from paths import TRAINING_XML`` below — and evaluate's own
    # ``from paths import …`` — bind to the regenerated path.
    paths.TRAINING_XML = out_xml


_ensure_training_xml()

import contextlib
import csv
import random
from collections import Counter, defaultdict

import pytest

from ic_core.classifier import InteractiveClassifier
from ic_core.features import feature_normalization
from ic_core.glyph import Glyph
from ic_core.io_xml import load_glyphs
from ic_core.normalize import NormalizeConfig
from evaluate import classify_page, ingest_glyphs_to_classify
from paths import CSV_VOCAB, TRAINING_XML


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def training_db() -> list[Glyph]:
    """The legacy GameraXML training database, loaded once per module."""
    return load_glyphs(TRAINING_XML)


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
    with open(CSV_VOCAB, newline="") as f:
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


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Unweighted mean per-class F1 over the classes present in ``y_true``.

    Macro (not micro) so rare neume classes count as much as common
    ones — the whole point of tracking F1 alongside accuracy is to
    catch a model that nails ``punctum`` while quietly failing every
    minority class, which raw accuracy would mask. A class with no
    predictions (or no true instances) contributes an F1 of 0.
    """
    labels = sorted(set(y_true))
    f1s: list[float] = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _run_cv(
    glyphs: list[Glyph],
    n_splits: int,
    *,
    cfg: NormalizeConfig | None = None,
    k: int = 1,
) -> tuple[float, float, int]:
    """Stratified k-fold CV. Returns ``(accuracy, macro_f1, n_tested)``.

    When ``cfg`` is given, the entire fit/predict loop runs inside a
    :func:`feature_normalization` block, so both training and query
    features are normalized identically. ``cfg=None`` is the raw-mask
    baseline. Predictions are pooled across folds, then scored once.
    """
    folds = _stratified_folds(glyphs, n_splits)
    assert sum(len(f) for f in folds) > 0, "stratifier dropped all glyphs"

    y_true: list[str] = []
    y_pred: list[str] = []
    ctx = feature_normalization(cfg) if cfg is not None else contextlib.nullcontext()
    with ctx:
        for i in range(n_splits):
            test = [glyphs[j] for j in folds[i]]
            train = [glyphs[j] for kk in range(n_splits) if kk != i for j in folds[kk]]
            clf = InteractiveClassifier(k=k).fit(train)
            for g, pred in zip(test, clf.predict_many(test)):
                y_true.append(g.class_name)
                y_pred.append(pred.class_name)

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)
    return accuracy, _macro_f1(y_true, y_pred), len(y_true)


@pytest.mark.slow
def test_xml_db_5fold_accuracy(training_db):
    accuracy, macro_f1, seen = _run_cv(training_db, n_splits=5, k=1)
    print(
        f"\n[5-fold baseline] tested {seen} glyphs, "
        f"accuracy={accuracy:.4f}  macro-F1={macro_f1:.4f}"
    )
    # Floor calibrated against the first green run (~0.95) with a
    # 10-point margin for jitter from the shuffle seed. A real
    # regression will drop accuracy well below this band.
    assert accuracy >= 0.85, f"k=1 5-fold accuracy below 0.85: {accuracy:.4f}"


# ---------------------------------------------------------------------------
# 3b. A/B: raw-mask baseline vs. feature normalization
# ---------------------------------------------------------------------------


def _normalization_from_env() -> NormalizeConfig:
    """Build the treatment-arm config from IC_NORM_* env vars.

    Defaults exercise all three steps gently: drop <=3px specks,
    centre by mass, 2px border. Override per run, e.g.::

        IC_NORM_DESPECKLE=5 IC_NORM_CENTER=1 IC_NORM_PAD=0 \\
          uv run pytest ../tests/test_real_input_knn.py::test_xml_db_5fold_normalization_ab -s
    """
    import os

    return NormalizeConfig(
        despeckle_min_size=int(os.environ.get("IC_NORM_DESPECKLE", "3")),
        center=os.environ.get("IC_NORM_CENTER", "1") == "1",
        pad=int(os.environ.get("IC_NORM_PAD", "2")),
    )


@pytest.mark.slow
def test_xml_db_5fold_normalization_ab(training_db):
    """Compare raw-mask features against normalized features under CV.

    Caveat baked into the assertion: this CV is entirely *within* the
    GameraXML database — one annotation source. Normalization mainly
    buys *cross-source* consistency (train on VIA crops, classify
    MOTHRA crops), which this harness cannot see. So we print the delta
    for inspection but only assert the normalized arm isn't
    catastrophically broken — a within-source improvement is a bonus,
    not a requirement.
    """
    cfg = _normalization_from_env()
    base_acc, base_f1, seen = _run_cv(training_db, n_splits=5, k=1)
    norm_acc, norm_f1, _ = _run_cv(training_db, n_splits=5, k=1, cfg=cfg)

    print(
        f"\n[5-fold A/B] {seen} glyphs, k=1, config={cfg}\n"
        f"  baseline    accuracy={base_acc:.4f}  macro-F1={base_f1:.4f}\n"
        f"  normalized  accuracy={norm_acc:.4f}  macro-F1={norm_f1:.4f}\n"
        f"  delta       accuracy={norm_acc - base_acc:+.4f}  "
        f"macro-F1={norm_f1 - base_f1:+.4f}"
    )
    assert norm_acc >= 0.70, (
        f"normalized 5-fold accuracy collapsed to {norm_acc:.4f} "
        f"(config={cfg}) — likely a normalization bug, not a tuning miss"
    )


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


