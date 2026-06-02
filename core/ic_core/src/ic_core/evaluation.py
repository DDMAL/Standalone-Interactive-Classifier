"""Classifier-agnostic evaluation utilities.

Supports two evaluation modes:

1. **Cross-validation** (:func:`cross_validate`) — stratified k-fold on a
   single labelled dataset. Training and test both come from the same source;
   useful when you have no held-out set.

2. **Train/test evaluation** (:func:`evaluate`) — you supply a labelled
   training set and a separate labelled test set (ground truth). The
   classifier trains on the training set and is scored against the test
   labels. Use this when you have a genuine held-out ground truth.

Both modes accept any callable that conforms to the :class:`ClassifierProtocol`
interface, so kNN, ViT, random-forest, or any future method can be dropped in
without touching this file.

Example — kNN cross-validation::

    from pathlib import Path
    from ic_core.classifier import InteractiveClassifier
    from ic_core.evaluation import cross_validate, print_report, knn_factory
    from ic_core.io_xml import load_glyphs

    glyphs = load_glyphs(Path("training_data.xml"))
    result = cross_validate(glyphs, classifier_factory=knn_factory(k=1))
    print_report(result)

Example — held-out test set::

    from ic_core.evaluation import evaluate, print_report, knn_factory

    result = evaluate(
        train_glyphs=train_glyphs,
        test_glyphs=test_glyphs,   # ground-truth labels in .class_name
        classifier_factory=knn_factory(k=1),
    )
    print_report(result)
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from ic_core.classifier import UNCLASSIFIED, _is_transient
from ic_core.glyph import Glyph


# ---------------------------------------------------------------------------
# Classifier protocol — the only contract any classifier must satisfy
# ---------------------------------------------------------------------------


class ClassifierProtocol(Protocol):
    """Minimal interface an evaluatable classifier must implement."""

    def fit(self, training_glyphs: Sequence[Glyph]) -> "ClassifierProtocol":
        """Train on ``training_glyphs``; returns self."""
        ...

    def predict_many(self, glyphs: Sequence[Glyph]) -> list:
        """Return one prediction object per input glyph.

        Each prediction object must expose a ``.class_name`` attribute.
        """
        ...


#: Factory type: a zero-argument callable that returns a fresh classifier.
ClassifierFactory = Callable[[], ClassifierProtocol]


def knn_factory(k: int = 1) -> ClassifierFactory:
    """Return a factory that builds a fresh :class:`InteractiveClassifier` each call."""
    from ic_core.classifier import InteractiveClassifier

    def _factory() -> InteractiveClassifier:
        return InteractiveClassifier(k=k)

    _factory.__name__ = f"kNN(k={k})"
    return _factory


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ClassMetrics:
    """Per-class precision, recall, and F1."""

    label: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvaluationResult:
    """Results from either :func:`evaluate` or :func:`cross_validate`."""

    mode: str                  # "train_test" or "cross_validation"
    classifier_name: str
    n_train: int
    n_test: int
    classes: list[str]

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float

    per_class: list[ClassMetrics]

    # confusion_matrix[i][j] = # times true class i was predicted as class j
    confusion_matrix: list[list[int]]

    # cross-validation only
    fold_accuracies: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _labelled(glyphs: Sequence[Glyph]) -> list[Glyph]:
    """Drop UNCLASSIFIED and transient-prefix glyphs."""
    return [
        g for g in glyphs
        if g.class_name != UNCLASSIFIED and not _is_transient(g.class_name)
    ]


def _compute_metrics(
    true_labels: list[str],
    pred_labels: list[str],
    classes: list[str],
) -> tuple[float, list[ClassMetrics], list[list[int]]]:
    n = len(true_labels)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    nc = len(classes)

    cm = [[0] * nc for _ in range(nc)]
    correct = 0
    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            correct += 1
        true_i = label_to_idx.get(true, -1)
        pred_i = label_to_idx.get(pred, -1)
        if true_i >= 0 and pred_i >= 0:
            cm[true_i][pred_i] += 1

    accuracy = correct / n if n > 0 else 0.0

    per_class: list[ClassMetrics] = []
    for i, label in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(nc)) - tp
        fn = sum(cm[i][j] for j in range(nc)) - tp
        support = sum(cm[i])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        per_class.append(ClassMetrics(label=label, support=support,
                                      precision=precision, recall=recall, f1=f1))

    active = [m for m in per_class if m.support > 0]
    macro_p = sum(m.precision for m in active) / len(active) if active else 0.0
    macro_r = sum(m.recall for m in active) / len(active) if active else 0.0
    macro_f = sum(m.f1 for m in active) / len(active) if active else 0.0

    return accuracy, per_class, cm


def _classifier_name(factory: ClassifierFactory) -> str:
    fn = getattr(factory, "__name__", None)
    if fn:
        return fn
    return type(factory()).__name__


# ---------------------------------------------------------------------------
# Mode 1 — train/test with an explicit ground-truth test set
# ---------------------------------------------------------------------------


def evaluate(
    train_glyphs: Sequence[Glyph],
    test_glyphs: Sequence[Glyph],
    *,
    classifier_factory: ClassifierFactory | None = None,
) -> EvaluationResult:
    """Evaluate a classifier against an explicit held-out ground-truth test set.

    The labels in ``test_glyphs.class_name`` are treated as ground truth.
    The classifier is trained once on ``train_glyphs`` and scored on
    ``test_glyphs``.

    Args:
        train_glyphs: Labelled glyphs used for training. UNCLASSIFIED and
            transient-prefix entries are filtered out automatically.
        test_glyphs: Ground-truth labelled glyphs. UNCLASSIFIED entries
            are excluded from scoring.
        classifier_factory: Zero-argument callable returning a fresh
            classifier. Defaults to :func:`knn_factory` (k=1).

    Returns:
        :class:`EvaluationResult` with ``mode="train_test"``.
    """
    if classifier_factory is None:
        classifier_factory = knn_factory(k=1)

    train = _labelled(train_glyphs)
    test = _labelled(test_glyphs)

    if not train:
        raise ValueError("train_glyphs contains no usable labelled glyphs.")
    if not test:
        raise ValueError("test_glyphs contains no usable labelled glyphs.")

    clf = classifier_factory().fit(train)
    predictions = clf.predict_many(test)

    true_labels = [g.class_name for g in test]
    pred_labels = [p.class_name for p in predictions]

    # Union of classes seen in train and test
    classes = sorted({g.class_name for g in train} | {g.class_name for g in test})

    accuracy, per_class, cm = _compute_metrics(true_labels, pred_labels, classes)
    active = [m for m in per_class if m.support > 0]

    return EvaluationResult(
        mode="train_test",
        classifier_name=_classifier_name(classifier_factory),
        n_train=len(train),
        n_test=len(test),
        classes=classes,
        accuracy=accuracy,
        macro_precision=sum(m.precision for m in active) / max(1, len(active)),
        macro_recall=sum(m.recall for m in active) / max(1, len(active)),
        macro_f1=sum(m.f1 for m in active) / max(1, len(active)),
        per_class=per_class,
        confusion_matrix=cm,
    )


# ---------------------------------------------------------------------------
# Mode 2 — stratified k-fold cross-validation on a single dataset
# ---------------------------------------------------------------------------


def _stratified_kfold(
    glyphs: list[Glyph], k: int, seed: int = 42
) -> list[tuple[list[Glyph], list[Glyph]]]:
    rng = random.Random(seed)
    by_class: dict[str, list[Glyph]] = defaultdict(list)
    for g in glyphs:
        by_class[g.class_name].append(g)
    for lst in by_class.values():
        rng.shuffle(lst)

    class_folds: dict[str, list[list[Glyph]]] = {}
    for label, items in by_class.items():
        buckets: list[list[Glyph]] = [[] for _ in range(k)]
        for i, g in enumerate(items):
            buckets[i % k].append(g)
        class_folds[label] = buckets

    folds: list[tuple[list[Glyph], list[Glyph]]] = []
    for fold_idx in range(k):
        test: list[Glyph] = []
        train: list[Glyph] = []
        for buckets in class_folds.values():
            for i, bucket in enumerate(buckets):
                (test if i == fold_idx else train).extend(bucket)
        folds.append((train, test))
    return folds


def cross_validate(
    glyphs: Sequence[Glyph],
    *,
    k_folds: int = 5,
    classifier_factory: ClassifierFactory | None = None,
    seed: int = 42,
) -> EvaluationResult:
    """Stratified k-fold cross-validation on a single labelled dataset.

    The same dataset provides both training and test glyphs across folds.
    Predictions from all folds are pooled for overall metric computation.

    Args:
        glyphs: All labelled glyphs. UNCLASSIFIED and transient entries
            are filtered out automatically.
        k_folds: Number of folds. Defaults to 5.
        classifier_factory: Zero-argument callable returning a fresh
            classifier. Defaults to :func:`knn_factory` (k=1).
        seed: RNG seed for reproducible splits.

    Returns:
        :class:`EvaluationResult` with ``mode="cross_validation"``.
    """
    if classifier_factory is None:
        classifier_factory = knn_factory(k=1)

    usable = _labelled(glyphs)
    if not usable:
        raise ValueError("No labelled glyphs found after filtering.")

    classes = sorted({g.class_name for g in usable})
    folds = _stratified_kfold(usable, k_folds, seed=seed)

    all_true: list[str] = []
    all_pred: list[str] = []
    fold_accuracies: list[float] = []

    for fold_idx, (train_glyphs, test_glyphs) in enumerate(folds):
        if not train_glyphs:
            raise ValueError(f"Fold {fold_idx} produced an empty training set.")
        if not test_glyphs:
            continue

        clf = classifier_factory().fit(train_glyphs)
        predictions = clf.predict_many(test_glyphs)

        true_labels = [g.class_name for g in test_glyphs]
        pred_labels = [p.class_name for p in predictions]

        correct = sum(t == p for t, p in zip(true_labels, pred_labels))
        fold_accuracies.append(correct / len(test_glyphs))
        all_true.extend(true_labels)
        all_pred.extend(pred_labels)

    accuracy, per_class, cm = _compute_metrics(all_true, all_pred, classes)
    active = [m for m in per_class if m.support > 0]

    return EvaluationResult(
        mode="cross_validation",
        classifier_name=_classifier_name(classifier_factory),
        n_train=len(usable),
        n_test=len(usable),
        classes=classes,
        accuracy=accuracy,
        macro_precision=sum(m.precision for m in active) / max(1, len(active)),
        macro_recall=sum(m.recall for m in active) / max(1, len(active)),
        macro_f1=sum(m.f1 for m in active) / max(1, len(active)),
        per_class=per_class,
        confusion_matrix=cm,
        fold_accuracies=fold_accuracies,
    )


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_report(result: EvaluationResult) -> None:
    """Print a human-readable evaluation report to stdout."""
    print("=" * 62)
    print("  Classifier Evaluation Report")
    print("=" * 62)
    print(f"  Classifier : {result.classifier_name}")
    print(f"  Mode       : {result.mode}")
    if result.mode == "cross_validation":
        print(f"  Folds      : {len(result.fold_accuracies)}")
        print(f"  Dataset    : {result.n_train} labelled glyphs")
    else:
        print(f"  Train set  : {result.n_train} glyphs")
        print(f"  Test set   : {result.n_test} glyphs  ← ground truth")
    print(f"  Classes    : {len(result.classes)}")
    print()

    print(f"  Accuracy         : {result.accuracy:.4f}")
    if result.fold_accuracies:
        fold_str = "  ".join(f"{a:.4f}" for a in result.fold_accuracies)
        std = float(np.std(result.fold_accuracies))
        print(f"  Per-fold accuracy: {fold_str}")
        print(f"  Fold std dev     : {std:.4f}")
    print(f"  Macro precision  : {result.macro_precision:.4f}")
    print(f"  Macro recall     : {result.macro_recall:.4f}")
    print(f"  Macro F1         : {result.macro_f1:.4f}")
    print()

    print(f"  {'Class':<28} {'Support':>7} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("  " + "-" * 58)
    for m in sorted(result.per_class, key=lambda m: -m.support):
        if m.support == 0:
            continue
        print(
            f"  {m.label:<28} {m.support:>7} "
            f"{m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f}"
        )
    print()

    active_classes = [c for c, m in zip(result.classes, result.per_class) if m.support > 0]
    active_idx = [result.classes.index(c) for c in active_classes]
    if len(active_classes) <= 15:
        print("  Confusion matrix (rows=true, cols=predicted):")
        header = "  " + " " * 28 + "  " + "  ".join(f"{c[:6]:>6}" for c in active_classes)
        print(header)
        for i in active_idx:
            row_vals = "  ".join(
                f"{result.confusion_matrix[i][j]:>6}" for j in active_idx
            )
            print(f"  {result.classes[i]:<28}  {row_vals}")
        print()

    print("=" * 62)
