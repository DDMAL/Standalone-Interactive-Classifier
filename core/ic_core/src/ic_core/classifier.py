"""kNN training and classification — hand-rolled, dependency-free.

Phase-1 replacement for the legacy ``prepare_classifier`` and
``run_correction_stage`` from
``../Rodan-lite/backend/django/code/jobs/interactive_classifier/interactive_classifier.py``.

We deliberately do **not** depend on scikit-learn (or any third-party
ML library). The Phase-1 algorithm is just:

1. Compute feature vectors for the training glyphs.
2. Standardise each feature dimension to zero mean / unit variance.
3. For each query glyph, compute Euclidean distances to every
   training row and take the ``k`` smallest.
4. With ``k=1`` (the default) → predicted label is the nearest
   neighbour's label. With ``k>1`` → majority vote, breaking ties by
   the closest neighbour.

That fits comfortably in numpy, runs fast for the dataset sizes we
care about (≪10⁵ training glyphs in the foreseeable future), and
removes a heavyweight dependency from the wheel. If we ever need
acceleration we can drop in a KD-/Ball-tree later behind the same
:class:`InteractiveClassifier` API.

Semantics preserved verbatim from ``docs/KNN_ALGORITHM.md`` /
``docs/migration_plan.md`` §"Algorithm semantics to preserve verbatim":

* **Full re-train every round.** No incremental updates; every call to
  :meth:`InteractiveClassifier.fit` discards prior state and rebuilds.
  Keeps the model consistent with the latest manual corrections at
  the cost of recomputing features each round.
* **k=1 default.** Winner-takes-all, no voting. ``k`` is exposed for
  experimentation but defaults to 1 for parity with the legacy job.
* **Manual glyphs feed training, never classification.** The
  ``id_state_manual`` flag is the boundary: ``True`` → training pool;
  ``False`` → candidate for auto-classification.
* **Ascending-confidence sort order.** Lowest-confidence (most
  uncertain) glyphs surface first for review.
* **Strip transient prefixes (``_group``, ``_delete``) before
  training and before export** via :func:`filter_parts`. The legacy
  ``_split`` prefix is intentionally dropped along with the deferred
  split action.

Deltas from the legacy behaviour, documented for downstream consumers:

* **Confidence is not Gamera-equivalent.** Gamera's ``get_confidence``
  formula was private to its kNN implementation. Here we expose a
  distance-derived proxy (``1 / (1 + distance)`` against the nearest
  neighbour in standardised feature space) that is monotonic in
  distance and therefore preserves the ordering the frontend relies
  on. Absolute values will not match the old XML and should not be
  compared across implementations. See
  ``docs/migration_plan.md`` §"Risks and gotchas" (2).
* **Feature vectors are versioned.** See
  :data:`ic_core.features.FEATURE_VERSION`. Old GameraXML feature
  blobs are not interchangeable.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ic_core.feature_extractor import FeatureExtractorProtocol, HandcraftedExtractor
from ic_core.features import FEATURE_VERSION
from ic_core.glyph import Glyph

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Class name assigned to glyphs that have not yet been classified.
#: Matches the legacy Gamera convention so existing GameraXML
#: round-trips keep producing the same sentinel value on export.
UNCLASSIFIED: str = "UNCLASSIFIED"

#: Default neighbour count. ``k=1`` (winner-takes-all) is the historical
#: behaviour from the Rodan job and must remain the default for parity.
DEFAULT_K: int = 1

#: Prefixes that mark transient user-intent glyphs. They are stripped
#: before every training round and before every export, never persisted.
#: The legacy ``_split`` prefix is intentionally absent — see
#: ``docs/migration_plan.md`` Phase 1 deferred items.
TRANSIENT_PREFIXES: tuple[str, ...] = ("_group", "_delete")

#: Numerical floor for per-feature standard deviation. Features that
#: are constant across the training set get a std of 0; dividing by 0
#: would produce NaNs that then poison every distance calculation. We
#: clamp to this tiny epsilon so constant features contribute zero
#: signal but don't break anything.
_STD_EPSILON: float = 1e-12


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Prediction:
    """A single auto-classification result.

    Attributes:
        class_name: Predicted class label drawn from the training set.
        confidence: Distance-derived proxy in ``(0, 1]``; higher means
            the nearest neighbour was closer in standardised feature
            space. Not directly comparable with the legacy Gamera
            confidence — see module docstring.
    """

    class_name: str
    confidence: float


# ---------------------------------------------------------------------------
# Utilities — filter, collect, sort
# ---------------------------------------------------------------------------


def filter_parts(glyphs: Iterable[Glyph]) -> list[Glyph]:
    """Drop glyphs whose ``class_name`` carries a transient prefix.

    Mirrors the legacy ``filter_parts`` in
    ``interactive_classifier.py``. Run this **before training** and
    **before export** — the prefixes encode UI-only intent (group
    candidates, deletion markers) that must not persist into the
    classifier or the exported XML.

    Args:
        glyphs: Any iterable of :class:`Glyph`.

    Returns:
        A new list containing only the glyphs that survived the filter.
        Order is preserved.
    """
    return [g for g in glyphs if not _is_transient(g.class_name)]


def _is_transient(class_name: str) -> bool:
    """Return ``True`` iff the class name starts with a transient prefix."""
    return any(class_name.startswith(p) for p in TRANSIENT_PREFIXES)


def collect_training_set(
    working_glyphs: Iterable[Glyph],
    training_glyphs: Iterable[Glyph] | None = None,
) -> list[Glyph]:
    """Assemble the training pool used to fit the classifier each round.

    Reproduces the two-step build from the legacy ``prepare_classifier``:

    1. Pull every **manual** glyph (``id_state_manual=True``) out of the
       active *working set* — these are the user's in-session
       corrections.
    2. Append every glyph from the optional external *training
       database* (e.g. a previously-saved GameraXML training file).
       Both manual and previously auto-classified examples in the
       database contribute as labels.

    Transient-prefix glyphs (``_group``, ``_delete``) and any glyph
    still labelled :data:`UNCLASSIFIED` are excluded — they cannot
    contribute a usable label.

    Args:
        working_glyphs: The current session's glyphs (mixed manual /
            auto / unclassified).
        training_glyphs: Optional external training database loaded
            from disk; pass ``None`` (or an empty iterable) when the
            user has not supplied one.

    Returns:
        A fresh list of :class:`Glyph` objects suitable for feeding
        into :meth:`InteractiveClassifier.fit`.
    """
    pool: list[Glyph] = []

    # Step 1 — manual glyphs from the working set are training data.
    for g in working_glyphs:
        if g.id_state_manual and _is_labelled(g):
            pool.append(g)

    # Step 2 — merge in the persisted training database, if any.
    if training_glyphs is not None:
        for g in training_glyphs:
            if _is_labelled(g):
                pool.append(g)

    return pool


def _is_labelled(glyph: Glyph) -> bool:
    """A glyph contributes a label only if it has a real class name."""
    return glyph.class_name != UNCLASSIFIED and not _is_transient(glyph.class_name)


def sort_by_confidence_ascending(glyphs: Iterable[Glyph]) -> list[Glyph]:
    """Return glyphs sorted ascending by ``confidence``.

    The frontend surfaces the *least* confident classifications first
    so reviewers spend their attention where it matters. This helper
    encapsulates the ordering rule in one place so the API layer
    doesn't have to remember it.
    """
    return sorted(glyphs, key=lambda g: g.confidence)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class InteractiveClassifier:
    """A train-from-scratch kNN classifier over :class:`Glyph` objects.

    Hand-rolled in numpy — no scikit-learn dependency. The classifier
    stores three things after :meth:`fit`:

    * ``_mean`` / ``_std`` — the per-feature standardisation
      parameters computed on the training set. Predictions transform
      query feature vectors with the same parameters.
    * ``_X`` — the standardised training feature matrix.
    * ``_y`` — the parallel array of training labels.

    Standardisation is required because the Phase-1 feature vector
    (see :mod:`ic_core.features`) mixes raw pixel counts
    (``nrows_feature``, ``ncols_feature`` — order 10²) with
    unit-interval ratios (``volume``, ``volume16regions_*`` — order
    10⁰). Without scaling the pixel-count dimensions would dominate
    the Euclidean distance and the classifier would effectively
    ignore everything else.

    Typical lifecycle per user round-trip:

    >>> clf = InteractiveClassifier(k=1)
    >>> clf.fit(collect_training_set(session.glyphs, session.training_glyphs))
    >>> updated, _ = run_correction_stage(session.glyphs, session.training_glyphs)
    >>> session.glyphs = sort_by_confidence_ascending(updated)

    The classifier is intentionally cheap to construct — a fresh
    instance per round is the design (full re-train every round).
    """

    def __init__(
        self,
        k: int = DEFAULT_K,
        extractor: FeatureExtractorProtocol | None = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k!r}")

        #: User-configurable neighbour count. ``k=1`` is the default
        #: for parity with the legacy Gamera job.
        self.k: int = k

        #: Feature extractor. Defaults to the 29-dim handcrafted extractor
        #: for backwards compatibility.
        self._extractor: FeatureExtractorProtocol = extractor or HandcraftedExtractor()

        #: The feature-vector version the classifier was trained on.
        #: Persisted alongside any exported state so callers can
        #: detect cross-version mismatches.
        self.feature_version: str = FEATURE_VERSION

        # Internals — populated on .fit(); shape annotations below.
        self._mean: np.ndarray | None = None  # shape (D,)
        self._std: np.ndarray | None = None   # shape (D,) — never zero, see _STD_EPSILON
        self._X: np.ndarray | None = None     # shape (N, D), standardised
        self._y: np.ndarray | None = None     # shape (N,), dtype object (class names)
        self._classes: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """``True`` once :meth:`fit` has run on a non-empty training set."""
        return self._X is not None

    @property
    def training_size(self) -> int:
        """Number of training examples used in the most recent ``fit``."""
        return 0 if self._X is None else int(self._X.shape[0])

    @property
    def classes(self) -> tuple[str, ...]:
        """Class labels the classifier has seen, in sorted order."""
        return self._classes

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, training_glyphs: Sequence[Glyph]) -> "InteractiveClassifier":
        """Train (or re-train, from scratch) on ``training_glyphs``.

        Calling ``fit`` discards any prior model state. This is the
        intended pattern: the migration plan and ``KNN_ALGORITHM.md``
        both require a **full re-train every round** to keep the
        model in sync with the latest user corrections.

        Args:
            training_glyphs: A sequence (not just iterable — order
                matters so that two consecutive ``fit`` calls on the
                same input are bit-identical) of :class:`Glyph`
                objects with real labels. Use
                :func:`collect_training_set` to assemble this list
                from the session.

        Returns:
            ``self`` so calls can be chained (``clf.fit(...).predict(...)``).

        Raises:
            ValueError: If the training set is empty or smaller than ``k``.
        """
        if not training_glyphs:
            # No training data → the classifier is unusable. We raise
            # rather than silently producing UNCLASSIFIED everywhere
            # so the caller can decide on a UX (e.g. ask the user to
            # label at least one glyph manually).
            raise ValueError(
                "Cannot fit InteractiveClassifier with zero training glyphs"
            )

        if len(training_glyphs) < self.k:
            raise ValueError(
                f"Need at least k={self.k} training glyphs, "
                f"got {len(training_glyphs)}"
            )

        X = self._extractor.extract_batch(training_glyphs)  # (N, D), float64
        y = np.asarray([g.class_name for g in training_glyphs], dtype=object)

        # Fit per-feature standardisation parameters on the training
        # set. Features with zero variance get clamped to _STD_EPSILON
        # so the divide is safe; those dimensions then contribute zero
        # signal (any query value minus the mean is also zero in
        # standardised space, give or take floating-point noise).
        self._mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < _STD_EPSILON] = _STD_EPSILON
        self._std = std

        # Pre-standardise the stored training matrix once. Predictions
        # re-use this — they only need to standardise the query rows.
        self._X = (X - self._mean) / self._std
        self._y = y
        self._classes = tuple(sorted({str(label) for label in y.tolist()}))

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, glyph: Glyph) -> Prediction:
        """Classify a single glyph.

        Returns:
            A :class:`Prediction` carrying the winning class name and
            a confidence in ``(0, 1]``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        # Single-glyph convenience wrapper around the batch path.
        # Keeping the batch path canonical avoids subtle divergence
        # between the two.
        return self.predict_many([glyph])[0]

    def predict_many(self, glyphs: Sequence[Glyph]) -> list[Prediction]:
        """Classify a batch of glyphs in one feature-extraction pass.

        Args:
            glyphs: A sequence of :class:`Glyph` objects. May be empty,
                in which case an empty list is returned.

        Returns:
            One :class:`Prediction` per input glyph, in the same order.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._require_trained()
        assert self._mean is not None  # narrow Optional → np.ndarray for the type checker
        assert self._std is not None
        assert self._X is not None
        assert self._y is not None

        if not glyphs:
            return []

        # Standardise query features with the *training* statistics.
        Q = self._extractor.extract_batch(glyphs)       # (M, D)
        Q_scaled = (Q - self._mean) / self._std         # (M, D)

        # Pairwise squared Euclidean distances between each query and
        # every training row. Using the identity
        #     ||q - x||² = ||q||² + ||x||² - 2 q·x
        # is the standard numerically-cheap formulation; clamping the
        # result at 0 guards against tiny negative values from
        # floating-point rounding on near-duplicate points.
        train_sq = np.sum(self._X * self._X, axis=1)      # (N,)
        query_sq = np.sum(Q_scaled * Q_scaled, axis=1)    # (M,)
        cross = Q_scaled @ self._X.T                      # (M, N)
        d2 = query_sq[:, None] + train_sq[None, :] - 2.0 * cross
        np.maximum(d2, 0.0, out=d2)
        distances = np.sqrt(d2)                           # (M, N)

        # For each query row, find the k smallest distances.
        # np.argpartition is O(N) per query and sufficient because we
        # only need the *set* of k nearest neighbours; we then sort
        # that small slice exactly so tie-breaking is deterministic.
        k = min(self.k, distances.shape[1])
        nearest_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        # Sort the k nearest by distance for stable tie-breaking.
        for i in range(nearest_idx.shape[0]):
            row = nearest_idx[i]
            nearest_idx[i] = row[np.argsort(distances[i, row])]

        return [
            self._vote(distances[i], nearest_idx[i])
            for i in range(distances.shape[0])
        ]

    def _vote(self, row_distances: np.ndarray, row_indices: np.ndarray) -> Prediction:
        """Combine the k nearest neighbours of a single query into a Prediction.

        Voting rule:
        * k=1 → the single nearest neighbour wins outright (fast path).
        * k>1 → majority vote; if multiple labels tie, prefer the
          label whose closest representative is itself closest.

        Confidence is always derived from the *winning* neighbour's
        distance, not from a vote-share statistic — that keeps the
        scale comparable between the k=1 and k>1 cases.
        """
        assert self._y is not None  # narrowed by caller

        labels = [str(self._y[j]) for j in row_indices]
        dists = [float(row_distances[j]) for j in row_indices]

        if len(labels) == 1:
            # k=1 fast path — also the default for parity.
            return Prediction(
                class_name=labels[0],
                confidence=_distance_to_confidence(dists[0]),
            )

        # k>1: majority vote with closest-rep tie-break. row_indices
        # is already sorted ascending by distance, so the first time
        # we see each label is also its closest distance.
        counts = Counter(labels)
        top_count = max(counts.values())
        tied_labels = [lbl for lbl, c in counts.items() if c == top_count]
        if len(tied_labels) == 1:
            winner = tied_labels[0]
        else:
            # Tie-break: among tied labels, pick whichever appears
            # first in `labels` (i.e. has the closest representative).
            winner = next(lbl for lbl in labels if lbl in tied_labels)

        # Confidence based on the winning label's closest neighbour.
        winner_distance = next(d for lbl, d in zip(labels, dists) if lbl == winner)
        return Prediction(
            class_name=winner,
            confidence=_distance_to_confidence(winner_distance),
        )

    def _require_trained(self) -> None:
        if self._X is None:
            raise RuntimeError(
                "InteractiveClassifier is not trained; call .fit() first"
            )


# ---------------------------------------------------------------------------
# Pipeline-level helpers
# ---------------------------------------------------------------------------


def _distance_to_confidence(distance: float) -> float:
    """Map a non-negative distance to a confidence in ``(0, 1]``.

    Uses the monotonically decreasing transform
    ``1 / (1 + distance)``: distance ``0`` (exact match) → confidence
    ``1.0``; large distances → confidence approaching ``0``.

    This is a *proxy*, not the Gamera ``get_confidence`` value — see
    the module docstring for the compatibility caveat.
    """
    # Numerical safety: clamp negatives (shouldn't happen but the
    # extra epsilon costs nothing) and avoid weirdness if a caller
    # ever passes a non-finite value.
    if not np.isfinite(distance) or distance < 0.0:
        return 0.0
    return 1.0 / (1.0 + distance)


def run_correction_stage(
    glyphs: Sequence[Glyph],
    training_glyphs: Sequence[Glyph] | None = None,
    *,
    k: int = DEFAULT_K,
) -> tuple[list[Glyph], InteractiveClassifier]:
    """End-to-end equivalent of the legacy ``run_correction_stage``.

    Trains a fresh classifier on ``manual(glyphs) ∪ training_glyphs``
    and re-classifies every non-manual glyph in ``glyphs``.
    :class:`Glyph` is frozen, so this returns a new list rather than
    mutating in place.

    Manual glyphs are passed through untouched. Transient-prefix
    glyphs are excluded from both training and the returned output.

    Args:
        glyphs: The current working set.
        training_glyphs: Optional external training database.
        k: Neighbour count (defaults to 1 for parity).

    Returns:
        A 2-tuple of ``(new_glyphs, classifier)`` where
        ``new_glyphs`` is the updated working set (in input order
        post-filter — sort with :func:`sort_by_confidence_ascending`
        for display) and ``classifier`` is the trained model in case
        the caller wants to reuse it (e.g. for an immediate export
        step).

    Raises:
        ValueError: If the assembled training pool is empty —
            propagated from :meth:`InteractiveClassifier.fit`.
    """
    # 1. Strip transient-prefix glyphs from the working set before
    #    anything else so `_group` / `_delete` markers do not enter
    #    the training pool and are not included in the returned
    #    output.
    cleaned = filter_parts(glyphs)

    # 2. Assemble the training pool and fit.
    pool = collect_training_set(cleaned, training_glyphs)
    classifier = InteractiveClassifier(k=k).fit(pool)

    # 3. Split the cleaned working set into:
    #    - manual glyphs (passed through unchanged), and
    #    - unmanual glyphs (to be auto-classified now).
    unmanual_indices: list[int] = []
    unmanual_glyphs: list[Glyph] = []
    for i, g in enumerate(cleaned):
        if not g.id_state_manual:
            unmanual_indices.append(i)
            unmanual_glyphs.append(g)

    # 4. Classify the unmanual glyphs in one batched pass.
    predictions = classifier.predict_many(unmanual_glyphs)

    # 5. Stitch the predictions back into the working-set order.
    #    Using ``classify_automatic`` preserves the glyph's UUID and
    #    geometry, only overwriting class_name + confidence +
    #    id_state_manual=False (which it was already, but explicit
    #    is better than implicit).
    out: list[Glyph] = list(cleaned)
    for idx, pred in zip(unmanual_indices, predictions):
        out[idx] = cleaned[idx].classify_automatic(
            class_name=pred.class_name,
            confidence=pred.confidence,
        )

    return out, classifier
