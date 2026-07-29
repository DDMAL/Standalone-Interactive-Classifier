"""Optional SSL+HC fused feature classifier -- an alternative backend
to the default :class:`ic_core.classifier.InteractiveClassifier` kNN.

This module is entirely separate from the rest of ``ic_core`` and is
**not imported by anything on the default classification path**. It
requires ``scikit-learn`` (and, transitively via
:mod:`ic_core.ssl_extractor`, ``torch``/``transformers``), none of
which are core dependencies of this package -- install them via the
``ssl`` extra (``pip install ic-core[ssl]``) before using this module.

Model selection summary: DINO SimCLR checkpoint at epoch 5, final-layer
CLS+mean pooling, fused with 29-dim handcrafted features via weighted
concatenation (HC block scaled by ``hc_weight=4.0`` before
concatenation, correcting for the 768:29 dimensionality imbalance
between the two blocks), classified with a linear-kernel SVM
(``C=1.0``).

Real-pixel crops (``Glyph.image_gray_b64``, despite the name) are
colour, not greyscale -- the checkpoint was trained on real colour
manuscript photographs (see ``prepare_ssl_crops.py``), and feeding it
a greyscale-then-channel-replicated image is a different input
distribution than what it saw during pretraining. See
``ic_core.ingest._load_page_color``.

Re-selected via leave-one-manuscript-out cross-validation across
Hufnagel, MS234, and Antiphonal+NZ-Wt MSR-03 (each held out in turn,
trained on the other two), plus a single-manuscript stress test
(training on only 2 pages of one manuscript, testing on its third
page) -- both with real colour crops. An RBF kernel was deployed here
previously; it collapses onto majority classes whenever trained on a
single small/imbalanced manuscript (the common real case: a user
selecting one training preset), and this linear config beat it on
every re-test, including being the first configuration tested to
beat the plain handcrafted-feature kNN baseline on the
single-manuscript stress test (79.4% vs. 77.0%).

Exposes the same public shape as
:class:`ic_core.classifier.InteractiveClassifier` --
``.fit()``/``.predict()``/``.predict_many()`` plus
``is_trained``/``training_size``/``classes`` -- so it is a drop-in
alternative wherever an ``InteractiveClassifier``-like object is
expected (see ``run_correction_stage``'s ``classifier_factory``
parameter).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from ic_core.classifier import Prediction
from ic_core.features import compute_features_batch
from ic_core.glyph import Glyph

#: Selected via cross-validation -- see module docstring. Not re-tuned
#: per deployment; override via the constructor if you have reason to.
DEFAULT_HC_WEIGHT: float = 4.0
DEFAULT_C: float = 1.0


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SSLFusionClassifier requires scikit-learn, which is an "
            "optional dependency of ic-core. Install it with "
            "`pip install ic-core[ssl]` before using ic_core.ssl_classifier."
        ) from exc


class SSLFusionClassifier:
    """SSL+HC fused features, classified with a linear-kernel SVM.

    Args:
        checkpoint: Path to the DINO SimCLR fine-tuned checkpoint
            directory (or ``None`` to fall back to the plain
            pretrained backbone -- not recommended, see the extractor
            comparison in the accompanying report).
        hc_weight: Scalar applied to the standardized handcrafted
            feature block before concatenating with the standardized
            SSL block. Defaults to the cross-validated value; only
            override this if you're re-running model selection.
        C: Inverse regularization strength for the SVM classifier.
            Defaults to the cross-validated value.

    Raises:
        ImportError: If scikit-learn (or, at extraction time,
            torch/transformers) is not installed.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        hc_weight: float = DEFAULT_HC_WEIGHT,
        C: float = DEFAULT_C,
    ) -> None:
        _require_sklearn()
        self.checkpoint = checkpoint
        self.hc_weight = hc_weight
        self.C = C

        self._ssl_extractor = None  # lazily constructed on first fit/predict
        self._sc_ssl = None
        self._sc_hc = None
        self._clf = None
        self._classes: tuple[str, ...] = ()
        self._training_size = 0
        self._calibrated = False

    # ------------------------------------------------------------------
    # Introspection -- mirrors InteractiveClassifier's public surface
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._clf is not None

    @property
    def training_size(self) -> int:
        return self._training_size

    @property
    def classes(self) -> tuple[str, ...]:
        return self._classes

    @property
    def is_calibrated(self) -> bool:
        """Whether ``predict_many`` returns real probabilities.

        ``False`` when the last :meth:`fit` had a training class with
        fewer than 2 examples -- calibration was skipped and every
        prediction gets a fixed confidence of 1.0 instead.
        """
        return self._calibrated

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _get_extractor(self):
        if self._ssl_extractor is None:
            from ic_core.ssl_extractor import ViTExtractor
            self._ssl_extractor = ViTExtractor(checkpoint=self.checkpoint)
        return self._ssl_extractor

    def _ssl_features(self, glyphs: Sequence[Glyph]) -> np.ndarray:
        """Get the SSL feature block, per glyph preferring a precomputed
        ``ssl_embedding`` (see ``ic_core.ssl_preset_embeddings``) over a
        live extractor pass on ``image_gray_b64``.

        Glyphs within one call must be uniformly one or the other --
        :meth:`fit`/:meth:`predict_many` partition their input by
        ``ssl_embedding`` availability before calling this, so a mixed
        batch never reaches here.
        """
        if glyphs[0].ssl_embedding is not None:
            return np.asarray([g.ssl_embedding for g in glyphs], dtype=np.float64)
        return self._get_extractor().extract_batch(glyphs, pooling="cls_mean")

    def _fused_features(self, glyphs: Sequence[Glyph], fit_scalers: bool) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler

        X_hc = compute_features_batch(glyphs)
        precomputed = [g for g in glyphs if g.ssl_embedding is not None]
        live = [g for g in glyphs if g.ssl_embedding is None]
        if precomputed and live:
            precomputed_dim = len(precomputed[0].ssl_embedding)
            live_features = self._ssl_features(live)
            if live_features.shape[1] != precomputed_dim:
                raise ValueError(
                    f"Live-extracted SSL features are {live_features.shape[1]}-dim "
                    f"but precomputed ssl_embedding vectors are {precomputed_dim}-dim "
                    "-- the configured checkpoint doesn't match the one the "
                    "precomputed embeddings (e.g. a preset's .ssl_embeddings.npz) "
                    "were generated with."
                )
            X_ssl = np.empty((len(glyphs), precomputed_dim))
            idx_precomputed = [i for i, g in enumerate(glyphs) if g.ssl_embedding is not None]
            idx_live = [i for i, g in enumerate(glyphs) if g.ssl_embedding is None]
            X_ssl[idx_precomputed] = self._ssl_features(precomputed)
            X_ssl[idx_live] = live_features
        else:
            X_ssl = self._ssl_features(glyphs)

        if fit_scalers:
            self._sc_ssl = StandardScaler().fit(X_ssl)
            self._sc_hc = StandardScaler().fit(X_hc)

        X_ssl_s = self._sc_ssl.transform(X_ssl)
        X_hc_s = self._sc_hc.transform(X_hc) * self.hc_weight
        return np.concatenate([X_ssl_s, X_hc_s], axis=1)

    # ------------------------------------------------------------------
    # Training / prediction
    # ------------------------------------------------------------------

    def fit(self, training_glyphs: Sequence[Glyph]) -> "SSLFusionClassifier":
        """Train (or re-train, from scratch) on ``training_glyphs``.

        Mirrors :meth:`InteractiveClassifier.fit`'s "full re-train
        every round" semantics: calling this discards any prior model
        state.

        Glyphs with neither a precomputed ``ssl_embedding`` (see
        ``ic_core.ssl_preset_embeddings`` -- only set for presets that
        ship a companion embeddings file) nor a real-pixel crop
        (``image_gray_b64`` -- set only by a live page upload with
        ``store_real_crop=True``) are silently excluded from this
        backend's training pool, since the SSL extractor cannot use
        them. This is a difference from :class:`InteractiveClassifier`,
        which uses every training glyph regardless of crop
        availability -- worth knowing if a classify round trains on
        fewer examples than the training-set count displayed in the UI
        suggests.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC

        if not training_glyphs:
            raise ValueError(
                "Cannot fit SSLFusionClassifier with zero training glyphs"
            )

        usable = [
            g
            for g in training_glyphs
            if g.ssl_embedding is not None or g.image_gray_b64 is not None
        ]
        if not usable:
            raise ValueError(
                f"None of the {len(training_glyphs)} training glyphs have "
                "usable SSL features -- neither a precomputed ssl_embedding "
                "(only available for presets shipping a companion "
                ".ssl_embeddings.npz file) nor a real-pixel crop "
                "(image_gray_b64, set by a live page upload with "
                "store_real_crop=True). Training data sourced from a preset "
                "without embeddings or an uploaded GameraXML file only "
                "carries the binary mask and cannot be used by the "
                "ssl_fusion backend. Label at least one glyph manually in "
                "this session, or switch back to the 'knn' backend."
            )
        training_glyphs = usable

        X = self._fused_features(training_glyphs, fit_scalers=True)
        y = np.asarray([g.class_name for g in training_glyphs], dtype=object)

        # CalibratedClassifierCV(..., ensemble=False) rather than
        # SVC(probability=True) directly: same Platt-scaling probability
        # estimates, but not on sklearn's deprecation path (probability=True
        # is slated for removal in a future release).
        #
        # Calibration needs stratified CV, which needs every class to have
        # at least `cv` examples -- unlike the LogisticRegression this
        # replaced, which had no such floor and fit fine on a single
        # example per class. Real sessions often start there (a user has
        # only just begun labelling), so this must degrade gracefully
        # rather than raise: shrink `cv` to the rarest class's count, and
        # below 2 (where no CV split is even possible) skip calibration
        # entirely -- fit an uncalibrated SVC and report a fixed
        # confidence, matching the "no predict_proba" fallback pattern
        # used elsewhere in this codebase's classifier wrappers.
        counts = Counter(y.tolist())
        min_class_count = min(counts.values())
        base = SVC(C=self.C, kernel="linear", class_weight="balanced")
        if min_class_count < 2:
            clf = base
            self._calibrated = False
        else:
            clf = CalibratedClassifierCV(base, cv=min(5, min_class_count), ensemble=False)
            self._calibrated = True
        clf.fit(X, y)

        self._clf = clf
        self._classes = tuple(sorted({str(label) for label in y.tolist()}))
        self._training_size = len(training_glyphs)
        return self

    def predict(self, glyph: Glyph) -> Prediction:
        """Classify a single glyph. See :meth:`predict_many`."""
        return self.predict_many([glyph])[0]

    def predict_many(self, glyphs: Sequence[Glyph]) -> list[Prediction]:
        """Classify a batch of glyphs in one feature-extraction pass.

        Raises:
            ValueError: If any glyph has neither a precomputed
                ``ssl_embedding`` nor a real-pixel crop
                (``image_gray_b64``) -- unlike :meth:`fit`, which can
                silently drop unusable training glyphs, every glyph
                passed here must yield a prediction, so this raises
                immediately with the offending glyph ids rather than
                failing deep inside feature extraction.
        """
        self._require_trained()
        if not glyphs:
            return []

        unusable = [g.id for g in glyphs if g.ssl_embedding is None and g.image_gray_b64 is None]
        if unusable:
            raise ValueError(
                f"{len(unusable)} of {len(glyphs)} glyph(s) have neither a "
                f"precomputed ssl_embedding nor a real-pixel crop "
                f"(image_gray_b64), so ssl_fusion cannot classify them: "
                f"{unusable[:10]}{'...' if len(unusable) > 10 else ''}"
            )

        X = self._fused_features(glyphs, fit_scalers=False)
        classes = self._clf.classes_
        if self._calibrated:
            proba = self._clf.predict_proba(X)
        else:
            # Fit skipped calibration (a class had <2 training examples) --
            # no predict_proba available. One-hot the hard prediction with
            # confidence=1.0, same fallback other classifier wrappers in
            # this codebase use when proba isn't available.
            preds = self._clf.predict(X)
            proba = np.zeros((len(glyphs), len(classes)))
            class_index = {c: i for i, c in enumerate(classes)}
            for i, p in enumerate(preds):
                proba[i, class_index[p]] = 1.0
        pred_idx = proba.argmax(axis=1)

        return [
            Prediction(
                class_name=str(classes[pred_idx[i]]),
                confidence=float(proba[i, pred_idx[i]]),
            )
            for i in range(len(glyphs))
        ]

    def _require_trained(self) -> None:
        if self._clf is None:
            raise RuntimeError(
                "SSLFusionClassifier is not trained; call .fit() first"
            )


def default_ssl_classifier_factory() -> SSLFusionClassifier:
    """Build an :class:`SSLFusionClassifier` using the cross-validated
    hyperparameters and a checkpoint path read from the
    ``IC_SSL_CHECKPOINT`` environment variable.

    This is the one place a deployment's fine-tuned checkpoint path is
    configured, so callers (e.g. ``ic_core.state.Session.classify``)
    don't need to know anything about the filesystem layout -- they
    just pick the ``ssl_fusion`` backend and this resolves it.

    Raises:
        RuntimeError: If ``IC_SSL_CHECKPOINT`` is not set.
    """
    import os

    checkpoint = os.environ.get("IC_SSL_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError(
            "The ssl_fusion classifier backend requires the IC_SSL_CHECKPOINT "
            "environment variable to point at a fine-tuned DINO SimCLR "
            "checkpoint directory."
        )
    return SSLFusionClassifier(checkpoint=checkpoint)
