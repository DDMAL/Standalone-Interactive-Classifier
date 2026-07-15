"""Optional SSL+HC fused feature classifier -- an alternative backend
to the default :class:`ic_core.classifier.InteractiveClassifier` kNN.

This module is entirely separate from the rest of ``ic_core`` and is
**not imported by anything on the default classification path**. It
requires ``scikit-learn`` (and, transitively via
:mod:`ic_core.ssl_extractor`, ``torch``/``transformers``), none of
which are core dependencies of this package -- install them via the
``ssl`` extra (``pip install ic-core[ssl]``) before using this module.

Model selection summary (see the accompanying paper/report for the
full methodology): DINO SimCLR checkpoint at epoch 11, final-layer
CLS+mean pooling, fused with 29-dim handcrafted features via weighted
concatenation (HC block scaled by ``hc_weight=4.0`` before
concatenation, to correct for the 768:29 dimensionality imbalance
between the two feature blocks), classified with L2-regularized
logistic regression (``C=0.3``) -- all hyperparameters selected purely
via cross-validation on held-out training manuscripts, never on data
this classifier would later be evaluated against.

Exposes the same public shape as
:class:`ic_core.classifier.InteractiveClassifier` --
``.fit()``/``.predict()``/``.predict_many()`` plus
``is_trained``/``training_size``/``classes`` -- so it is a drop-in
alternative wherever an ``InteractiveClassifier``-like object is
expected (see ``run_correction_stage``'s ``classifier_factory``
parameter).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ic_core.classifier import Prediction
from ic_core.features import compute_features_batch
from ic_core.glyph import Glyph

#: Selected via cross-validation on held-out training manuscripts --
#: see module docstring. Not re-tuned per deployment; override via the
#: constructor if you have reason to.
DEFAULT_HC_WEIGHT: float = 4.0
DEFAULT_C: float = 0.3


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
    """SSL+HC fused features, classified with logistic regression.

    Args:
        checkpoint: Path to the DINO SimCLR fine-tuned checkpoint
            directory (or ``None`` to fall back to the plain
            pretrained backbone -- not recommended, see the extractor
            comparison in the accompanying report).
        hc_weight: Scalar applied to the standardized handcrafted
            feature block before concatenating with the standardized
            SSL block. Defaults to the cross-validated value; only
            override this if you're re-running model selection.
        C: Inverse regularization strength for the logistic
            regression classifier. Defaults to the cross-validated
            value.

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

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _get_extractor(self):
        if self._ssl_extractor is None:
            from ic_core.ssl_extractor import ViTExtractor
            self._ssl_extractor = ViTExtractor(checkpoint=self.checkpoint)
        return self._ssl_extractor

    def _fused_features(self, glyphs: Sequence[Glyph], fit_scalers: bool) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler

        X_hc = compute_features_batch(glyphs)
        X_ssl = self._get_extractor().extract_batch(glyphs, pooling="cls_mean")

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
        """
        from sklearn.linear_model import LogisticRegression

        if not training_glyphs:
            raise ValueError(
                "Cannot fit SSLFusionClassifier with zero training glyphs"
            )

        X = self._fused_features(training_glyphs, fit_scalers=True)
        y = np.asarray([g.class_name for g in training_glyphs], dtype=object)

        clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=500)
        clf.fit(X, y)

        self._clf = clf
        self._classes = tuple(sorted({str(label) for label in y.tolist()}))
        self._training_size = len(training_glyphs)
        return self

    def predict(self, glyph: Glyph) -> Prediction:
        """Classify a single glyph. See :meth:`predict_many`."""
        return self.predict_many([glyph])[0]

    def predict_many(self, glyphs: Sequence[Glyph]) -> list[Prediction]:
        """Classify a batch of glyphs in one feature-extraction pass."""
        self._require_trained()
        if not glyphs:
            return []

        X = self._fused_features(glyphs, fit_scalers=False)
        proba = self._clf.predict_proba(X)
        classes = self._clf.classes_
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
