"""Simple MLP classifier implemented in pure numpy.

Conforms to :class:`~ic_core.evaluation.ClassifierProtocol` so it can be
dropped into :func:`~ic_core.evaluation.cross_validate` or
:func:`~ic_core.evaluation.evaluate` as a drop-in replacement for the kNN.

Architecture: input (29) → [hidden layers with ReLU] → output (n_classes, softmax).
Optimiser: mini-batch Adam with cross-entropy loss and optional L2 regularisation.

Example — cross-validation with an MLP::

    from ic_core.evaluation import cross_validate, print_report
    from ic_core.nn_classifier import mlp_factory
    from ic_core.io_xml import load_glyphs

    glyphs = load_glyphs(Path("training_data.xml"))
    result = cross_validate(glyphs, classifier_factory=mlp_factory())
    print_report(result)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ic_core.classifier import UNCLASSIFIED
from ic_core.features import compute_features_batch
from ic_core.glyph import Glyph

# ---------------------------------------------------------------------------
# Prediction result (mirrors ic_core.classifier.Prediction interface)
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class NNPrediction:
    class_name: str
    confidence: float  # max softmax probability


# ---------------------------------------------------------------------------
# MLP internals
# ---------------------------------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _softmax(x: np.ndarray) -> np.ndarray:
    # Row-wise stable softmax
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy loss. targets is a 1-D int array of class indices."""
    n = len(targets)
    clipped = np.clip(probs[np.arange(n), targets], 1e-12, 1.0)
    return float(-np.log(clipped).mean())


class _AdamState:
    """Per-parameter Adam moment accumulators."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0

    def step(
        self,
        grad: np.ndarray,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> np.ndarray:
        self.t += 1
        self.m = beta1 * self.m + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * grad**2
        m_hat = self.m / (1 - beta1**self.t)
        v_hat = self.v / (1 - beta2**self.t)
        return lr * m_hat / (np.sqrt(v_hat) + eps)


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------


class MLPClassifier:
    """A multi-layer perceptron trained with mini-batch Adam.

    Args:
        hidden_sizes: Sequence of hidden layer widths, e.g. ``(64, 32)``.
        epochs: Number of full passes over the training data.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        l2: L2 weight-decay coefficient.
        seed: RNG seed for reproducible weight initialisation.
        verbose: If True, print loss every 10 epochs.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (128, 64),
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        l2: float = 1e-4,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        self.verbose = verbose

        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []
        self._classes: list[str] = []

        # feature standardisation params
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, training_glyphs: Sequence[Glyph]) -> "MLPClassifier":
        if not training_glyphs:
            raise ValueError("Cannot fit MLPClassifier with zero training glyphs.")

        rng = np.random.default_rng(self.seed)

        # Build label index
        self._classes = sorted({g.class_name for g in training_glyphs})
        label_to_idx = {c: i for i, c in enumerate(self._classes)}
        n_classes = len(self._classes)

        # Feature matrix
        X = compute_features_batch(training_glyphs).astype(np.float64)
        y = np.array([label_to_idx[g.class_name] for g in training_glyphs], dtype=np.int64)

        # Standardise
        self._mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-12] = 1e-12
        self._std = std
        X = (X - self._mean) / self._std

        # Initialise weights (He initialisation for ReLU layers)
        layer_sizes = [X.shape[1], *self.hidden_sizes, n_classes]
        self._weights = []
        self._biases = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)
            self._weights.append(rng.normal(0.0, scale, (fan_in, fan_out)))
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

        # Adam state per parameter tensor
        w_adam = [_AdamState(w.shape) for w in self._weights]
        b_adam = [_AdamState(b.shape) for b in self._biases]

        n = len(X)
        for epoch in range(self.epochs):
            # Shuffle
            perm = rng.permutation(n)
            X, y = X[perm], y[perm]

            for start in range(0, n, self.batch_size):
                Xb = X[start : start + self.batch_size]
                yb = y[start : start + self.batch_size]

                # Forward pass — store pre-activation values for backprop
                activations = [Xb]
                pre_acts = []
                h = Xb
                for W, b in zip(self._weights[:-1], self._biases[:-1]):
                    z = h @ W + b
                    pre_acts.append(z)
                    h = _relu(z)
                    activations.append(h)

                # Output layer (no activation — softmax in loss)
                z_out = h @ self._weights[-1] + self._biases[-1]
                probs = _softmax(z_out)

                # Backward pass
                nb = len(Xb)
                delta = probs.copy()
                delta[np.arange(nb), yb] -= 1.0
                delta /= nb

                dW_list = []
                db_list = []
                for layer_idx in range(len(self._weights) - 1, -1, -1):
                    a_in = activations[layer_idx]
                    dW = a_in.T @ delta + self.l2 * self._weights[layer_idx]
                    db = delta.sum(axis=0)
                    dW_list.insert(0, dW)
                    db_list.insert(0, db)
                    if layer_idx > 0:
                        delta = (delta @ self._weights[layer_idx].T) * _relu_grad(
                            pre_acts[layer_idx - 1]
                        )

                # Adam updates
                for i, (dW, db) in enumerate(zip(dW_list, db_list)):
                    self._weights[i] -= w_adam[i].step(dW, self.lr)
                    self._biases[i] -= b_adam[i].step(db, self.lr)

            if self.verbose and (epoch + 1) % 10 == 0:
                probs_all = self._forward(X)
                loss = _cross_entropy(probs_all, y)
                acc = float((probs_all.argmax(axis=1) == y).mean())
                print(f"  epoch {epoch + 1:>4}/{self.epochs}  loss={loss:.4f}  train_acc={acc:.4f}")

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = X
        for W, b in zip(self._weights[:-1], self._biases[:-1]):
            h = _relu(h @ W + b)
        return _softmax(h @ self._weights[-1] + self._biases[-1])

    def predict_many(self, glyphs: Sequence[Glyph]) -> list[NNPrediction]:
        if not glyphs:
            return []
        if not self._weights:
            raise RuntimeError("MLPClassifier is not trained; call .fit() first.")

        assert self._mean is not None and self._std is not None
        X = compute_features_batch(glyphs).astype(np.float64)
        X = (X - self._mean) / self._std

        probs = self._forward(X)
        idxs = probs.argmax(axis=1)
        return [
            NNPrediction(
                class_name=self._classes[int(idx)],
                confidence=float(probs[i, idx]),
            )
            for i, idx in enumerate(idxs)
        ]

    def predict(self, glyph: Glyph) -> NNPrediction:
        return self.predict_many([glyph])[0]


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def mlp_factory(
    hidden_sizes: tuple[int, ...] = (128, 64),
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    l2: float = 1e-4,
    seed: int = 42,
    verbose: bool = False,
) -> "ClassifierFactory":
    """Return a factory that builds a fresh :class:`MLPClassifier` each call.

    Args:
        hidden_sizes: Hidden layer widths (default ``(128, 64)``).
        epochs: Training epochs per fold (default 100).
        lr: Adam learning rate (default 1e-3).
        batch_size: Mini-batch size (default 64).
        l2: L2 weight decay (default 1e-4).
        seed: Weight init seed (default 42).
        verbose: Print per-epoch loss/accuracy during training.
    """
    def _factory() -> MLPClassifier:
        return MLPClassifier(
            hidden_sizes=hidden_sizes,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            l2=l2,
            seed=seed,
            verbose=verbose,
        )

    layers = "-".join(str(h) for h in hidden_sizes)
    _factory.__name__ = f"MLP({layers}, epochs={epochs})"
    return _factory


# type alias re-export for callers that import from here
from typing import Callable
ClassifierFactory = Callable[[], MLPClassifier]
