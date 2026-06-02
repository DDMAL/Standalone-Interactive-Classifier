"""Feature extractor abstraction.

Defines a protocol that any feature extractor must satisfy, plus two
concrete implementations:

* :class:`HandcraftedExtractor` — the existing 29-dimensional geometry
  features (aspect ratio, volume, compactness, Hu moments, etc.).
  Zero extra dependencies; fast enough to recompute on every round.

* :class:`ViTExtractor` — CLS-token embeddings from a pretrained
  ``google/vit-base-patch16-224`` backbone, optionally with a LoRA
  checkpoint applied. Produces 768-dimensional vectors. Requires
  ``torch`` and ``transformers`` (and ``peft`` for LoRA weights).
  Expensive to run — use :class:`~ic_core.store.NpzStore` to cache
  pre-computed vectors rather than extracting per session.

Switching extractors::

    from ic_core.feature_extractor import HandcraftedExtractor, ViTExtractor
    from ic_core.classifier import InteractiveClassifier

    clf = InteractiveClassifier(extractor=ViTExtractor(checkpoint="path/to/lora"))
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ic_core.glyph import Glyph


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FeatureExtractorProtocol(Protocol):
    """Minimal interface any feature extractor must implement."""

    @property
    def dim(self) -> int:
        """Dimensionality of the output feature vector."""
        ...

    def extract_batch(self, glyphs: Sequence[Glyph]) -> np.ndarray:
        """Return an ``(N, dim)`` float64 feature matrix for ``glyphs``.

        The order of rows matches the order of ``glyphs``.
        If ``glyphs`` is empty, returns a ``(0, dim)`` array.
        """
        ...


# ---------------------------------------------------------------------------
# Handcrafted (existing 29-dim geometry features)
# ---------------------------------------------------------------------------


class HandcraftedExtractor:
    """Wraps the existing ``compute_features_batch`` (29-dim).

    No extra dependencies — pure numpy / scipy / scikit-image.
    Fast enough to run on every classify round.
    """

    @property
    def dim(self) -> int:
        return 29

    def extract_batch(self, glyphs: Sequence[Glyph]) -> np.ndarray:
        from ic_core.features import compute_features_batch
        return compute_features_batch(glyphs)

    def __repr__(self) -> str:
        return "HandcraftedExtractor(dim=29)"


# ---------------------------------------------------------------------------
# ViT (768-dim CLS-token embeddings)
# ---------------------------------------------------------------------------


class ViTExtractor:
    """CLS-token embeddings from ``google/vit-base-patch16-224``.

    Uses the pretrained backbone with no fine-tuning. Produces 768-dimensional
    vectors per glyph. Expensive to run — pre-compute and cache with
    :class:`~ic_core.store.NpzStore` rather than extracting per session.

    Args:
        device: Torch device string (default ``"cpu"``).
        batch_size: Glyphs per forward pass.

    Requires::

        pip install torch transformers
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32) -> None:
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import ViTModel, ViTImageProcessor
        except ImportError:
            raise ImportError(
                "ViTExtractor requires torch and transformers: "
                "pip install torch transformers"
            )
        self._processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self._model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self._model = self._model.to(self.device).eval()

    @property
    def dim(self) -> int:
        return 768

    def extract_batch(self, glyphs: Sequence[Glyph]) -> np.ndarray:
        """Run glyphs through the ViT and return CLS-token embeddings (N, 768)."""
        if not glyphs:
            return np.zeros((0, self.dim), dtype=np.float64)

        self._load()

        import torch
        from PIL import Image, ImageOps

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(glyphs), self.batch_size):
            batch = list(glyphs[start : start + self.batch_size])
            images = []
            for g in batch:
                arr = g.to_array()  # (H, W) bool
                # Ensure a minimum size so the processor doesn't see
                # degenerate 1×1 or 2×2 images (very small glyphs).
                pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB")
                min_side = 16
                if pil.width < min_side or pil.height < min_side:
                    pil = pil.resize(
                        (max(pil.width, min_side), max(pil.height, min_side)),
                        Image.NEAREST,
                    )
                # Pad to square so ViT sees consistent framing.
                side = max(pil.width, pil.height)
                pil = ImageOps.pad(pil, (side, side), color=(255, 255, 255))
                images.append(pil)

            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float64)
            all_embeddings.append(cls)

        return np.vstack(all_embeddings)

    def __repr__(self) -> str:
        return f"ViTExtractor(device={self.device!r})"
