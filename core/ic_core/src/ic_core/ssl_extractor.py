"""Optional SSL (self-supervised) feature extractor for glyph images.

This module is entirely separate from the rest of ``ic_core`` and is
**not imported by anything on the default (kNN) classification path**.
It requires ``torch`` and ``transformers``, which are NOT core
dependencies of this package -- install them via the ``ssl`` extra
(``pip install ic-core[ssl]``) before using anything in this module.

Supports loading a fine-tuned checkpoint directory in either of two
formats, auto-detected from its contents:

  - ``adapter_config.json`` present -> a PEFT/LoRA-style checkpoint,
    loaded via the ``peft`` library on top of a HuggingFace base model.
  - ``method.json`` + ``backbone.pt`` present -> a custom
    Adapter/SSF-style checkpoint (Houlsby bottleneck adapters or
    scale-and-shift), loaded by hand.

If ``checkpoint`` is ``None``, the plain pretrained backbone is used
(no fine-tuning).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

from ic_core.glyph import Glyph
from ic_core.image import png_base64_to_array

#: Maps the Narval-local model names baked into fine-tuned checkpoints
#: to their public HuggingFace equivalents.
_NARVAL_TO_HF = {
    "pretrained_models/vit-tiny-patch16-224": "WinKawaks/vit-tiny-patch16-224",
    "pretrained_models/dino-vits16": "facebook/dino-vits16",
}

_DEFAULT_MODEL_NAME = "WinKawaks/vit-tiny-patch16-224"


def _require_ssl_deps():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The SSL feature extractor requires torch and transformers, "
            "which are optional dependencies of ic-core. Install them with "
            "`pip install ic-core[ssl]` before using ic_core.ssl_extractor."
        ) from exc


def crop_glyph_to_square(glyph: Glyph, padding: int = 4) -> Image.Image:
    """Decode ``glyph.image_gray_b64`` and pad it to a square RGB image.

    Raises:
        ValueError: If the glyph has no stored real-pixel crop -- it
            must have been ingested with ``store_real_crop=True``
            (see ``ic_core.ingest``).
    """
    if glyph.image_gray_b64 is None:
        raise ValueError(
            f"Glyph {glyph.id!r} has no real-pixel crop (image_gray_b64 is "
            "None). Re-ingest the page with store_real_crop=True to use the "
            "SSL feature extractor on this glyph."
        )
    arr = png_base64_to_array(glyph.image_gray_b64)
    mode = "L" if arr.ndim == 2 else "RGB"
    img = Image.fromarray(arr, mode=mode).convert("RGB")
    side = max(img.width, img.height, 16) + 2 * padding
    return ImageOps.pad(img, (side, side), color=(255, 255, 255))


class ViTExtractor:
    """A (possibly fine-tuned) Vision Transformer feature extractor.

    Args:
        checkpoint: Path to a fine-tuned checkpoint directory, or
            ``None`` for the plain pretrained backbone.
        model_name: HuggingFace model id to use when ``checkpoint`` is
            ``None``, or as a fallback if the checkpoint doesn't
            record its own base model.
        batch_size: Glyphs processed per forward pass.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        batch_size: int = 32,
    ) -> None:
        _require_ssl_deps()
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._processor = None
        self._dim: int | None = None

        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _checkpoint_type(self) -> str:
        if self.checkpoint is None:
            return "none"
        if (self.checkpoint / "adapter_config.json").exists():
            return "lora"
        if (self.checkpoint / "method.json").exists():
            return "adapter_ssf"
        raise FileNotFoundError(
            f"Checkpoint directory '{self.checkpoint}' contains neither "
            "'adapter_config.json' (LoRA/PEFT) nor 'method.json' (adapter/SSF)."
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoImageProcessor, ViTModel

        ckpt_type = self._checkpoint_type()
        effective_model = self.model_name

        if ckpt_type == "lora":
            cfg = json.loads((self.checkpoint / "adapter_config.json").read_text())
            raw = cfg.get("base_model_name_or_path", self.model_name)
            effective_model = _NARVAL_TO_HF.get(raw, raw)
        elif ckpt_type == "adapter_ssf":
            meta_peek = json.loads((self.checkpoint / "method.json").read_text())
            raw = meta_peek.get("model_name", self.model_name)
            effective_model = _NARVAL_TO_HF.get(raw, raw)

        self._processor = AutoImageProcessor.from_pretrained(effective_model)

        if ckpt_type == "lora":
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "Loading a LoRA/PEFT checkpoint requires the peft "
                    "package: pip install ic-core[ssl]"
                ) from exc
            base = AutoModel.from_pretrained(effective_model, ignore_mismatched_sizes=True)
            self._model = PeftModel.from_pretrained(base, str(self.checkpoint))
        elif ckpt_type == "adapter_ssf":
            import torch

            meta = json.loads((self.checkpoint / "method.json").read_text())
            method = meta["method"]
            bottleneck = meta.get("bottleneck", 64)

            backbone = ViTModel.from_pretrained(effective_model, ignore_mismatched_sizes=True)
            if method == "adapter":
                backbone = _apply_adapters(backbone, bottleneck)
            elif method == "ssf":
                backbone = _apply_ssf(backbone)
            else:
                raise ValueError(f"Unknown method '{method}' in method.json.")

            state = torch.load(
                self.checkpoint / "backbone.pt", map_location="cpu", weights_only=True
            )
            backbone.load_state_dict(state)
            self._model = backbone
        else:
            self._model = AutoModel.from_pretrained(self.model_name, ignore_mismatched_sizes=True)

        self._model = self._model.to(self.device).eval()
        import torch
        dummy = torch.zeros(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            out = self._model(pixel_values=dummy)
        self._dim = out.last_hidden_state.shape[-1]

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim

    def extract_batch(
        self, glyphs: Sequence[Glyph], pooling: str = "cls_mean"
    ) -> np.ndarray:
        """Extract pooled features for a batch of glyphs.

        Args:
            glyphs: Glyphs with ``image_gray_b64`` populated (see
                :func:`crop_glyph_to_square`).
            pooling: ``"cls"`` (CLS token only, dim ``D``), ``"mean"``
                (mean over patch tokens, dim ``D``), or ``"cls_mean"``
                (concatenated, dim ``2D`` -- the default, matching the
                configuration validated during model selection).

        Returns:
            ``(N, D)`` or ``(N, 2D)`` float64 array, one row per glyph.
        """
        import torch

        self._load()
        images = [crop_glyph_to_square(g) for g in glyphs]

        all_embeddings = []
        for start in range(0, len(images), self.batch_size):
            batch_imgs = images[start : start + self.batch_size]
            inputs = self._processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model(**inputs)
            hidden = out.last_hidden_state
            cls_tok = hidden[:, 0, :]
            mean_tok = hidden[:, 1:, :].mean(dim=1)
            if pooling == "cls":
                vec = cls_tok
            elif pooling == "mean":
                vec = mean_tok
            elif pooling == "cls_mean":
                vec = torch.cat([cls_tok, mean_tok], dim=-1)
            else:
                raise ValueError(f"Unknown pooling {pooling!r}. Choose cls|mean|cls_mean.")
            all_embeddings.append(vec.cpu().numpy().astype(np.float64))
        return np.vstack(all_embeddings)

    def __repr__(self) -> str:
        ckpt = f", checkpoint={self.checkpoint!r}" if self.checkpoint else ""
        return f"ViTExtractor({self.model_name!r}{ckpt})"


def extract_ssl_embeddings(
    glyphs: Sequence[Glyph],
    checkpoint: str | Path | None,
    pooling: str = "cls_mean",
) -> np.ndarray:
    """Pure SSL feature vectors for ``glyphs``, one row per glyph in order.

    For each glyph, prefers an already-attached ``ssl_embedding`` (see
    ``ic_core.ssl_preset_embeddings`` -- set for SSL-compatible presets or
    uploads paired with a companion ``.ssl_embeddings.npz``) over a live
    model pass; glyphs without one are extracted via
    :meth:`ViTExtractor.extract_batch` on their ``image_gray_b64`` crop.
    A mixed batch of both is handled in one call.

    This is the standalone counterpart to the same precomputed-vs-live
    logic ``ic_core.ssl_classifier.SSLFusionClassifier`` uses internally
    while fitting/predicting -- exposed here so a caller that just wants
    the SSL vectors themselves (e.g. to export a companion embeddings
    file for later reuse) doesn't need to go through a classifier.

    Raises:
        ValueError: If any glyph has neither ``ssl_embedding`` nor
            ``image_gray_b64``.
    """
    missing = [
        g.id for g in glyphs if g.ssl_embedding is None and g.image_gray_b64 is None
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(glyphs)} glyphs have neither a "
            "precomputed ssl_embedding nor a real-pixel crop "
            "(image_gray_b64), so no SSL feature vector can be produced "
            f"for them: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    live = [g for g in glyphs if g.ssl_embedding is None]
    if not live:
        return np.asarray([g.ssl_embedding for g in glyphs], dtype=np.float64)

    extractor = ViTExtractor(checkpoint=checkpoint)
    precomputed = [g for g in glyphs if g.ssl_embedding is not None]
    if not precomputed:
        return extractor.extract_batch(glyphs, pooling=pooling)

    dim = len(precomputed[0].ssl_embedding)
    out = np.empty((len(glyphs), dim), dtype=np.float64)
    idx_precomputed = [i for i, g in enumerate(glyphs) if g.ssl_embedding is not None]
    idx_live = [i for i, g in enumerate(glyphs) if g.ssl_embedding is None]
    out[idx_precomputed] = np.asarray(
        [g.ssl_embedding for g in precomputed], dtype=np.float64
    )
    out[idx_live] = extractor.extract_batch(live, pooling=pooling)
    return out


def _apply_adapters(backbone, bottleneck: int = 64):
    """Inject Houlsby bottleneck adapters and return the modified backbone."""
    import torch.nn as nn

    class _AdapterModule(nn.Module):
        def __init__(self, hidden_size, bottleneck):
            super().__init__()
            self.down = nn.Linear(hidden_size, bottleneck)
            self.act = nn.GELU()
            self.up = nn.Linear(bottleneck, hidden_size)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        def forward(self, x):
            return x + self.up(self.act(self.down(x)))

    hidden_size = backbone.config.hidden_size
    adapters = []
    for layer in backbone.layers:
        a_attn = _AdapterModule(hidden_size, bottleneck)
        a_ffn = _AdapterModule(hidden_size, bottleneck)
        adapters += [a_attn, a_ffn]
        layer.attention.register_forward_hook(
            lambda m, inp, out, a=a_attn: (a(out[0]),) + out[1:]
        )
        layer.mlp.register_forward_hook(
            lambda m, inp, out, a=a_ffn: a(out)
        )
    backbone.adapters = nn.ModuleList(adapters)
    return backbone


def _apply_ssf(backbone):
    """Wrap every linear layer in each transformer block with SSF scale+shift."""
    import torch
    import torch.nn as nn

    class _SSFLinear(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear
            self.scale = nn.Parameter(torch.ones(linear.out_features))
            self.shift = nn.Parameter(torch.zeros(linear.out_features))

        def forward(self, x):
            return self.linear(x) * self.scale + self.shift

    for layer in backbone.layers:
        attn = layer.attention
        attn.q_proj = _SSFLinear(attn.q_proj)
        attn.k_proj = _SSFLinear(attn.k_proj)
        attn.v_proj = _SSFLinear(attn.v_proj)
        attn.o_proj = _SSFLinear(attn.o_proj)
        layer.mlp.fc1 = _SSFLinear(layer.mlp.fc1)
        layer.mlp.fc2 = _SSFLinear(layer.mlp.fc2)
    return backbone
