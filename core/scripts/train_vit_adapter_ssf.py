"""PEFT fine-tuning of ViT: Adapter and SSF approaches.

Same masked patch prediction (MIM) objective as train_vit_lora.py, but
with two alternative parameter-efficient methods:

  adapter  — Houlsby bottleneck adapters injected after the attention and
             FFN sub-layers of every transformer block.  Learns a down-
             projection, GELU, and up-projection (initialised near-zero so
             training starts close to the pretrained model).  ~0.5 M extra
             params for ViT-tiny with bottleneck=64.

  ssf      — Scale and Shift Features (Lian et al., 2022): wraps every
             linear layer in the transformer with learnable per-channel
             scale and shift vectors applied after the projection.
             Very few parameters (~20 k for ViT-tiny) and no architectural
             changes beyond those multiplications.

Checkpoints are saved as::

    <output-dir>/epoch_NNN/
        method.json     — {"method": "adapter"|"ssf", "model_name": ..., ...}
        backbone.pt     — full backbone state_dict (base + injected params)

Load them in ViTExtractor by passing the epoch directory as ``checkpoint``.

Usage::

    python train_vit_adapter_ssf.py --method adapter \\
        --crops-dir /path/to/ssl_crops --output-dir checkpoints/adapter

    python train_vit_adapter_ssf.py --method ssf \\
        --crops-dir /path/to/ssl_crops --output-dir checkpoints/ssf

Dependencies::

    pip install torch transformers webdataset
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTConfig, ViTModel

try:
    import webdataset as wds
except ImportError:
    raise ImportError("Install webdataset: pip install webdataset")

NUM_PATCHES = 196
PATCH_SIZE  = 16
CHANNELS    = 3


# ---------------------------------------------------------------------------
# Adapter (Houlsby)
# ---------------------------------------------------------------------------

class AdapterModule(nn.Module):
    """Bottleneck adapter: down → GELU → up + residual.

    Up-projection is zero-initialised so the adapter starts as a no-op.
    """

    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


def apply_adapters(backbone: ViTModel, bottleneck: int) -> ViTModel:
    """Inject adapter modules after attention-output and FFN-output in every layer.

    Adapters are stored in ``backbone.adapters`` so their parameters are
    part of the state dict.  Forward hooks route activations through them.
    """
    hidden_size = backbone.config.hidden_size
    adapters: list[AdapterModule] = []

    for layer in backbone.layers:
        a_attn = AdapterModule(hidden_size, bottleneck)
        a_ffn  = AdapterModule(hidden_size, bottleneck)
        adapters += [a_attn, a_ffn]

        # attention returns (hidden_states, attn_weights) tuple
        layer.attention.register_forward_hook(
            lambda m, inp, out, a=a_attn: (a(out[0]),) + out[1:]
        )
        layer.mlp.register_forward_hook(
            lambda m, inp, out, a=a_ffn: a(out)
        )

    backbone.adapters = nn.ModuleList(adapters)

    for name, p in backbone.named_parameters():
        if "adapters" not in name:
            p.requires_grad_(False)

    n = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Adapter — trainable params: {n:,}")
    return backbone


# ---------------------------------------------------------------------------
# SSF (Scale and Shift Features)
# ---------------------------------------------------------------------------

class SSFLinear(nn.Module):
    """Linear layer with per-output-channel affine SSF transform: scale*y + shift."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear
        self.scale  = nn.Parameter(torch.ones(linear.out_features))
        self.shift  = nn.Parameter(torch.zeros(linear.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale + self.shift


def apply_ssf(backbone: ViTModel) -> ViTModel:
    """Wrap every linear layer in each transformer block with SSFLinear.

    Targets:
      - Attention Q, K, V, and output projections
      - FFN fc1 and fc2

    All original weights are frozen; only scale/shift parameters train.
    """
    for layer in backbone.layers:
        attn = layer.attention
        attn.q_proj = SSFLinear(attn.q_proj)
        attn.k_proj = SSFLinear(attn.k_proj)
        attn.v_proj = SSFLinear(attn.v_proj)
        attn.o_proj = SSFLinear(attn.o_proj)
        layer.mlp.fc1 = SSFLinear(layer.mlp.fc1)
        layer.mlp.fc2 = SSFLinear(layer.mlp.fc2)

    for name, p in backbone.named_parameters():
        if "scale" not in name and "shift" not in name:
            p.requires_grad_(False)

    n = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"SSF — trainable params: {n:,}")
    return backbone


# ---------------------------------------------------------------------------
# Shared: MaskedViT, build / save / load
# ---------------------------------------------------------------------------

class MaskedViT(nn.Module):
    """ViT backbone + linear decoder for masked patch reconstruction."""

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder  = nn.Linear(hidden_size, PATCH_SIZE * PATCH_SIZE * CHANNELS)

    def forward(
        self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor
    ) -> torch.Tensor:
        outputs      = self.backbone(pixel_values=pixel_values)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # skip CLS
        masked_tokens   = patch_tokens[bool_masked_pos]
        reconstructed   = self.decoder(masked_tokens)

        B, p = pixel_values.shape[0], PATCH_SIZE
        x = pixel_values.reshape(B, CHANNELS, 14, p, 14, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, NUM_PATCHES, CHANNELS * p * p)
        targets = x[bool_masked_pos]
        return nn.functional.mse_loss(reconstructed, targets)


def build_model(model_name: str, method: str, bottleneck: int) -> MaskedViT:
    print(f"Loading {model_name}  method={method} ...")
    config   = ViTConfig.from_pretrained(model_name)
    backbone = ViTModel.from_pretrained(model_name, ignore_mismatched_sizes=True)

    if method == "adapter":
        backbone = apply_adapters(backbone, bottleneck)
    elif method == "ssf":
        backbone = apply_ssf(backbone)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'adapter' or 'ssf'.")

    return MaskedViT(backbone, hidden_size=config.hidden_size)


def save_checkpoint(
    model: MaskedViT, ckpt_dir: Path, method: str, model_name: str, bottleneck: int
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta = {"method": method, "model_name": model_name, "bottleneck": bottleneck}
    (ckpt_dir / "method.json").write_text(json.dumps(meta, indent=2))
    torch.save(model.backbone.state_dict(), ckpt_dir / "backbone.pt")


def load_backbone(ckpt_dir: Path) -> ViTModel:
    """Reconstruct the fine-tuned backbone from a saved checkpoint directory."""
    meta       = json.loads((ckpt_dir / "method.json").read_text())
    method     = meta["method"]
    model_name = meta["model_name"]
    bottleneck = meta.get("bottleneck", 64)

    backbone = ViTModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
    if method == "adapter":
        apply_adapters(backbone, bottleneck)
    elif method == "ssf":
        apply_ssf(backbone)
    else:
        raise ValueError(f"Unknown method '{method}' in checkpoint.")
    # suppress requires_grad changes — we just want the structure

    state = torch.load(ckpt_dir / "backbone.pt", map_location="cpu")
    backbone.load_state_dict(state)
    return backbone


# ---------------------------------------------------------------------------
# Dataset (identical to train_vit_lora.py)
# ---------------------------------------------------------------------------

def _shard_pattern(crops_dir: Path) -> str:
    shards = sorted(crops_dir.glob("shard-*.tar"))
    if not shards:
        raise ValueError(f"No shard-*.tar files in {crops_dir}")
    lo = int(shards[0].stem.split("-")[1])
    hi = int(shards[-1].stem.split("-")[1])
    d  = len(shards[0].stem.split("-")[1])
    return str(crops_dir / f"shard-{{{lo:0{d}d}..{hi:0{d}d}}}.tar")


def build_dataloader(
    crops_dir: Path,
    processor: AutoImageProcessor,
    mask_ratio: float,
    batch_size: int,
    num_workers: int,
    n_crops: int | None,
) -> tuple[DataLoader, int]:

    def preprocess(sample):
        raw = next(
            (v for k, v in sample.items() if not k.startswith("__") and isinstance(v, bytes)),
            None,
        )
        if raw is None:
            return None
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None
        side   = max(image.size)
        padded = ImageOps.pad(image, (side, side), color=(255, 255, 255))
        pv     = processor(images=padded, return_tensors="pt")["pixel_values"].squeeze(0)
        n_mask = int(mask_ratio * NUM_PATCHES)
        mask   = torch.zeros(NUM_PATCHES, dtype=torch.bool)
        mask[torch.randperm(NUM_PATCHES)[:n_mask]] = True
        return {"pixel_values": pv, "bool_masked_pos": mask}

    pattern = _shard_pattern(crops_dir)
    print(f"Shard pattern: {pattern}")
    dataset = (
        wds.WebDataset(pattern, shardshuffle=False)
        .shuffle(1000)
        .map(preprocess, handler=wds.warn_and_continue)
        .select(lambda x: x is not None)
        .batched(batch_size, partial=False)
    )
    loader = DataLoader(
        dataset, batch_size=None, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )
    shards = sorted(crops_dir.glob("shard-*.tar"))
    total  = n_crops if n_crops else len(shards) * 1000
    steps  = max(1, total // batch_size)
    return loader, steps


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model_name: str,
    method: str,
    crops_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    mask_ratio: float,
    bottleneck: int,
    num_workers: int,
    n_crops: int | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor               = AutoImageProcessor.from_pretrained(model_name)
    loader, steps_per_epoch = build_dataloader(
        crops_dir, processor, mask_ratio, batch_size, num_workers, n_crops
    )
    print(f"~{steps_per_epoch} batches/epoch")

    model     = build_model(model_name, method, bottleneck).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader, 1):
            pv   = batch["pixel_values"].to(device)
            mask = batch["bool_masked_pos"].to(device)
            loss = model(pv, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step >= steps_per_epoch:
                break

        scheduler.step()
        avg = total_loss / steps_per_epoch
        print(f"Epoch {epoch}/{epochs}  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        ckpt_dir = output_dir / f"epoch_{epoch:03d}"
        save_checkpoint(model, ckpt_dir, method, model_name, bottleneck)
        print(f"  Saved → {ckpt_dir}/")

    print(f"\nDone. Checkpoints in {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="PEFT fine-tuning of ViT (Adapter or SSF) with masked patch prediction."
    )
    parser.add_argument("--method",      choices=["adapter", "ssf"], required=True)
    parser.add_argument("--model",       default="WinKawaks/vit-tiny-patch16-224")
    parser.add_argument("--crops-dir",   type=Path, required=True)
    parser.add_argument("--output-dir",  type=Path, default=Path("vit_peft_checkpoints"))
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--mask-ratio",  type=float, default=0.75)
    parser.add_argument("--bottleneck",  type=int,   default=64,
                        help="Adapter bottleneck size (ignored for --method ssf).")
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--n-crops",     type=int,   default=None)
    args = parser.parse_args()

    train(
        model_name  = args.model,
        method      = args.method,
        crops_dir   = args.crops_dir,
        output_dir  = args.output_dir,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        mask_ratio  = args.mask_ratio,
        bottleneck  = args.bottleneck,
        num_workers = args.num_workers,
        n_crops     = args.n_crops,
    )


if __name__ == "__main__":
    main()
