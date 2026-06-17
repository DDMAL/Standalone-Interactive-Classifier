"""SimCLR contrastive fine-tuning of a pretrained ViT.

Instead of masked patch reconstruction (MIM/MAE), this script uses
contrastive self-supervised learning (SimCLR / NT-Xent loss) to fine-tune
the ViT backbone with LoRA adapters.

For each crop two independently-augmented views are created. The model is
trained to pull their CLS-token embeddings together while pushing apart
all other crops in the batch. This produces CLS tokens that are directly
usable for kNN classification — unlike MIM-tuned models which degrade kNN
quality (see: arXiv:2401.00463, arXiv:2304.07193).

Augmentations chosen for medieval manuscript glyphs:
  - RandomResizedCrop (scale 0.5–1.0): simulates scribal size variation
  - RandomRotation ±12°: simulates pen angle variation
  - GaussianBlur: simulates scan/parchment texture differences
  - ColorJitter (brightness/contrast only): simulates ink darkness variation
  - No horizontal/vertical flip — would change neume shape meaning

Usage::

    # ViT-tiny (default)
    python train_vit_simclr.py \\
        --model pretrained_models/vit-tiny-patch16-224 \\
        --crops-dir /path/to/ssl_crops \\
        --output-dir /path/to/vit_simclr_checkpoints \\
        --batch-size 128 --epochs 20 --num-workers 16 --n-crops 2679730

    # DINO ViT-small
    python train_vit_simclr.py \\
        --model pretrained_models/dino-vits16 \\
        --crops-dir /path/to/ssl_crops \\
        --output-dir /path/to/dino_simclr_checkpoints \\
        --batch-size 64 --epochs 20 --num-workers 16 --n-crops 2679730

After training, load with ViTExtractor (checkpoint type is LoRA, same as before)::

    extractor = ViTExtractor(checkpoint="path/to/checkpoints/epoch_020")

Dependencies::

    pip install torch torchvision transformers peft webdataset
"""
from __future__ import annotations

import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, ViTConfig, ViTModel

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise ImportError("Install peft: pip install peft")

try:
    import webdataset as wds
except ImportError:
    raise ImportError("Install webdataset: pip install webdataset")


# ---------------------------------------------------------------------------
# Augmentation pipeline — two independent views per crop
# ---------------------------------------------------------------------------

def _build_augment(image_size: int = 224):
    """Return a stochastic augmentation transform for one view."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomRotation(degrees=12, fill=255),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# Dataset — WebDataset tar shards, two views per sample
# ---------------------------------------------------------------------------

def _shard_pattern(crops_dir: Path) -> str:
    shards = sorted(crops_dir.glob("shard-*.tar"))
    if not shards:
        raise ValueError(f"No shard-*.tar files found in {crops_dir}")
    lo = int(shards[0].stem.split("-")[1])
    hi = int(shards[-1].stem.split("-")[1])
    digits = len(shards[0].stem.split("-")[1])
    return str(crops_dir / f"shard-{{{lo:0{digits}d}..{hi:0{digits}d}}}.tar")


def build_dataloader(
    crops_dir: Path,
    batch_size: int,
    num_workers: int,
    n_crops: int | None,
) -> tuple[DataLoader, int]:
    augment = _build_augment()

    def preprocess(sample):
        raw = next((v for k, v in sample.items()
                    if not k.startswith("__") and isinstance(v, bytes)), None)
        if raw is None:
            return None
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None
        w, h = image.size
        side = max(w, h, 32)
        # Pad to square with white background before augmenting
        padded = ImageOps.pad(image, (side, side), color=(255, 255, 255))
        view1 = augment(padded)
        view2 = augment(padded)
        return {"view1": view1, "view2": view2}

    pattern = _shard_pattern(crops_dir)
    print(f"Shard pattern: {pattern}")

    dataset = (
        wds.WebDataset(pattern, shardshuffle=500)
        .shuffle(1000)
        .map(preprocess, handler=wds.warn_and_continue)
        .select(lambda x: x is not None)
        .batched(batch_size, partial=False)
    )

    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers,
                        pin_memory=True, persistent_workers=True)

    shards = sorted(crops_dir.glob("shard-*.tar"))
    total = n_crops if n_crops else len(shards) * 1000
    steps = max(1, total // batch_size)
    return loader, steps


# ---------------------------------------------------------------------------
# NT-Xent loss (SimCLR)
# ---------------------------------------------------------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy loss.

    z1, z2: (B, D) L2-normalised projection vectors.
    For each sample i, its positive pair is z2[i]; all other 2B-2 are negatives.
    """
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2B, D)
    sim = torch.mm(z, z.T) / temperature                 # (2B, 2B)

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positive indices: for i in [0,B) → positive is i+B; for i in [B,2B) → i-B
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])
    loss = F.cross_entropy(sim, labels)
    return loss


# ---------------------------------------------------------------------------
# Model — backbone (LoRA) + projection head
# ---------------------------------------------------------------------------

class SimCLRViT(nn.Module):
    """ViT backbone with LoRA adapters + a 2-layer MLP projection head.

    Only the backbone is saved at checkpoint time; the projection head is
    discarded after training (same pattern as SimCLR / BYOL).
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        cls1 = self.backbone(pixel_values=view1).last_hidden_state[:, 0, :]
        cls2 = self.backbone(pixel_values=view2).last_hidden_state[:, 0, :]
        z1 = self.projector(cls1)
        z2 = self.projector(cls2)
        return z1, z2


def build_model(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    proj_dim: int,
) -> SimCLRViT:
    print(f"Loading {model_name}...")
    config = ViTConfig.from_pretrained(model_name)
    backbone = ViTModel.from_pretrained(model_name, ignore_mismatched_sizes=True)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    return SimCLRViT(backbone, hidden_size=config.hidden_size, proj_dim=proj_dim)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model_name: str,
    crops_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    temperature: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    proj_dim: int,
    num_workers: int,
    n_crops: int | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader, steps_per_epoch = build_dataloader(crops_dir, batch_size, num_workers, n_crops)
    print(f"~{steps_per_epoch} batches/epoch")

    model = build_model(model_name, lora_r, lora_alpha, lora_dropout, proj_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader, 1):
            view1 = batch["view1"].to(device)
            view2 = batch["view2"].to(device)

            z1, z2 = model(view1, view2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % 200 == 0:
                print(f"  step {step}/{steps_per_epoch}  loss={loss.item():.4f}")

            if step >= steps_per_epoch:
                break

        scheduler.step()
        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Save only the LoRA backbone — projection head is discarded (not needed at eval)
        ckpt_dir = output_dir / f"epoch_{epoch:03d}"
        model.backbone.save_pretrained(ckpt_dir)
        print(f"  Saved → {ckpt_dir}/")

    print(f"\nTraining complete. Checkpoints in {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="SimCLR contrastive fine-tuning of ViT with LoRA on music crops."
    )
    parser.add_argument("--model", type=str, default="WinKawaks/vit-tiny-patch16-224")
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("vit_simclr_checkpoints"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Larger batches = more negatives = better NT-Xent (use 256+ if VRAM allows)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="NT-Xent temperature (default 0.07 from SimCLR paper)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--proj-dim", type=int, default=128,
                        help="Projection head output dimension (discarded after training)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-crops", type=int, default=None)
    args = parser.parse_args()

    train(
        model_name=args.model,
        crops_dir=args.crops_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        proj_dim=args.proj_dim,
        num_workers=args.num_workers,
        n_crops=args.n_crops,
    )


if __name__ == "__main__":
    main()
