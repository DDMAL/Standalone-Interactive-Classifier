"""LoRA fine-tuning of a pretrained ViT using masked patch prediction.

Uses a pretrained ViT (google/vit-base-patch16-224) as the backbone and
applies LoRA adapters to the attention layers. Trains with a masked image
modelling objective: randomly masks patches and learns to reconstruct them.
This adapts the pretrained ViT's representations toward manuscript music
notation without needing a large labelled dataset.

Usage::

    python train_vit_lora.py \\
        --crops-dir /path/to/music_crops \\
        --output-dir /path/to/checkpoints

Dependencies::

    pip install torch torchvision transformers peft
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import ViTConfig, ViTForMaskedImageModeling, ViTImageProcessor

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise ImportError("Install peft: pip install peft")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ViT-base/16 divides 224×224 into 14×14 = 196 patches.
NUM_PATCHES = 196


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MusicCropDataset(Dataset):
    """Loads music crop images, pads to square, resizes to 224×224.

    Padding uses white (255) to match parchment backgrounds rather than
    the default black padding, preserving aspect ratio as neutral whitespace.
    """

    def __init__(self, crops_dir: Path, processor: ViTImageProcessor, mask_ratio: float = 0.75):
        self.paths = sorted(
            p for p in crops_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No images found in {crops_dir}")
        self.processor = processor
        self.mask_ratio = mask_ratio

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.paths[idx]).convert("RGB")

        # Pad to square with white background to preserve aspect ratio.
        w, h = image.size
        side = max(w, h)
        padded = ImageOps.pad(image, (side, side), color=(255, 255, 255))

        inputs = self.processor(images=padded, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (3, 224, 224)

        # Random boolean mask over patches: True = masked.
        num_masked = int(self.mask_ratio * NUM_PATCHES)
        mask = torch.zeros(NUM_PATCHES, dtype=torch.bool)
        masked_indices = random.sample(range(NUM_PATCHES), num_masked)
        mask[masked_indices] = True

        return {"pixel_values": pixel_values, "bool_masked_pos": mask}


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def build_model(lora_r: int, lora_alpha: int, lora_dropout: float) -> nn.Module:
    """Load pretrained ViT and wrap with LoRA adapters on attention projections."""
    print("Loading pretrained ViT (google/vit-base-patch16-224)...")
    model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # Apply LoRA to query and value projections in all attention layers.
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    crops_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    mask_ratio: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    dataset = MusicCropDataset(crops_dir, processor, mask_ratio=mask_ratio)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} crops, {len(loader)} batches/epoch")

    model = build_model(lora_r, lora_alpha, lora_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # Cosine LR schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            bool_masked_pos = batch["bool_masked_pos"].to(device)

            outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Save LoRA checkpoint every epoch.
        ckpt_dir = output_dir / f"epoch_{epoch:03d}"
        model.save_pretrained(ckpt_dir)
        print(f"  Saved checkpoint → {ckpt_dir}/")

    print(f"\nTraining complete. Checkpoints in {output_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of ViT with masked patch prediction on music crops."
    )
    parser.add_argument("--crops-dir", type=Path, required=True, help="Directory of music crop images.")
    parser.add_argument("--output-dir", type=Path, default=Path("vit_lora_checkpoints"), help="Where to save LoRA checkpoints.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="Fraction of patches to mask (default: 0.75).")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling.")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout.")
    args = parser.parse_args()

    train(
        crops_dir=args.crops_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


if __name__ == "__main__":
    main()
