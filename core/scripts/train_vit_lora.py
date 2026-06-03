"""LoRA fine-tuning of a pretrained ViT using masked patch prediction.

Loads any ViT backbone from HuggingFace, attaches a lightweight linear
decoder head for masked patch reconstruction, and fine-tunes only the
LoRA adapters on the attention layers. Adapts the ViT representations
toward manuscript music notation without needing labelled data.

Usage::

    # ViT-tiny (default, recommended)
    python train_vit_lora.py \\
        --crops-dir /path/to/music_crops \\
        --output-dir /path/to/checkpoints

    # ViT-base
    python train_vit_lora.py \\
        --model google/vit-base-patch16-224 \\
        --crops-dir /path/to/music_crops \\
        --output-dir /path/to/checkpoints

After training, point ViTExtractor at the saved checkpoint::

    extractor = ViTExtractor(
        model_name="WinKawaks/vit-tiny-patch16-224",
        lora_checkpoint="/path/to/checkpoints/epoch_020",
    )

Dependencies::

    pip install torch torchvision transformers peft
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, ViTConfig, ViTModel

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise ImportError("Install peft: pip install peft")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ViT patch16/224 divides the image into 14×14 = 196 patches.
NUM_PATCHES = 196
PATCH_SIZE = 16
CHANNELS = 3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MusicCropDataset(Dataset):
    """Loads music crop images, pads to square, resizes to 224×224."""

    def __init__(self, crops_dir: Path, processor: AutoImageProcessor, mask_ratio: float = 0.75):
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
        w, h = image.size
        side = max(w, h)
        padded = ImageOps.pad(image, (side, side), color=(255, 255, 255))

        inputs = self.processor(images=padded, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (3, 224, 224)

        num_masked = int(self.mask_ratio * NUM_PATCHES)
        mask = torch.zeros(NUM_PATCHES, dtype=torch.bool)
        mask[random.sample(range(NUM_PATCHES), num_masked)] = True

        return {"pixel_values": pixel_values, "bool_masked_pos": mask}


# ---------------------------------------------------------------------------
# Model — backbone + reconstruction decoder
# ---------------------------------------------------------------------------


class MaskedViT(nn.Module):
    """ViT backbone with LoRA adapters + a linear masked-patch decoder.

    Works with any ViT size (tiny, small, base) since the decoder head
    is built from the model's hidden_size rather than relying on a
    pretrained MIM-specific architecture.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        # Reconstruct raw pixel values for each masked patch:
        # patch_size × patch_size × channels values per patch token.
        self.decoder = nn.Linear(hidden_size, PATCH_SIZE * PATCH_SIZE * CHANNELS)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        # patch tokens: skip the CLS token (index 0)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (B, 196, hidden)

        # Reconstruct only the masked patches
        masked_tokens = patch_tokens[bool_masked_pos]       # (n_masked_total, hidden)
        reconstructed = self.decoder(masked_tokens)         # (n_masked_total, 768)

        # Build reconstruction targets from pixel values
        B = pixel_values.shape[0]
        # Patchify: (B, C, H, W) → (B, num_patches, patch_pixels)
        p = PATCH_SIZE
        x = pixel_values.reshape(B, CHANNELS, 14, p, 14, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, NUM_PATCHES, CHANNELS * p * p)
        targets = x[bool_masked_pos]  # (n_masked_total, patch_pixels)

        loss = nn.functional.mse_loss(reconstructed, targets)
        return loss


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def build_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float) -> MaskedViT:
    """Load pretrained ViT backbone, apply LoRA, wrap with decoder."""
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

    model = MaskedViT(backbone, hidden_size=config.hidden_size)
    return model


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
    mask_ratio: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    dataset = MusicCropDataset(crops_dir, processor, mask_ratio=mask_ratio)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} crops, {len(loader)} batches/epoch")

    model = build_model(model_name, lora_r, lora_alpha, lora_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            bool_masked_pos = batch["bool_masked_pos"].to(device)

            loss = model(pixel_values, bool_masked_pos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Save LoRA adapter weights only (small — not the full backbone)
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
        description="LoRA fine-tuning of ViT with masked patch prediction on music crops."
    )
    parser.add_argument("--model", type=str, default="WinKawaks/vit-tiny-patch16-224",
                        help="HuggingFace model name (default: WinKawaks/vit-tiny-patch16-224)")
    parser.add_argument("--crops-dir", type=Path, required=True,
                        help="Directory of music crop images.")
    parser.add_argument("--output-dir", type=Path, default=Path("vit_lora_checkpoints"),
                        help="Where to save LoRA checkpoints.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default 64, larger than base since tiny is smaller).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    args = parser.parse_args()

    train(
        model_name=args.model,
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
