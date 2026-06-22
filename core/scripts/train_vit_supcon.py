"""SupCon with pseudo-labels from a prototype bank built on labeled eval pages.

Motivation
----------
SimCLR treats any two augmented crops of the SAME IMAGE as positives. But two
crops of different neumes that look similar will be pushed apart — hurting the
embedding.

Pseudo-label SupCon fixes this: we build a prototype bank (per-class centroid)
from the small labeled eval set, assign pseudo-labels to every unlabeled SSL
crop, then run Supervised Contrastive loss so ALL crops sharing a pseudo-label
pull together (not just the two-view pair).

Training flow
-------------
1. Load labeled eval pages → extract CLS features → compute per-class centroids
   (prototype bank, shape C × D).
2. For each SSL batch: assign pseudo_label[i] = argmax(cos_sim(z_i, prototypes)).
3. SupCon loss: for each anchor, pull together all batch samples sharing its
   pseudo-label, push away all others.
4. Refresh prototype bank every --refresh-interval epochs so centroids track the
   evolving representation.

Warm-starting from a SimCLR checkpoint (e.g. dino_simclr_e020) is strongly
recommended — SupCon with random features produces noisy pseudo-labels.

Usage::

    # DINO (recommended — warm-start from SimCLR e5, 15 epochs)
    python train_vit_supcon.py \\
        --model pretrained_models/dino-vits16 \\
        --crops-dir /path/to/ssl_crops \\
        --output-dir /path/to/dino_supcon_checkpoints \\
        --simclr-checkpoint /path/to/dino_simclr_checkpoints/epoch_020 \\
        --epochs 15 --batch-size 64 --num-workers 16 --n-crops 2679730

    # ViT-tiny
    python train_vit_supcon.py \\
        --model pretrained_models/vit-tiny-patch16-224 \\
        --crops-dir /path/to/ssl_crops \\
        --output-dir /path/to/vit_supcon_checkpoints \\
        --simclr-checkpoint /path/to/vit_simclr_checkpoints/epoch_030 \\
        --epochs 15 --batch-size 128 --num-workers 16 --n-crops 2679730

After training, load exactly like any other LoRA checkpoint::

    extractor = ViTExtractor(checkpoint="path/to/checkpoints/epoch_010")
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, ViTConfig, ViTModel

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:
    raise ImportError("Install peft: pip install peft")

try:
    import webdataset as wds
except ImportError:
    raise ImportError("Install webdataset: pip install webdataset")

# Reuse data-loading helpers from the eval script
_HERE = Path(__file__).parent
_CORE = _HERE.parent / "ic_core" / "src"
sys.path.insert(0, str(_CORE))
sys.path.insert(0, str(_HERE))

from eval_real_crops import ALL_PAGES, PAGES, load_page


# ---------------------------------------------------------------------------
# Augmentation — same as SimCLR (single view per sample)
# ---------------------------------------------------------------------------

def _build_augment(image_size: int = 224) -> transforms.Compose:
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
# Prototype bank
# ---------------------------------------------------------------------------

class PrototypeBank:
    """Per-class feature centroids built from labeled eval pages.

    Prototypes are unit-norm vectors in CLS-token space (pre-projection).
    Refresh after every few epochs to track the evolving backbone.
    """

    def __init__(self, page_ids: list[str]) -> None:
        self.page_ids = [p for p in page_ids if PAGES.get(p) and PAGES[p][0].exists()]
        self.prototypes: Optional[torch.Tensor] = None  # (C, D)
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}

    def build(self, model: nn.Module, processor, device: torch.device) -> None:
        """Extract CLS features from labeled pages, compute per-class centroids."""
        all_glyphs, all_crops = [], []
        for pid in self.page_ids:
            glyphs, crops = load_page(pid)
            for g, c in zip(glyphs, crops):
                if g.class_name != "UNCLASSIFIED":
                    all_glyphs.append(g)
                    all_crops.append(c)

        if not all_glyphs:
            raise ValueError(f"No labeled glyphs found for pages: {self.page_ids}")

        classes = sorted({g.class_name for g in all_glyphs})
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # Extract features in batches
        model.eval()
        batch_size = 64
        all_feats = []
        with torch.no_grad():
            for start in range(0, len(all_crops), batch_size):
                batch = all_crops[start : start + batch_size]
                inputs = processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = model.backbone(pixel_values=inputs["pixel_values"])
                cls = out.last_hidden_state[:, 0, :]
                all_feats.append(F.normalize(cls, dim=1).cpu())
        model.train()

        features = torch.cat(all_feats, dim=0)  # (N, D)
        labels = torch.tensor([self.class_to_idx[g.class_name] for g in all_glyphs])

        C, D = len(classes), features.shape[1]
        centroids = torch.zeros(C, D)
        counts = torch.zeros(C)
        for feat, lbl in zip(features, labels):
            centroids[lbl] += feat
            counts[lbl] += 1

        counts = counts.clamp(min=1)
        self.prototypes = F.normalize(centroids / counts.unsqueeze(1), dim=1)

        print(f"  Prototype bank: {C} classes from {len(all_glyphs)} labeled glyphs")
        for i, cls in enumerate(classes):
            print(f"    {cls:<35}  n={int(counts[i])}")

    def assign(self, cls_features: torch.Tensor) -> torch.Tensor:
        """Return pseudo-label indices for a batch of L2-normalized CLS tokens."""
        assert self.prototypes is not None, "Call build() first"
        sim = torch.mm(cls_features, self.prototypes.to(cls_features.device))
        return sim.argmax(dim=1)

    def confidence(self, cls_features: torch.Tensor) -> torch.Tensor:
        """Return top-1 cosine similarity score (proxy for pseudo-label confidence)."""
        assert self.prototypes is not None
        sim = torch.mm(cls_features, self.prototypes.to(cls_features.device))
        return sim.max(dim=1).values


# ---------------------------------------------------------------------------
# Supervised Contrastive loss
# ---------------------------------------------------------------------------

def supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al. 2020).

    features : (N, D)  L2-normalised projection vectors
    labels   : (N,)    integer pseudo-labels
    Anchors with no positive partner in the batch are excluded from the loss.
    """
    N = features.shape[0]
    device = features.device

    sim = torch.mm(features, features.T) / temperature  # (N, N)

    # Numerical stability: shift by row maximum
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    eye = torch.eye(N, device=device)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float() * (1 - eye)  # same class, not self

    exp_sim = torch.exp(sim) * (1 - eye)           # zero out diagonal
    denom = exp_sim.sum(dim=1, keepdim=True)        # (N, 1)

    log_prob = sim - torch.log(denom + 1e-8)        # (N, N)

    n_pos = pos_mask.sum(dim=1)                     # (N,)
    valid = n_pos > 0

    if not valid.any():
        # No positive pairs — batch too small or all same class
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / n_pos.clamp(min=1)
    return loss_per_anchor[valid].mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SupConViT(nn.Module):
    """LoRA-adapted ViT with a projection head for SupCon loss."""

    def __init__(self, backbone: nn.Module, hidden_size: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (proj_z, cls_z) — both L2-normalised."""
        cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        z = F.normalize(self.projector(cls), dim=1)
        cls_norm = F.normalize(cls, dim=1)
        return z, cls_norm


def build_model(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    proj_dim: int,
    simclr_checkpoint: Optional[Path] = None,
) -> SupConViT:
    print(f"Loading {model_name}...")
    config = ViTConfig.from_pretrained(model_name)

    if simclr_checkpoint is not None:
        print(f"Warm-starting LoRA from SimCLR checkpoint {simclr_checkpoint} ...")
        base = ViTModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        backbone = PeftModel.from_pretrained(base, str(simclr_checkpoint), is_trainable=True)
    else:
        print("No SimCLR checkpoint — initialising fresh LoRA adapters.")
        backbone = ViTModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"], bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)

    backbone.print_trainable_parameters()
    return SupConViT(backbone, hidden_size=config.hidden_size, proj_dim=proj_dim)


# ---------------------------------------------------------------------------
# DataLoader (single augmented view per crop)
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
    n_crops: Optional[int],
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
        padded = ImageOps.pad(image, (side, side), color=(255, 255, 255))
        return {"view": augment(padded)}

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
# Training loop
# ---------------------------------------------------------------------------

def train(
    model_name: str,
    crops_dir: Path,
    labeled_page_ids: list[str],
    output_dir: Path,
    epochs: int,
    start_epoch: int,
    batch_size: int,
    lr: float,
    temperature: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    proj_dim: int,
    num_workers: int,
    n_crops: Optional[int],
    simclr_checkpoint: Optional[Path],
    refresh_interval: int,
    min_confidence: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader, steps_per_epoch = build_dataloader(crops_dir, batch_size, num_workers, n_crops)
    print(f"~{steps_per_epoch} batches/epoch")

    model = build_model(model_name, lora_r, lora_alpha, lora_dropout, proj_dim,
                        simclr_checkpoint=simclr_checkpoint).to(device)
    processor = AutoImageProcessor.from_pretrained(model_name)

    print("\nBuilding initial prototype bank...")
    bank = PrototypeBank(labeled_page_ids)
    bank.build(model, processor, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in range(start_epoch - 1):
        scheduler.step()

    output_dir.mkdir(parents=True, exist_ok=True)
    total_epochs = start_epoch + epochs - 1
    print(f"\nTraining epochs {start_epoch} → {total_epochs}")

    for epoch in range(start_epoch, total_epochs + 1):
        if epoch > start_epoch and (epoch - start_epoch) % refresh_interval == 0:
            print(f"\nRefreshing prototype bank (epoch {epoch})...")
            bank.build(model, processor, device)

        model.train()
        total_loss = 0.0
        n_skipped = 0

        for step, batch in enumerate(loader, 1):
            view = batch["view"].to(device)  # (B, 3, 224, 224)

            z, cls_norm = model(view)  # both (B, D) normalised

            # Assign pseudo-labels; optionally filter low-confidence samples
            with torch.no_grad():
                pseudo_labels = bank.assign(cls_norm)
                if min_confidence > 0:
                    conf = bank.confidence(cls_norm)
                    keep = conf >= min_confidence
                    if keep.sum() < 2:
                        n_skipped += 1
                        continue
                    z = z[keep]
                    pseudo_labels = pseudo_labels[keep]

            loss = supcon_loss(z, pseudo_labels, temperature=temperature)

            if loss.item() == 0.0:
                n_skipped += 1

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
        skip_msg = f"  ({n_skipped} steps skipped)" if n_skipped else ""
        print(f"Epoch {epoch}/{total_epochs}  loss={avg_loss:.4f}"
              f"  lr={scheduler.get_last_lr()[0]:.2e}{skip_msg}")

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
        description="Pseudo-label SupCon fine-tuning of ViT with LoRA."
    )
    parser.add_argument("--model", type=str, default="WinKawaks/vit-tiny-patch16-224")
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("vit_supcon_checkpoints"))
    parser.add_argument("--simclr-checkpoint", type=Path, default=None,
                        help="LoRA checkpoint from SimCLR training to warm-start from.")
    parser.add_argument("--labeled-pages", type=str, default=",".join(ALL_PAGES),
                        help="Comma-separated page IDs for prototype bank (default: all eval pages).")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Lower than SimCLR LR because we warm-start.")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-crops", type=int, default=None)
    parser.add_argument("--refresh-interval", type=int, default=5,
                        help="Re-build prototype bank every N epochs.")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Skip crops whose nearest-prototype cosine similarity is below "
                             "this threshold (0 = keep all). Helps filter ambiguous glyphs.")
    args = parser.parse_args()

    labeled_page_ids = [p.strip() for p in args.labeled_pages.split(",")]

    train(
        model_name=args.model,
        crops_dir=args.crops_dir,
        labeled_page_ids=labeled_page_ids,
        output_dir=args.output_dir,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        proj_dim=args.proj_dim,
        num_workers=args.num_workers,
        n_crops=args.n_crops,
        simclr_checkpoint=args.simclr_checkpoint,
        refresh_interval=args.refresh_interval,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
