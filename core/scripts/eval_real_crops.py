"""Evaluate ViT checkpoints using real grayscale crops from manuscript images.

Instead of the binary RLE bitmaps stored in GameraXML, this script crops
actual pixel patches from the source page images using the bounding box
coordinates (ulx, uly, ncols, nrows) stored in each glyph.

Usage:
    python3 eval_real_crops.py                          # train/test split
    python3 eval_real_crops.py --cv                     # 5-fold cross-validation
    python3 eval_real_crops.py --checkpoint path/to/ckpt
    python3 eval_real_crops.py --k-values 1,3,5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_HERE = Path(__file__).parent
_CORE = _HERE.parent / "ic_core" / "src"
sys.path.insert(0, str(_CORE))

from ic_core.io_xml import load_glyphs
from ic_core.evaluation import evaluate, cross_validate, knn_factory, print_report
from ic_core.feature_extractor import PrecomputedExtractor, ViTExtractor
from ic_core.nn_classifier import mlp_factory

DATA = _HERE.parent / "data"
NEW_CKPTS = _HERE.parent.parent.parent / "new_checkpoints"

# (xml_path, image_path) pairs — one entry per page
PAGES = {
    "826dd1b4": (
        DATA / "derived" / "826dd1b4_train.xml",
        DATA / "train" / "hufnagel_example_826dd1b4.png",
    ),
    "a77ec16f": (
        DATA / "derived" / "a77ec16f_test.xml",
        DATA / "train" / "hufnagel_example_a77ec16f.png",
    ),
    "fbed8126": (
        DATA / "derived" / "fbed8126_test.xml",
        DATA / "train" / "hufnagel_example_fbed8126.png",
    ),
    "Antiphonal_001v": (
        Path("/tmp/ic-session-Antiphonal_001v.xml"),
        DATA / "train" / "Antiphonal_001v.jpg",
    ),
    "Antiphonal_012v": (
        Path("/tmp/ic-session-Antiphonal_012v.xml"),
        DATA / "train" / "Antiphonal_012v.jpg",
    ),
    "Antiphonal_044v": (
        Path("/tmp/ic-session-Antiphonal_044v.xml"),
        DATA / "train" / "Antiphonal_044v.jpg",
    ),
    "NZ_013r": (
        Path("/tmp/ic-session-NZ_013r.xml"),
        DATA / "test" / "NZ-Wt MSR-03 013r.png",
    ),
    "NZ_065r": (
        Path("/tmp/ic-session-NZ_065r.xml"),
        DATA / "test" / "NZ-Wt MSR-03 065r.png",
    ),
    "NZ_109v": (
        Path("/tmp/ic-session-NZ_109v.xml"),
        DATA / "test" / "NZ-Wt MSR-03 109v.png",
    ),
}

ALL_PAGES   = ["826dd1b4", "a77ec16f", "fbed8126",
               "Antiphonal_001v", "Antiphonal_012v", "Antiphonal_044v",
               "NZ_013r", "NZ_065r", "NZ_109v"]
TRAIN_PAGES = ["826dd1b4", "a77ec16f", "fbed8126",
               "Antiphonal_001v", "Antiphonal_012v", "Antiphonal_044v"]
TEST_PAGES  = ["NZ_013r", "NZ_065r", "NZ_109v"]


def crop_real(glyph, page_img: Image.Image, padding: int = 4) -> Image.Image:
    x, y, w, h = glyph.ulx, glyph.uly, glyph.ncols, glyph.nrows
    W, H = page_img.size
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(W, x + w + padding)
    y1 = min(H, y + h + padding)
    crop = page_img.crop((x0, y0, x1, y1)).convert("RGB")
    side = max(crop.width, crop.height, 16)
    return ImageOps.pad(crop, (side, side), color=(255, 255, 255))


def load_page(page_id: str) -> tuple:
    xml_path, img_path = PAGES[page_id]
    glyphs = load_glyphs(xml_path)
    img = Image.open(img_path)
    crops = [crop_real(g, img) for g in glyphs]
    return glyphs, crops


def extract_features(
    glyphs, images, extractor_raw: ViTExtractor, pooling: str = "cls"
) -> np.ndarray:
    """Extract features using one of three pooling strategies:
      cls      — CLS token only              (dim = D)
      mean     — mean over patch tokens      (dim = D)
      cls_mean — CLS concatenated with mean  (dim = 2*D)
    """
    import torch
    extractor_raw._load()
    model      = extractor_raw._model
    processor  = extractor_raw._processor
    device     = extractor_raw.device
    batch_size = extractor_raw.batch_size

    all_embeddings = []
    for start in range(0, len(glyphs), batch_size):
        batch_imgs = images[start : start + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        hidden = out.last_hidden_state  # (B, 1+N_patches, D)
        cls_tok  = hidden[:, 0, :]
        mean_tok = hidden[:, 1:, :].mean(dim=1)
        if pooling == "cls":
            vec = cls_tok
        elif pooling == "mean":
            vec = mean_tok
        elif pooling == "cls_mean":
            vec = torch.cat([cls_tok, mean_tok], dim=-1)
        else:
            raise ValueError(f"Unknown pooling '{pooling}'. Choose cls|mean|cls_mean.")
        all_embeddings.append(vec.cpu().numpy().astype(np.float64))
    return np.vstack(all_embeddings)


def run_cv(label, factory, glyphs, folds=5):
    print(f"\n── {label} ──")
    result = cross_validate(glyphs, k_folds=folds, classifier_factory=factory)
    print_report(result)


def run_train_test(label, factory, train_glyphs, test_glyphs):
    print(f"\n── {label} ──")
    result = evaluate(train_glyphs, test_glyphs, classifier_factory=factory)
    print_report(result)


def run_page_pair_cv(label, factory_fn, page_glyphs: dict):
    """Train on every combination of 2 pages, test on the remaining 7.

    page_glyphs: dict mapping page_id -> list[Glyph]
    factory_fn: callable(train_glyphs, test_glyphs) -> (train_acc, test_acc)
    """
    from itertools import combinations

    page_ids = list(page_glyphs.keys())
    accs = []
    rows = []

    for train_pages in combinations(page_ids, 2):
        test_pages = [p for p in page_ids if p not in train_pages]
        train_glyphs = [g for p in train_pages for g in page_glyphs[p]]
        test_glyphs  = [g for p in test_pages  for g in page_glyphs[p]]

        # Filter test to classes seen in training
        train_classes = set(g.class_name for g in train_glyphs)
        test_filtered = [g for g in test_glyphs if g.class_name in train_classes]

        if not test_filtered:
            continue

        result = evaluate(train_glyphs, test_filtered, classifier_factory=factory_fn)
        acc = result.accuracy
        accs.append(acc)
        rows.append((train_pages, acc, len(train_glyphs), len(test_filtered)))

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)

    print(f"\n── {label} | page-pair CV (train=2 pages, test=7 pages) ──")
    print(f"  Combinations : {len(accs)} of C(9,2)=36")
    print(f"  Mean accuracy: {mean_acc:.4f}  ±{std_acc:.4f}")
    print(f"  Min / Max    : {min(accs):.4f} / {max(accs):.4f}")
    print()
    print(f"  {'Train pages':<45}  {'Test acc':>8}  {'#train':>6}  {'#test':>6}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*6}  {'-'*6}")
    for (tp, acc, ntr, nte) in sorted(rows, key=lambda x: x[1], reverse=True):
        tp_str = f"{tp[0]} + {tp[1]}"
        print(f"  {tp_str:<45}  {acc:>8.4f}  {ntr:>6}  {nte:>6}")
    return mean_acc, std_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--pooling", choices=["cls", "mean", "cls_mean"], default="cls",
                        help="Feature pooling strategy: cls | mean | cls_mean (default: cls)")
    parser.add_argument("--cv", action="store_true", help="5-fold cross-validation on all pages")
    parser.add_argument("--page-pair-cv", action="store_true",
                        help="Train on every pair of 2 pages, test on remaining 7 (36 folds)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k-values", type=str, default="1,3,5")
    parser.add_argument("--hidden", type=str, default="128,64")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    hidden   = tuple(int(x) for x in args.hidden.split(","))

    # Load pages
    page_ids = ALL_PAGES if (args.cv or args.page_pair_cv) else TRAIN_PAGES + TEST_PAGES
    print("Loading pages and cropping real patches...")
    page_glyphs, page_crops = {}, {}
    all_glyphs, all_crops = [], []
    for pid in page_ids:
        g, c = load_page(pid)
        page_glyphs[pid] = g
        page_crops[pid]  = c
        all_glyphs += g
        all_crops  += c
        print(f"  {pid}: {len(g)} glyphs")

    if args.cv:
        print(f"\n[5-fold cross-validation]  {len(all_glyphs)} glyphs across {len(page_ids)} pages")
    elif args.page_pair_cv:
        print(f"\n[page-pair CV]  train=2 pages, test=7 pages, 36 combinations")
    else:
        n_train = sum(len(load_glyphs(PAGES[p][0])) for p in TRAIN_PAGES)
        print(f"\n[train/test]  {n_train} train / {len(all_glyphs) - n_train} test")
        train_glyphs = all_glyphs[:n_train]
        test_glyphs  = [g for g in all_glyphs[n_train:]
                        if g.class_name in set(g2.class_name for g2 in train_glyphs)]
        unseen = len(all_glyphs) - n_train - len(test_glyphs)
        if unseen:
            print(f"  ({unseen} test glyphs skipped — unseen classes)")

    # Build checkpoint list
    if args.checkpoint:
        checkpoints = [(args.checkpoint.name, args.checkpoint)]
    else:
        checkpoints = [("vanilla ViT", None)]
        for label, path in [
            ("ViT LoRA e20",      NEW_CKPTS / "vit_lora_e020"),
            ("DINO LoRA e3",      NEW_CKPTS / "dino_lora_e003"),
            ("DINO LoRA e16",     NEW_CKPTS / "dino_lora_e016"),
            ("ViT Adapter e20",   NEW_CKPTS / "vit_adapter_e020"),
            ("DINO Adapter e1",   NEW_CKPTS / "dino_adapter_e001"),
            ("DINO Adapter e14",  NEW_CKPTS / "dino_adapter_e014"),
            ("ViT SSF e20",       NEW_CKPTS / "vit_ssf_e020"),
            ("DINO SSF e1",       NEW_CKPTS / "dino_ssf_e001"),
            ("DINO SSF e12",      NEW_CKPTS / "dino_ssf_e012"),
            ("ViT SimCLR e20",   NEW_CKPTS / "vit_simclr_e020"),
            ("DINO SimCLR e5",   NEW_CKPTS / "dino_simclr_e005"),
            ("DINO SimCLR e20",  NEW_CKPTS / "dino_simclr_e020"),
            ("ViT SupCon e15",   NEW_CKPTS / "vit_supcon_e015"),
            ("DINO SupCon e15",  NEW_CKPTS / "dino_supcon_e015"),
        ]:
            if path.exists():
                checkpoints.append((label, path))

    for label, ckpt in checkpoints:
        print(f"\n{'='*50}")
        print(f">>> {label}")
        print(f"{'='*50}")
        raw_vit = ViTExtractor(checkpoint=ckpt)
        print(f"Extracting features for {len(all_crops)} real crops (pooling={args.pooling})...")
        features  = extract_features(all_glyphs, all_crops, raw_vit, pooling=args.pooling)
        print(f"Done — {features.shape[1]}-dim features.")
        extractor = PrecomputedExtractor(all_glyphs, features)

        if args.cv:
            run_cv(f"MLP {hidden} | {label}",
                   mlp_factory(hidden_sizes=hidden, epochs=args.epochs, lr=args.lr, extractor=extractor),
                   all_glyphs, folds=args.folds)
            for k in k_values:
                run_cv(f"kNN k={k} | {label}",
                       knn_factory(k=k, extractor=extractor),
                       all_glyphs, folds=args.folds)
        elif args.page_pair_cv:
            # Build per-page PrecomputedExtractors using the already-extracted features
            page_extractors = {}
            offset = 0
            for pid in page_ids:
                n = len(page_glyphs[pid])
                page_extractors[pid] = page_glyphs[pid]
                offset += n
            # Pass per-page glyph lists; factory uses precomputed extractor
            run_page_pair_cv(
                f"MLP {hidden} | {label}",
                mlp_factory(hidden_sizes=hidden, epochs=args.epochs, lr=args.lr, extractor=extractor),
                page_glyphs,
            )
            for k in k_values:
                run_page_pair_cv(
                    f"kNN k={k} | {label}",
                    knn_factory(k=k, extractor=extractor),
                    page_glyphs,
                )
        else:
            run_train_test(f"MLP {hidden} | {label}",
                           mlp_factory(hidden_sizes=hidden, epochs=args.epochs, lr=args.lr, extractor=extractor),
                           train_glyphs, test_glyphs)
            for k in k_values:
                run_train_test(f"kNN k={k} | {label}",
                               knn_factory(k=k, extractor=extractor),
                               train_glyphs, test_glyphs)


if __name__ == "__main__":
    main()
