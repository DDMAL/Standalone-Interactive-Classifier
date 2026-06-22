"""Incremental page evaluation.

Evaluates how much each additional labeled page improves classification.

Setup:
  - Baseline pages: always included in training (e.g. the 3 original hufnagel pages)
  - Eval pages: rotated as train additions and test pages

For each N in --n-train-values:
  For every combination of N pages chosen from eval-pages:
    train = baseline_pages + those N eval pages
    test  = remaining eval pages not in the N

  Reports mean/std/min/max accuracy across all C(|eval|, N) combinations.

Special case N=0:
  train = baseline_pages only
  test  = all eval pages (single fold, no combinations)

Usage::

    # Default: baseline=3 hufnagel pages, eval=6 new pages, test N=0,1,2,3
    python eval_incremental.py --pooling cls_mean

    # Only test N=0 and N=2 (reproduce page-pair CV with a baseline)
    python eval_incremental.py --n-train-values 0,2 --pooling cls_mean

    # No baseline — pure rotation over all 9 pages
    python eval_incremental.py --baseline-pages none --eval-pages all --n-train-values 2,3

    # Specific checkpoint
    python eval_incremental.py --checkpoint /path/to/ckpt --pooling cls_mean
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_CORE = _HERE.parent / "ic_core" / "src"
sys.path.insert(0, str(_CORE))

from ic_core.evaluation import evaluate, knn_factory
from ic_core.feature_extractor import HandcraftedExtractor, PrecomputedExtractor, ViTExtractor
from ic_core.nn_classifier import mlp_factory
from eval_real_crops import ALL_PAGES, extract_features, load_page

NEW_CKPTS = _HERE.parent.parent.parent / "new_checkpoints"

DEFAULT_BASELINE = ["826dd1b4", "a77ec16f", "fbed8126"]
DEFAULT_EVAL     = ["Antiphonal_001v", "Antiphonal_012v", "Antiphonal_044v",
                    "NZ_013r", "NZ_065r", "NZ_109v"]


def audit_labels(page_glyphs: dict, baseline_ids: list, eval_ids: list) -> None:
    """Print a cross-page label frequency report to catch annotation inconsistencies."""
    from collections import Counter

    all_ids = baseline_ids + [p for p in eval_ids if p not in baseline_ids]
    all_classes = sorted({g.class_name for glyphs in page_glyphs.values() for g in glyphs})

    # Per-page counts
    counts = {pid: Counter(g.class_name for g in page_glyphs[pid]) for pid in all_ids}
    total  = Counter(g.class_name for glyphs in page_glyphs.values() for g in glyphs)

    print("\n" + "="*70)
    print("LABEL AUDIT")
    print("="*70)

    # Summary per page
    print(f"\n{'Page':<25}  {'#glyphs':>7}  {'#classes':>8}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*8}")
    for pid in all_ids:
        tag = "[B]" if pid in baseline_ids else "[E]"
        print(f"  {tag} {pid:<22}  {sum(counts[pid].values()):>7}  {len(counts[pid]):>8}")

    # Classes present across pages
    print(f"\n{'Class':<35}  {'total':>5}  " + "  ".join(f"{p[:8]:>8}" for p in all_ids))
    print(f"  {'-'*35}  {'-'*5}  " + "  ".join("-"*8 for _ in all_ids))
    for cls in sorted(all_classes, key=lambda c: -total[c]):
        row = f"  {cls:<35}  {total[cls]:>5}  "
        row += "  ".join(f"{counts[pid].get(cls, 0):>8}" for pid in all_ids)
        print(row)

    # Classes unique to a single page (likely annotation artifacts)
    singletons = [cls for cls in all_classes
                  if sum(1 for pid in all_ids if counts[pid].get(cls, 0) > 0) == 1]
    if singletons:
        print(f"\n  !! {len(singletons)} class(es) appear in only ONE page (possible typos/artifacts):")
        for cls in sorted(singletons):
            owner = next(pid for pid in all_ids if counts[pid].get(cls, 0) > 0)
            print(f"     '{cls}'  (n={total[cls]}, page={owner})")

    # Very rare classes (< 3 total) — may hurt kNN
    rare = [cls for cls in all_classes if total[cls] < 3]
    if rare:
        print(f"\n  !! {len(rare)} class(es) have fewer than 3 total samples:")
        for cls in sorted(rare):
            print(f"     '{cls}'  (n={total[cls]})")

    print("="*70 + "\n")


def run_incremental(
    label: str,
    factory_fn,
    baseline_glyphs: list,
    page_glyphs: dict,
    eval_page_ids: list,
    n_train_values: list[int],
):
    """Run incremental evaluation for all requested N values."""
    print(f"\n{'='*60}")
    print(f">>> {label}")
    print(f"{'='*60}")
    print(f"  Baseline : {len(baseline_glyphs)} glyphs")
    print(f"  Eval pool: {len(eval_page_ids)} pages — {eval_page_ids}")

    for n in n_train_values:
        if n == 0:
            # Single fold: train on baseline only, test on all eval pages
            test_glyphs = [g for pid in eval_page_ids for g in page_glyphs[pid]]
            if not baseline_glyphs:
                print(f"\n  N=0  [SKIP — baseline is empty, no training data]")
                continue
            train_cls     = set(g.class_name for g in baseline_glyphs)
            test_filtered = [g for g in test_glyphs if g.class_name in train_cls]
            skipped = len(test_glyphs) - len(test_filtered)
            result  = evaluate(baseline_glyphs, test_filtered, classifier_factory=factory_fn)
            print(f"\n  N=0  (baseline only → all {len(eval_page_ids)} eval pages)")
            print(f"    train={len(baseline_glyphs)}  test={len(test_filtered)}"
                  f"  skipped={skipped}  acc={result.accuracy:.4f}")
        else:
            n_combos = len(list(combinations(eval_page_ids, n)))
            accs = []
            skips = 0
            for train_pages in combinations(eval_page_ids, n):
                test_pages    = [p for p in eval_page_ids if p not in train_pages]
                extra_glyphs  = [g for p in train_pages for g in page_glyphs[p]]
                train_glyphs  = baseline_glyphs + extra_glyphs
                test_glyphs   = [g for p in test_pages for g in page_glyphs[p]]

                if not train_glyphs or not test_glyphs:
                    skips += 1
                    continue

                train_cls     = set(g.class_name for g in train_glyphs)
                test_filtered = [g for g in test_glyphs if g.class_name in train_cls]
                if not test_filtered:
                    skips += 1
                    continue

                result = evaluate(train_glyphs, test_filtered, classifier_factory=factory_fn)
                accs.append(result.accuracy)

            if not accs:
                print(f"\n  N={n}  [SKIP — no valid folds]")
                continue

            mean, std = np.mean(accs), np.std(accs)
            print(f"\n  N={n}  ({len(accs)}/{n_combos} folds"
                  + (f", {skips} skipped" if skips else "") + ")")
            print(f"    mean={mean:.4f}  std={std:.4f}  "
                  f"min={min(accs):.4f}  max={max(accs):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Incremental page evaluation — train on baseline + N eval pages, test on rest."
    )
    parser.add_argument("--baseline-pages", type=str,
                        default=",".join(DEFAULT_BASELINE),
                        help="Comma-separated page IDs always in training, or 'none' for empty baseline.")
    parser.add_argument("--eval-pages", type=str,
                        default=",".join(DEFAULT_EVAL),
                        help="Comma-separated page IDs to rotate as train additions and test pages. "
                             "Use 'all' to include every page.")
    parser.add_argument("--n-train-values", type=str, default="0,1,2,3",
                        help="Comma-separated N values: how many eval pages to add to training per fold.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to a ViT checkpoint directory. Omit for all default checkpoints.")
    parser.add_argument("--pooling", choices=["cls", "mean", "cls_mean"], default="cls_mean")
    parser.add_argument("--k-values", type=str, default="1,3,5")
    parser.add_argument("--hidden", type=str, default="128,64")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--handcrafted", action="store_true",
                        help="Also evaluate handcrafted features.")
    parser.add_argument("--combined", action="store_true",
                        help="Also evaluate DINO SimCLR + handcrafted concatenated features.")
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    hidden   = tuple(int(x) for x in args.hidden.split(","))
    n_train_values = [int(x) for x in args.n_train_values.split(",")]

    baseline_ids = [] if args.baseline_pages.lower() == "none" \
                   else [p.strip() for p in args.baseline_pages.split(",")]
    eval_ids = ALL_PAGES if args.eval_pages.lower() == "all" \
               else [p.strip() for p in args.eval_pages.split(",")]

    all_ids = list(dict.fromkeys(baseline_ids + eval_ids))  # preserve order, no duplicates

    # Load all pages
    print("Loading pages...")
    page_glyphs, page_crops = {}, {}
    for pid in all_ids:
        g, c = load_page(pid)
        page_glyphs[pid] = g
        page_crops[pid]  = c
        tag = "[baseline]" if pid in baseline_ids else "[eval]   "
        print(f"  {tag} {pid}: {len(g)} glyphs")

    # Filter out UNCLASSIFIED glyphs — they are annotation noise, not a real class
    filter_classes = {"UNCLASSIFIED"}
    for pid in all_ids:
        before = len(page_glyphs[pid])
        pairs = [(g, c) for g, c in zip(page_glyphs[pid], page_crops[pid])
                 if g.class_name not in filter_classes]
        page_glyphs[pid] = [g for g, c in pairs]
        page_crops[pid]  = [c for g, c in pairs]
        removed = before - len(page_glyphs[pid])
        if removed:
            print(f"  filtered {pid}: removed {removed} UNCLASSIFIED glyphs → {len(page_glyphs[pid])} remain")

    baseline_glyphs = [g for pid in baseline_ids for g in page_glyphs[pid]]
    all_glyphs = [g for pid in all_ids for g in page_glyphs[pid]]
    all_crops  = [c for pid in all_ids for c in page_crops[pid]]

    print(f"\nBaseline: {len(baseline_glyphs)} glyphs from {len(baseline_ids)} pages")
    print(f"Eval pool: {sum(len(page_glyphs[p]) for p in eval_ids)} glyphs from {len(eval_ids)} pages")
    print(f"N values to test: {n_train_values}")

    audit_labels(page_glyphs, baseline_ids, eval_ids)

    # Build checkpoint list
    if args.checkpoint:
        checkpoints = [(args.checkpoint.name, args.checkpoint)]
    else:
        checkpoints = [("vanilla ViT", None)]
        for lbl, path in [
            ("ViT SimCLR e20",   NEW_CKPTS / "vit_simclr_e020"),
            ("DINO SimCLR e5",   NEW_CKPTS / "dino_simclr_e005"),
            ("DINO SimCLR e20",  NEW_CKPTS / "dino_simclr_e020"),
            ("ViT SupCon e15",   NEW_CKPTS / "vit_supcon_e015"),
            ("DINO SupCon e15",  NEW_CKPTS / "dino_supcon_e015"),
            ("ViT LoRA e20",     NEW_CKPTS / "vit_lora_e020"),
            ("DINO LoRA e16",    NEW_CKPTS / "dino_lora_e016"),
        ]:
            if path.exists():
                checkpoints.append((lbl, path))

    mlp_kw = dict(hidden_sizes=hidden, epochs=args.epochs, lr=args.lr)

    # Handcrafted
    if args.handcrafted or args.combined:
        print("\nExtracting handcrafted features...")
        hc_ext  = HandcraftedExtractor()
        feats_hc = hc_ext.extract_batch(all_glyphs)
        ext_hc   = PrecomputedExtractor(all_glyphs, feats_hc)

    if args.handcrafted:
        for k in k_values:
            run_incremental(f"Handcrafted kNN k={k}", knn_factory(k=k, extractor=ext_hc),
                            baseline_glyphs, page_glyphs, eval_ids, n_train_values)
        run_incremental(f"Handcrafted MLP {hidden}", mlp_factory(**mlp_kw, extractor=ext_hc),
                        baseline_glyphs, page_glyphs, eval_ids, n_train_values)

    # ViT checkpoints
    for ckpt_label, ckpt in checkpoints:
        print(f"\nExtracting features: {ckpt_label} (pooling={args.pooling})...")
        vit_ext  = ViTExtractor(checkpoint=ckpt)
        feats_vit = extract_features(all_glyphs, all_crops, vit_ext, pooling=args.pooling)
        print(f"  {feats_vit.shape[1]}-dim features for {len(all_glyphs)} glyphs")

        ext_vit = PrecomputedExtractor(all_glyphs, feats_vit)

        for k in k_values:
            run_incremental(f"{ckpt_label} kNN k={k}", knn_factory(k=k, extractor=ext_vit),
                            baseline_glyphs, page_glyphs, eval_ids, n_train_values)
        run_incremental(f"{ckpt_label} MLP {hidden}", mlp_factory(**mlp_kw, extractor=ext_vit),
                        baseline_glyphs, page_glyphs, eval_ids, n_train_values)

        if args.combined:
            feats_combined = np.concatenate([feats_vit, feats_hc], axis=1)
            ext_combined   = PrecomputedExtractor(all_glyphs, feats_combined)
            for k in k_values:
                run_incremental(f"{ckpt_label}+HC kNN k={k}", knn_factory(k=k, extractor=ext_combined),
                                baseline_glyphs, page_glyphs, eval_ids, n_train_values)
            run_incremental(f"{ckpt_label}+HC MLP {hidden}", mlp_factory(**mlp_kw, extractor=ext_combined),
                            baseline_glyphs, page_glyphs, eval_ids, n_train_values)


if __name__ == "__main__":
    main()
