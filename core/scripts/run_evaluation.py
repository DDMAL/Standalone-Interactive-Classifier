"""Evaluate classifiers on GameraXML data.

When --extractor vit is used, ViT features are extracted once for all glyphs
and reused across every model and fold via PrecomputedExtractor.

Examples:
  python3 scripts/run_evaluation.py --k-values 1,3,5
  python3 scripts/run_evaluation.py --extractor vit --k-values 1,3,5
  python3 scripts/run_evaluation.py --train training.xml --test groundtruth.xml

Options:
  --extractor STR    "handcrafted" or "vit" (default "handcrafted")
  --folds N          CV folds (default 5)
  --k-values STR     Comma-separated kNN k values (default "1")
  --hidden STR       MLP hidden layer sizes (default "128,64")
  --epochs N         MLP training epochs (default 100)
  --lr F             MLP Adam learning rate (default 0.001)
  --xml PATH         Single XML for cross-validation
  --train PATH       Training XML for train/test mode
  --test PATH        Ground-truth test XML for train/test mode
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_CORE = _HERE.parent / "ic_core" / "src"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from ic_core.evaluation import cross_validate, evaluate, knn_factory, print_report
from ic_core.feature_extractor import HandcraftedExtractor, PrecomputedExtractor, ViTExtractor
from ic_core.nn_classifier import mlp_factory
from ic_core.io_xml import load_glyphs

_DEFAULT_FIXTURE = (
    _HERE.parent / "tests" / "fixtures"
    / "Interactive_Classifier_GameraXML_TrainingData.xml"
)


def _run(label, factory, glyphs=None, train_glyphs=None, test_glyphs=None, folds=5):
    print(f"\n── {label} ──")
    if train_glyphs is not None:
        result = evaluate(train_glyphs, test_glyphs, classifier_factory=factory)
    else:
        result = cross_validate(glyphs, k_folds=folds, classifier_factory=factory)
    print_report(result)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extractor", choices=["handcrafted", "vit"], default="handcrafted")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k-values", type=str, default="1")
    parser.add_argument("--hidden", type=str, default="128,64")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    hidden = tuple(int(x) for x in args.hidden.split(","))
    train_test_mode = args.train is not None or args.test is not None

    if train_test_mode:
        if args.train is None or args.test is None:
            sys.exit("Both --train and --test are required for train/test mode.")
        for p in (args.train, args.test):
            if not p.exists():
                sys.exit(f"File not found: {p}")
        train_glyphs = load_glyphs(args.train)
        test_glyphs  = load_glyphs(args.test)
        all_glyphs   = train_glyphs + test_glyphs
        glyphs = None
        print(f"[train/test]  {len(train_glyphs)} train / {len(test_glyphs)} test")
    else:
        xml_path = args.xml if args.xml is not None else _DEFAULT_FIXTURE
        if not xml_path.exists():
            sys.exit(f"File not found: {xml_path}")
        glyphs     = load_glyphs(xml_path)
        all_glyphs = glyphs
        train_glyphs = test_glyphs = None
        print(f"[cross-validation]  {args.folds} folds  —  {xml_path.name}")

    # Build extractor — precompute ViT features once for all glyphs
    if args.extractor == "vit":
        print(f"\nExtracting ViT features for {len(all_glyphs)} glyphs (runs once)...")
        raw_vit = ViTExtractor()
        features = raw_vit.extract_batch(all_glyphs)
        extractor = PrecomputedExtractor(all_glyphs, features)
        print(f"Done — {features.shape[1]}-dim features cached for all models.\n")
    else:
        extractor = HandcraftedExtractor()

    # MLP
    _run(f"MLP {hidden} epochs={args.epochs} | {extractor}",
         mlp_factory(hidden_sizes=hidden, epochs=args.epochs, lr=args.lr, extractor=extractor),
         glyphs=glyphs, train_glyphs=train_glyphs, test_glyphs=test_glyphs, folds=args.folds)

    # kNN sweep
    for k in k_values:
        _run(f"kNN k={k} | {extractor}",
             knn_factory(k=k, extractor=extractor),
             glyphs=glyphs, train_glyphs=train_glyphs, test_glyphs=test_glyphs, folds=args.folds)


if __name__ == "__main__":
    main()
