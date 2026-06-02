"""Evaluate the classifier in one of two modes:

  cross-validation (default)
    Stratified k-fold on a single XML file. Training and test both come
    from the same file — no separate ground truth is needed.

    python3 scripts/run_evaluation.py --xml training.xml

  train/test (explicit ground truth)
    Train on one XML file, score against a separate ground-truth XML file.

    python3 scripts/run_evaluation.py --train training.xml --test groundtruth.xml

Options:
  --folds N          Number of CV folds (default 5, cross-validation mode only)
  --classifier STR   "knn" or "mlp" (default "knn")
  --extractor STR    "handcrafted" or "vit" (default "handcrafted")
  --k N              kNN neighbour count (default 1, knn only)
  --hidden STR       MLP hidden layer sizes as comma-separated ints (default "128,64")
  --epochs N         MLP training epochs (default 100)
  --lr F             MLP Adam learning rate (default 0.001)
  --xml PATH         Single dataset for cross-validation
  --train PATH       Training set for train/test mode
  --test PATH        Ground-truth test set for train/test mode
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
from ic_core.feature_extractor import HandcraftedExtractor, ViTExtractor
from ic_core.nn_classifier import mlp_factory
from ic_core.io_xml import load_glyphs

_DEFAULT_FIXTURE = (
    _HERE.parent / "tests" / "fixtures"
    / "Interactive_Classifier_GameraXML_TrainingData.xml"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a classifier on GameraXML data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds (cross-validation mode, default 5)")
    parser.add_argument("--classifier", choices=["knn", "mlp"], default="knn",
                        help="Classifier to use (default: knn)")
    parser.add_argument("--extractor", choices=["handcrafted", "vit"], default="handcrafted",
                        help="Feature extractor (default: handcrafted)")
    parser.add_argument("--k", type=int, default=1,
                        help="kNN neighbour count (default 1)")
    parser.add_argument("--hidden", type=str, default="128,64",
                        help="MLP hidden layer sizes, comma-separated (default: 128,64)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="MLP training epochs (default 100)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="MLP Adam learning rate (default 0.001)")
    parser.add_argument("--xml", type=Path, default=None,
                        help="Single XML file for cross-validation mode")
    parser.add_argument("--train", type=Path, default=None,
                        help="Training XML for train/test mode")
    parser.add_argument("--test", type=Path, default=None,
                        help="Ground-truth test XML for train/test mode")
    args = parser.parse_args()

    extractor = ViTExtractor() if args.extractor == "vit" else HandcraftedExtractor()

    if args.classifier == "mlp":
        hidden = tuple(int(x) for x in args.hidden.split(","))
        factory = mlp_factory(hidden_sizes=hidden, epochs=args.epochs, lr=args.lr,
                              extractor=extractor)
    else:
        factory = knn_factory(k=args.k, extractor=extractor)

    # --- train/test mode ---
    if args.train is not None or args.test is not None:
        if args.train is None or args.test is None:
            sys.exit("Both --train and --test must be provided for train/test mode.")
        for p in (args.train, args.test):
            if not p.exists():
                sys.exit(f"File not found: {p}")

        print(f"[train/test mode]")
        print(f"  Training XML : {args.train}")
        print(f"  Test XML     : {args.test}  (ground truth)\n")

        train_glyphs = load_glyphs(args.train)
        test_glyphs = load_glyphs(args.test)
        print(f"Loaded {len(train_glyphs)} training glyphs, "
              f"{len(test_glyphs)} test glyphs.\n")

        result = evaluate(train_glyphs, test_glyphs, classifier_factory=factory)

    # --- cross-validation mode ---
    else:
        xml_path = args.xml if args.xml is not None else _DEFAULT_FIXTURE
        if not xml_path.exists():
            sys.exit(f"File not found: {xml_path}")

        print(f"[cross-validation mode]")
        print(f"  XML file : {xml_path}")
        print(f"  Folds    : {args.folds}\n")
        print("  NOTE: training and test data both come from the same file.")
        print("        Use --train / --test to evaluate against a separate ground truth.\n")

        glyphs = load_glyphs(xml_path)
        print(f"Loaded {len(glyphs)} glyphs.\n")

        result = cross_validate(glyphs, k_folds=args.folds, classifier_factory=factory)

    print_report(result)


if __name__ == "__main__":
    main()
