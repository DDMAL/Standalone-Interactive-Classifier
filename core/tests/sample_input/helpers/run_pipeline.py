"""Single-shot train → classify → visualise pipeline.

Glue script that wires the existing helpers together so a user can
pick a training set and a test set, run the classifier once, and
get two overlay PNGs out the other end. No validation: this script
does not measure accuracy, run cross-validation, or hold out a
fold — those belong in :mod:`tests.test_real_input_knn`.

The four pieces of state the caller can vary:

* ``--train-xml`` — GameraXML database to train on.
* ``--test-page`` — page image to classify.
* ``--test-json`` — MOTHRA-shaped JSON describing the bboxes on the
  test page. The same JSON drives both ingest (cropping) and the
  visualisation overlays (drawing boxes).
* ``--output-dir`` — directory the two overlay PNGs land in. The
  filenames are derived from the test page's stem:
  ``<stem>_annotated.png`` and ``<stem>_predicted.png``.

Defaults inherit from :mod:`evaluate` so running the script with no
arguments reproduces the same "train Hufnagel, classify MOTHRA"
configuration that ``evaluate.classify_page()`` uses by default.

Run::

    cd core/ic_core && uv run python ../tests/sample_input/helpers/run_pipeline.py
    # Or e.g. train + classify the Hufnagel page against itself:
    uv run python ../tests/sample_input/helpers/run_pipeline.py \\
        --train-xml ../tests/fixtures/Hufnagel-example_training_data.xml \\
        --test-page "../tests/sample_input/Hufnagel-example.png" \\
        --test-json "../tests/sample_input/Hufnagel-example_annotations.json"
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ic_core.classifier import run_correction_stage
from ic_core.io_xml import load_glyphs

from evaluate import (  # type: ignore[import-not-found]
    JSON_PATH as DEFAULT_TEST_JSON,
    PAGE_PATH as DEFAULT_TEST_PAGE,
    SAMPLE_DIR,
    TRAINING_XML_PATH as DEFAULT_TRAIN_XML,
    ingest_glyphs_to_classify,
)
from visualize import (  # type: ignore[import-not-found]
    draw_annotation_overlay,
    draw_prediction_overlay,
)

DEFAULT_OUTPUT_DIR = SAMPLE_DIR / "visualization"


def run(
    train_xml: Path,
    test_page: Path,
    test_json: Path,
    output_dir: Path,
) -> None:
    """Load training set, classify test page, write both overlay PNGs."""
    training = load_glyphs(train_xml)
    print(f"Loaded {len(training)} training glyphs from {train_xml.name}")

    # Filter to classId == 2 (real glyphs) to match the test-suite
    # ingest path. Non-glyph MOTHRA classes (staff lines, stray ink)
    # would otherwise be classified too, which is rarely useful.
    working = ingest_glyphs_to_classify(test_page, test_json)
    print(f"Ingested {len(working)} test glyphs from {test_page.name}")

    classified, _ = run_correction_stage(working, training)

    counts = Counter(g.class_name for g in classified)
    print(f"Classified into {len(counts)} distinct classes:")
    for name, n in counts.most_common():
        print(f"  {n:4d}  {name}")

    annotated_out = output_dir / f"{test_page.stem}_annotated.png"
    predicted_out = output_dir / f"{test_page.stem}_predicted.png"
    draw_annotation_overlay(
        image=test_page, annotations=test_json, output=annotated_out
    )
    draw_prediction_overlay(
        image=test_page,
        annotations=test_json,
        output=predicted_out,
        classified=classified,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-xml", type=Path, default=DEFAULT_TRAIN_XML)
    parser.add_argument("--test-page", type=Path, default=DEFAULT_TEST_PAGE)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run(args.train_xml, args.test_page, args.test_json, args.output_dir)


if __name__ == "__main__":
    main()
