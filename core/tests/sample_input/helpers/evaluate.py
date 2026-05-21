"""Test-support helpers for the real sample input.

Shared between ``tests/test_real_input_knn.py`` and the
``visualize.py`` script next door: filtering MOTHRA annotations to
the glyph-only ``classId == 2`` subset, classifying them against the
legacy GameraXML training database, and reporting a quick summary.

Lives under ``tests/`` (not ``ic_core/``) because the MOTHRA
``classId`` is a detector-side concept and ingest is deliberately
classId-agnostic — see :mod:`ic_core.ingest` module docstring.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from ic_core.classifier import (
    InteractiveClassifier,
    run_correction_stage,
)
from ic_core.glyph import Glyph
from ic_core.ingest import ingest_page
from ic_core.io_xml import load_glyphs

HERE = Path(__file__).parent
SAMPLE_DIR = HERE.parent
PAGE_PATH = SAMPLE_DIR / "NZ-Wt MSR-03 109v.png"
JSON_PATH = SAMPLE_DIR / "MOTHRA_NZ-Wt MSR-03 109v_annotations.json"

#: classId values in the MOTHRA JSON that mark actual glyphs to
#: classify. classIds 1 and 3 are non-neume artefacts (staff lines,
#: stray ink) and are excluded.
GLYPH_CLASS_ID: int = 2

#: Default training-database fixture.
# TRAINING_XML_PATH = (
#     SAMPLE_DIR.parent / "fixtures" / "Interactive_Classifier_GameraXML_TrainingData.xml"
# )
TRAINING_XML_PATH = (
    SAMPLE_DIR.parent / "fixtures" / "Hufnagel-example_training_data.xml"
)

def load_annotations(json_path: Path = JSON_PATH) -> dict:
    """Return the parsed MOTHRA annotations document."""
    return json.loads(json_path.read_bytes())


def ingest_glyphs_to_classify(
    page_path: Path = PAGE_PATH,
    json_path: Path = JSON_PATH,
) -> list[Glyph]:
    """Ingest the MOTHRA page, keeping only ``classId == 2`` boxes.

    Filtering is done before handing the JSON to
    :func:`ic_core.ingest.ingest_page`, so the returned glyphs match
    the production ingest path exactly (same UUIDs, same binarisation).
    """
    doc = load_annotations(json_path)
    doc["annotations"] = [
        a for a in doc["annotations"] if a["classId"] == GLYPH_CLASS_ID
    ]
    return ingest_page(
        page_path.read_bytes(),
        json.dumps(doc).encode("utf-8"),
        format="json",
    )


def classify_page(
    page_path: Path = PAGE_PATH,
    json_path: Path = JSON_PATH,
    training_xml: Path = TRAINING_XML_PATH,
) -> tuple[list[Glyph], InteractiveClassifier]:
    """Run the full ingest → train → classify pipeline on the sample page.

    Returns the classified glyphs (in ingest order) and the trained
    classifier so callers can inspect the fitted state.
    """
    working = ingest_glyphs_to_classify(page_path, json_path)
    training = load_glyphs(training_xml)
    return run_correction_stage(working, training)


def print_report(glyphs: list[Glyph]) -> None:
    """Print a one-screen summary of classification results to stdout."""
    confidences = [g.confidence for g in glyphs]
    counts = Counter(g.class_name for g in glyphs)

    if glyphs == []:
        print("No glyphs classified.")
        return

    print(f"Classified {len(glyphs)} glyphs into {len(counts)} classes.")
    print(
        "Confidence: "
        f"mean={statistics.mean(confidences):.3f}  "
        f"median={statistics.median(confidences):.3f}  "
        f"min={min(confidences):.3f}  max={max(confidences):.3f}"
    )
    print("Predicted-class histogram (most common first):")
    for name, n in counts.most_common():
        print(f"  {n:4d}  {name}")


if __name__ == "__main__":
    glyphs, _ = classify_page()
    print_report(glyphs)
