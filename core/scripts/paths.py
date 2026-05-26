"""Centralised path config for scripts and tests.

Every path defaults to a location under ``core/`` and may be
overridden via the corresponding ``IC_*`` environment variable, which
lets CI runs or local experiments target an alternate dataset without
code edits.

The library package (:mod:`ic_core`) deliberately has no path
knowledge — it takes bytes/paths as arguments. Path centralisation
belongs to the callers (scripts, tests), so this module lives under
``core/scripts/`` (already on ``sys.path`` for tests via
``tests/conftest.py``).

Env vars
--------
=========================  ==============================================
``IC_DATA_DIR``            Root data dir. Default: ``core/data``
``IC_TRAIN_DIR``           Default: ``$DATA_DIR/train``
``IC_TEST_DIR``            Default: ``$DATA_DIR/test``
``IC_DERIVED_DIR``         Default: ``$DATA_DIR/derived``
``IC_VIS_DIR``             Default: ``$DERIVED_DIR/visualization``
``IC_TEST_PAGE``           Default test page image.
``IC_TEST_JSON``           Default MOTHRA JSON for the test page.
``IC_TEST_YOLO``           Default YOLO txt for the test page.
``IC_TRAINING_XML``        Default training GameraXML database.
``IC_CSV_VOCAB``           Default canonical-vocabulary CSV.
=========================  ==============================================
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    """Return ``Path(os.environ[var])`` if set and non-empty, else ``default``."""
    val = os.environ.get(var)
    return Path(val) if val else default


SCRIPTS_DIR = Path(__file__).resolve().parent
CORE_DIR = SCRIPTS_DIR.parent

DATA_DIR = _env_path("IC_DATA_DIR", CORE_DIR / "data")
TRAIN_DIR = _env_path("IC_TRAIN_DIR", DATA_DIR / "train")
TEST_DIR = _env_path("IC_TEST_DIR", DATA_DIR / "test")
DERIVED_DIR = _env_path("IC_DERIVED_DIR", DATA_DIR / "derived")
VIS_DIR = _env_path("IC_VIS_DIR", DERIVED_DIR / "visualization")

TEST_PAGE = _env_path("IC_TEST_PAGE", TEST_DIR / "image_hfn_sample.png")
TEST_JSON = _env_path(
    "IC_TEST_JSON",
    TEST_DIR / "image_hfn_sample_annotations.json",
)
TEST_YOLO = _env_path("IC_TEST_YOLO", TEST_DIR / "NZ-Wt MSR-03 109v.txt")
TRAINING_XML = _env_path(
    "IC_TRAINING_XML", DERIVED_DIR / "Hufnagel_training_data.xml"
)
CSV_VOCAB = _env_path(
    "IC_CSV_VOCAB",
    TRAIN_DIR / "csv-square_notation_neume_level_newest.csv",
)
