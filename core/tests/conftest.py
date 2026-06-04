"""Pytest configuration shared by every test module under ``tests/``.

Lives next to the test files so pytest picks it up regardless of
whether the suite is invoked from ``core/``, ``core/ic_core/``, or
the repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``core/scripts/`` importable as top-level modules (``evaluate``,
# ``visualize``, …) for tests that share helpers with the CLI scripts.
# Done in conftest.py rather than pyproject.toml because pytest's
# rootdir resolution doesn't reach ``core/ic_core/pyproject.toml``
# when tests are invoked from outside that directory.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that retrain the classifier across folds — skipped by default",
    )
