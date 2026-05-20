"""Pytest configuration shared by every test module under ``tests/``.

Lives next to the test files so pytest picks it up regardless of
whether the suite is invoked from ``core/``, ``core/ic_core/``, or
the repo root.
"""
from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that retrain the classifier across folds — skipped by default",
    )
