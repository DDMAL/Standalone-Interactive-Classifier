# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

`ic_new` is a **ground-up rewrite** of the Interactive Classifier — a tool for interactively classifying document glyphs (specifically chant neumes) using a k-Nearest Neighbors model. It replaces the legacy Rodan job (Django + Celery + Gamera + Backbone.Marionette SPA) with a modern stack:

- **Algorithm core** — pure Python package, no Django, no Gamera
- **API layer** — FastAPI (planned, `api/` is currently empty)
- **Frontend** — React + Vite (planned, `frontend/` is currently empty)

**Two important deltas vs. the legacy system:**
1. **Input is cropped neume images, not page-level GameraXML.** The upstream connected-components-analysis (CCA) stage is removed. Each input image is one already-segmented neume.
2. **Manual split via CCA is deferred.** `core/ic_core/src/ic_core/splitting.py` exists as a placeholder but is not part of the initial pipeline. Reintroduce only if real data shows crops that contain multiple neumes.

The output format remains **GameraXML** so downstream MEI-encoded pipelines keep working unchanged.

## Reference Codebase (read-only spec)

The legacy implementation lives **outside this repo** at:

```
../Rodan-lite/backend/django/code/jobs/interactive_classifier/
```

Treat it as a behavioral specification, not a base to port. Key files:

- [interactive_classifier.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/interactive_classifier.py) — algorithm core (training, classify, group, export)
- [wrapper.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py) — state machine + user-input vocabulary (lines 389–476)
- [intermediary/gamera_xml.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/gamera_xml.py) — XML schema for export
- [intermediary/gamera_glyph.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/gamera_glyph.py) — Glyph dict shape (UUID generation at line 10)
- [intermediary/run_length_image.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/run_length_image.py) — RLE format (still needed for XML export)
- [KNN_ALGORITHM.md](../Rodan-lite/backend/django/code/jobs/interactive_classifier/KNN_ALGORITHM.md) — algorithm spec
- `../Rodan-lite/backend/django/code/test/files/Interactive_Classifier_*` — GameraXML fixtures (export round-trip oracles)

**Ignore** (Rodan-specific, not migrating): `__init__.py`, `resource_types.yaml`, `gamera_xml_distributor.py`, `interfaces/`, anything mentioning `RodanTask` / `module_loader` / `input_port_types` / `output_port_types`.

## Repository Layout

```
ic_new/
├── core/
│   ├── ic_core/                    # Phase 1: algorithm core (pip-installable, uv-managed)
│   │   ├── pyproject.toml
│   │   ├── uv.lock
│   │   └── src/ic_core/
│   │       ├── glyph.py            # Glyph dataclass (replaces intermediary/gamera_glyph.py)
│   │       ├── image.py            # numpy ↔ PIL conversion; RLE kept for XML round-trips
│   │       ├── features.py         # Feature extraction (replaces Gamera's internal features)
│   │       ├── classifier.py       # kNN training + classify
│   │       ├── grouping.py         # Spatial grouping (manual; auto-grouping needs page coords)
│   │       ├── io_xml.py           # GameraXML read/write (export-authoritative)
│   │       ├── state.py            # ClassifierStateEnum + session dataclass
│   │       └── splitting.py        # DEFERRED — placeholder; not used in initial pipeline
│   └── tests/
│       ├── test_features.py
│       ├── test_io_xml.py
│       └── fixtures/
│           ├── Interactive_Classifier_GameraXML_TrainingData.xml  # legacy fixture, export oracle
│           └── meta/
├── api/                            # Phase 2: FastAPI service (empty — not yet started)
├── frontend/                       # Phase 3: React + Vite UI (empty — not yet started)
└── docs/
    ├── CLAUDE.md                   # this file
    ├── migration_plan.md           # full migration strategy
    └── KNN_ALGORITHM.md            # algorithm spec (copied from Rodan-lite)
```

## Development Commands

The core package uses **uv** for environment + dependency management.

```bash
cd core/ic_core
uv sync                       # install dependencies into .venv
uv run pytest                 # run the test suite
uv run pytest ../tests/test_features.py    # single file
uv run ruff check .           # lint
uv run ruff format .          # format
```

Tests live in `core/tests/` (sibling to the package), not inside `core/ic_core/`.

## Architecture Notes

### Algorithm semantics to preserve verbatim

Documented in [KNN_ALGORITHM.md](KNN_ALGORITHM.md). The non-negotiable behaviors:

1. **Full re-train every round** — discard and rebuild the classifier on each user submission.
2. **`k=1`** default — winner-takes-all, no voting.
3. **Confidence sort order** — ascending; lowest-confidence glyphs surfaced first for review.
4. **Special prefixes `_group`, `_delete`** — stripped by `filter_parts` before training and before export. (The legacy `_split` prefix is dropped along with the deferred split action.)
5. **Manual glyphs feed training, not classification** — `id_state_manual` is the boundary.
6. **UUIDs survive round-trips** — newly created glyphs (manual group, ingestion) get fresh UUIDs; existing ones preserve theirs.
7. **`union_images` on manual group** sets `id_state_manual=True, confidence=1` — the grouped glyph becomes training data immediately.

### Gamera replacement map

| Gamera surface | Replacement here |
|---|---|
| `gamera.knn.kNNInteractive` | `sklearn.neighbors.KNeighborsClassifier` (`k=1` default) |
| Gamera feature vectors | `features.py` using `scikit-image` + `numpy` (versioned — old XML feature blobs not compatible) |
| `gamera.classify.ShapedGroupingFunction` | Custom pairwise pixel-distance using `scipy.ndimage.distance_transform_edt` — requires page coordinates (see below) |
| `gamera.plugins.image_utilities.union_images` | `np.logical_or` over aligned binary masks |
| `gamera.plugins.segmentation.*` | **Deferred** — not needed with per-neume cropped input |
| `gamera.gamera_xml` read/write | Hand-written `lxml` parser in `io_xml.py` (export-only path) |
| Gamera `ONEBIT`/`DENSE` image | `numpy.ndarray` (`bool` for ONEBIT, `uint8` for DENSE) |

### Input format

Inputs are **directories of cropped neume image files** (PNG). Each file = one glyph. Optional sidecar metadata (e.g. JSON) can carry:
- Per-crop source-page position `(page_id, x, y, w, h)` — required to enable spatial auto-grouping; without it, only manual grouping works.
- Per-crop class label — turns the crop into training data instead of an unclassified test glyph.

Legacy GameraXML inputs are **not** supported on the ingestion path; XML is export-only.

## Gotchas

- **Feature vectors are versioned.** Embedded feature vectors in exported GameraXML will differ from the Rodan output. Downstream consumers must not depend on the feature blob — only on the schema and class assignments.
- **Auto-grouping needs page coordinates.** If the input is purely cropped images without source-page positions, `ShapedGroupingFunction`-style grouping has no spatial frame to work in. Either require coordinate metadata or expose only manual grouping in the UI.
- **`splitting.py` is a stub.** Do not wire it into the pipeline or expose a `/split` endpoint without re-opening the deferred-scope discussion.
- **Tests against the legacy fixture** (`Interactive_Classifier_GameraXML_TrainingData.xml`) should treat it as an **export round-trip oracle**, not an ingestion sample. Build new image-based fixtures for ingestion tests.

## Pointers

- Migration strategy and phasing: [migration_plan.md](migration_plan.md)
- Algorithm details and invariants: [KNN_ALGORITHM.md](KNN_ALGORITHM.md)
