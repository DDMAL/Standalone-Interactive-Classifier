# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

`ic_new` is a **ground-up rewrite** of the Interactive Classifier — a tool for interactively classifying document glyphs (specifically chant neumes) using a k-Nearest Neighbors model. It replaces the legacy Rodan job (Django + Celery + Gamera + Backbone.Marionette SPA) with a modern stack:

- **Algorithm core** — pure Python package (`core/ic_core/`), no Django, no Gamera
- **API layer** — FastAPI service (`api/`), in-memory session store
- **Frontend** — React + Vite (planned, `frontend/` is currently empty)

**Important deltas vs. the legacy system:**
1. **Input is a page image + bbox annotation file, not page-level GameraXML.** Upstream connected-components-analysis (CCA) is removed. `ingest_page()` accepts raw `(page_image_bytes, annotations_bytes)` plus a format discriminator (`"json"` for MOTHRA JSON, `"yolo"` for YOLO TXT) and crops glyphs from the page on the fly.
2. **Manual grouping is implemented; auto-grouping is deferred.** [grouping.py:manual_group](../core/ic_core/src/ic_core/grouping.py) bitwise-ORs N selected glyphs into one new training example (`id_state_manual=True, confidence=1.0`, fresh UUID) — exposed via `POST /sessions/{id}/group`. `auto_group_shaped` / `auto_group_bounding_box` exist as stubs that raise `NotImplementedError`; `POST /sessions/{id}/auto-group` returns 501. The page+bbox ingest gives auto-grouping the page-coordinate frame it needs; the deferral is now about the design (adjacency-function choice, graph-size gating), not data shape.
3. **Manual split via CCA is out of scope.** A crop containing multiple neumes is an upstream-detector defect, not something IC should patch over. [splitting.py](../core/ic_core/src/ic_core/splitting.py) is a docstring-only file kept for documentation continuity.
4. **kNN is hand-rolled, dependency-free.** No `scikit-learn`. The full implementation is numpy-only in [classifier.py](../core/ic_core/src/ic_core/classifier.py): standardise features → pairwise Euclidean → `np.argpartition` for top-k. The interface mirrors what a sklearn-based version would look like, so a Ball-tree backend can be slotted in later if needed.
5. **Feature calculation is a clean break from Gamera.** See "Feature calculation" below. Feature vectors are cached on `Glyph` (optional `feature_vector` / `feature_version` fields) so the "full re-train every round" loop reuses computation across rounds.

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
│   │       ├── glyph.py            # Glyph dataclass; optional feature_vector / feature_version
│   │       │                       # cache fields (excluded from __eq__ / repr)
│   │       ├── image.py            # numpy ↔ PIL conversion, RLE encode/decode, base64 PNG preview
│   │       ├── features.py         # Feature extraction; LOGICAL_FEATURES, FEATURE_VERSION,
│   │       │                       # get_features (cache-aware), ensure_features
│   │       ├── classifier.py       # Dependency-free numpy kNN (no sklearn)
│   │       ├── ingest.py           # ingest_page() — page image + bbox JSON/YOLO → Glyph list
│   │       ├── grouping.py         # manual_group (implemented); auto_group_* (deferred stubs)
│   │       ├── io_xml.py           # GameraXML read/write; writer emits the legacy <features>
│   │       │                       # block carrying version="ic-core/v1"
│   │       ├── state.py            # ClassifierState enum + Session dataclass (direct mutation)
│   │       └── splitting.py        # Docstring-only stub; NOT wired into the pipeline
│   └── tests/
│       ├── test_classifier.py
│       ├── test_features.py
│       ├── test_grouping.py
│       ├── test_ingest.py
│       ├── test_io_xml.py
│       ├── test_io_xml_writer.py
│       ├── test_real_input_knn.py
│       ├── test_state.py
│       ├── conftest.py
│       ├── fixtures/
│       │   ├── Hufnagel-example_training_data.xml         # legacy oracle for writer shape
│       │   └── Square_notation-example_training_data.xml  # canonical <features> block example
│       └── sample_input/           # page+JSON pairs for end-to-end runs + helper scripts
│           ├── Hufnagel-example.png
│           ├── Hufnagel-example_annotations.json
│           ├── NZ-Wt MSR-03 109v.png
│           ├── MOTHRA_NZ-Wt MSR-03 109v_annotations.json
│           ├── helpers/            # run_pipeline.py, evaluate.py, visualize.py, csv converter
│           └── visualization/      # generated diagnostics (safe to delete and regenerate)
├── api/                            # Phase 2: FastAPI service (implemented)
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── README.md
│   ├── src/ic_api/
│   │   ├── main.py                 # FastAPI app + endpoints (sessions, classify, group, …)
│   │   ├── schemas.py              # Pydantic DTOs and conversion helpers
│   │   └── store.py                # InMemorySessionStore (process-local; not persisted)
│   └── tests/test_api.py
├── frontend/                       # Phase 3: React + Vite UI (not yet started)
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
4. **Special prefixes `_group`, `_delete`** — stripped by `filter_parts` before training and before export. Recognised when loading legacy GameraXML even though no current UI action emits a `_group`-prefixed class name. The `_split` prefix is dropped along with the deferred split action.
5. **Manual glyphs feed training, not classification** — `id_state_manual` is the boundary.
6. **UUIDs survive round-trips** — newly created glyphs (from ingestion or manual grouping) get fresh UUIDs; existing ones preserve theirs. For MOTHRA-JSON ingest, the per-annotation `id` becomes the glyph id so re-ingesting the same JSON produces stable ids.
7. **`manual_group` sets `id_state_manual=True, confidence=1.0`** — the union'd glyph becomes training data immediately rather than waiting for the next classify round.

### Gamera replacement map

| Gamera surface | Replacement here |
|---|---|
| `gamera.knn.kNNInteractive` | **Hand-rolled numpy kNN** in `classifier.py` (`k=1` default). No `sklearn` — pairwise Euclidean over standardised features, `np.argpartition` for top-k |
| Gamera feature vectors | `features.py` using `numpy` + `scipy.ndimage` + `skimage.measure`. Fixed 29-dim set, versioned via `FEATURE_VERSION`, cached on `Glyph`. See "Feature calculation" below |
| `gamera.classify.ShapedGroupingFunction` | `grouping.auto_group_shaped` — **deferred stub** (raises `NotImplementedError`) |
| `gamera.classify.BoundingBoxGroupingFunction` | `grouping.auto_group_bounding_box` — **deferred stub** |
| `gamera.plugins.image_utilities.union_images` | `grouping.manual_group` — **implemented**; bitwise-OR over masks in a shared canvas spanning the union of bboxes |
| `gamera.plugins.segmentation.*` | **Not implemented — out of scope** (upstream-detector defect if it surfaces) |
| `gamera.gamera_xml` read/write | Hand-written `lxml` parser in `io_xml.py`. Reader exists for round-trip tests; writer is the authoritative export path |
| Gamera `ONEBIT`/`DENSE` image | `numpy.ndarray` (`bool` for ONEBIT, `uint8` for DENSE) |

### Feature calculation — diff vs. legacy

The legacy IC handed feature extraction to Gamera entirely (`cknn.generate_features_on_glyphs`) and optionally filtered the active set via a `GameraXML - Feature Selection` file. We implement features directly in [features.py](../core/ic_core/src/ic_core/features.py) with a fixed, explicit set:

| Aspect | Legacy (Gamera) | New (`ic_core.features`) |
|---|---|---|
| Compute path | Gamera C++ internals (opaque) | Python: `numpy` + `scipy.ndimage` + `skimage.measure` |
| Feature set | Gamera's full suite, optionally subsetted by Feature-Selection XML | Fixed: `aspect_ratio`, `volume`, `nrows_feature`, `ncols_feature`, `compactness`, `nholes`, `volume16regions_*` (×16), `hu_moment_*` (×7) — **29 dimensions** |
| Dimensionality | Variable | Fixed at 29 |
| Versioning | None (implicit in Gamera version) | Explicit `FEATURE_VERSION = "ic-core/v1"`; bumped on any change |
| Feature Selection XML | Supported | **Not supported** — bump `FEATURE_VERSION` and re-train instead |
| `perform_splits=True` reweighting | On (Gamera-internal) | **Not replicated.** Replaced by per-feature standardisation (zero-mean / unit-variance) on the training set, reused at predict time |
| Distance metric | Gamera kNN's weighted Euclidean | Plain Euclidean over standardised features |
| Embedded in exported GameraXML? | Yes (`with_features=True`) | Yes — `<features version="ic-core/v1" scaling="1.0">` per glyph, one `<feature name=...>` per logical feature (single value for 1-d; space-separated floats for `volume16regions` and `hu_moment`). Mirrors the [Square_notation fixture](../core/tests/fixtures/Square_notation-example_training_data.xml) element shape. Downstream consumers **must check `version`** before interpreting numbers |
| Where the vector lives in memory | Computed-on-demand per glyph inside Gamera | Cached on `Glyph` (`feature_vector` / `feature_version` optional fields, `compare=False` so dataclass equality still works). `Glyph.classify_manual` / `classify_automatic` use `dataclasses.replace` so the cache survives label-change operations. `Session.classify` calls `ensure_features` before training, so the cache is materialised once and reused across rounds |
| Accuracy vs. legacy | n/a | Target ≥ 90% class-assignment agreement on a shared glyph set; bit-equality is not a goal |

Schema-compatible XML out (legacy parsers accept the file), intentionally non-equivalent feature numbers and decision boundary inside (gated by the `version` attribute).

### Input format

Inputs are **whole-page images paired with JSON files describing per-glyph bounding boxes**. IC crops each glyph from the page at ingestion time using the bbox JSON. The JSON may also carry per-glyph class labels, which turn a glyph into training data instead of an unclassified test glyph.

Legacy GameraXML inputs are **not** supported on the ingestion path; XML is export-only.

## Gotchas

- **Feature vectors are versioned.** The exported `<features>` block carries `version="ic-core/v1"`. Downstream consumers that read feature values **must gate on the version** — the set of features and the math behind them differ from Gamera's. The schema (element shape) is preserved so strict legacy parsers still accept the file.
- **`splitting.py` is a docstring-only stub.** Out of scope. Do not wire it into the pipeline or expose a `/split` endpoint without re-opening the scope discussion. A crop containing multiple neumes is an upstream-detector defect.
- **`grouping.py` splits into implemented + deferred halves.** `manual_group` is real code, called by `Session.manual_group` and exposed at `POST /sessions/{id}/group`. `auto_group_shaped` / `auto_group_bounding_box` raise `NotImplementedError` and the matching API endpoint returns 501 — adding real implementations requires picking an adjacency function and gating runaway graphs.
- **Glyph equality and the feature cache.** `feature_vector` is `compare=False, repr=False` so dataclass `__eq__` / `__hash__` / printing keep working with `ndarray`. Don't accidentally include it in equality checks elsewhere.
- **Legacy fixture filename:** `core/tests/fixtures/` holds `Hufnagel-example_training_data.xml` and `Square_notation-example_training_data.xml`. An older test (`test_features.py`) still references a renamed `Interactive_Classifier_GameraXML_TrainingData.xml` — that's a known pre-existing failure unrelated to current work; fix the path when convenient. Use these fixtures as **export shape oracles** for the writer, not as ingestion samples (ingestion takes page+bbox bytes).

## Pointers

- Migration strategy and phasing: [migration_plan.md](migration_plan.md)
- Algorithm details and invariants: [KNN_ALGORITHM.md](KNN_ALGORITHM.md)
