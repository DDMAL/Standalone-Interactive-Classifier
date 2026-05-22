# Migrating the Interactive Classifier to a Modern Stack

## Context

The reference codebase — the existing Rodan-lite Interactive Classifier — lives **outside this repo** at `../Rodan-lite/backend/django/code/jobs/interactive_classifier/` (sibling to `ic_new/`). It is a Rodan job (Django + Celery), uses Gamera's `kNNInteractive` for classification, and ships a Backbone.Marionette SPA frontend (83 JS files, gulp/webpack build). All file links in this plan point into that sibling tree; **do not edit those files** — they are the spec.

You want to move it into a **non-Django Python web app, without Gamera, with a React/Vue frontend, and modernize the algorithm**. That makes this a **ground-up rewrite, not a port**. Almost every layer changes; what survives is the *behavioral contract* documented in [KNN_ALGORITHM.md](../Rodan-lite/backend/django/code/jobs/interactive_classifier/KNN_ALGORITHM.md) and [CLAUDE.md](../Rodan-lite/backend/django/code/jobs/interactive_classifier/CLAUDE.md), and the *data structures* in the intermediary layer.

**Input format change:** the original IC sat downstream of a connected-components-analysis (CCA) job that segmented a full page into glyphs and emitted a GameraXML file. **This project removes that upstream CCA stage.** The new system takes **a full-page image paired with a bounding-box annotation document** (MOTHRA JSON or YOLO TXT, produced by an upstream detector) and crops glyphs on the fly at ingestion time. No CC XML, no pre-segmented per-neume PNGs.

**Scope decision — manual grouping kept, auto-grouping deferred.** The legacy IC supported two grouping flavours: *manual* (union N user-selected glyphs into one new training example) and *automatic* (build a spatial-adjacency graph over the working set, propose merges). We **keep manual grouping** because it gives the user a way to build composite training examples in-session, and the page-coordinate frame inherited from the bbox annotations makes the union geometry well-defined. We **defer auto-grouping** because the call on which adjacency function to use (`ShapedGroupingFunction` vs. `BoundingBoxGroupingFunction`) and how to gate runaway graphs hasn't been made yet; the public API for it exists as a stub that raises `NotImplementedError`, and the HTTP endpoint returns 501. Splitting (`_split`) remains out of scope — a crop containing multiple neumes is a defect in the upstream detector, not something IC should patch over.

**Scope decision — kNN is implemented dependency-free.** Rather than reach for `scikit-learn`, we ship a hand-rolled numpy kNN inside `ic_core` (see Phase 1). The math is small (standardise features, compute pairwise Euclidean distances, take the `k` smallest), the dataset sizes we care about (≪10⁵ training glyphs) run fast in plain numpy, and dropping the sklearn dependency keeps the wheel lean and the algorithm fully auditable. If we ever outgrow this, a KD-/Ball-tree can be swapped in behind the same `InteractiveClassifier` API.

The single most important piece of advice up front: **treat the existing code as a specification, not a base.** Trying to incrementally rewrite in-place will mire you in Gamera shims and Rodan plumbing. Start fresh, port the algorithm semantics, and use the existing system only as a behavior oracle.

---

## Recommended Architecture (target project)

Three loosely coupled layers, each independently testable:

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (React + Vite)         ← replaces ic_frontend/        │
│    glyph grid, class panel, modal flows, undo, manual group     │
└──────────────────────────────┬──────────────────────────────────┘
                       REST + WebSocket
┌──────────────────────────────┴──────────────────────────────────┐
│  API layer (FastAPI)             ← replaces wrapper.py          │
│    session state, state machine endpoints, file I/O, auth        │
└──────────────────────────────┬──────────────────────────────────┘
                          Python calls
┌──────────────────────────────┴──────────────────────────────────┐
│  Algorithm core (pure Python pkg) ← replaces interactive_classifier.py
│    + intermediary/  (Gamera-free)                               │
│    feature extraction, dependency-free kNN, manual grouping,    │
│    XML export. Auto-grouping & splitting are deferred stubs.    │
└─────────────────────────────────────────────────────────────────┘
```

Build the core first as a standalone `pip install`-able package with full unit tests. Only then add the API. Only then add the frontend.

---

## Phase 1 — Build the algorithm core (no web, no Django)

This is the bulk of the work and the highest risk.

**Package layout as built (under `core/ic_core/`):**
```
core/ic_core/
├── pyproject.toml
└── src/ic_core/
    ├── glyph.py            # Glyph dataclass (replaces intermediary/gamera_glyph.py); carries
    │                       # optional cached feature_vector + feature_version fields
    ├── image.py            # numpy ↔ PIL conversion, RLE encode/decode, base64 PNG preview
    ├── features.py         # Feature extraction (custom — see "Feature calculation" below);
    │                       # LOGICAL_FEATURES, get_features (cache-aware), ensure_features
    ├── classifier.py       # Dependency-free numpy kNN (replaces prepare_classifier,
    │                       # run_correction_stage)
    ├── ingest.py           # ingest_page() — crops a page image into Glyph objects using a
    │                       # MOTHRA-JSON or YOLO-TXT bbox annotation document
    ├── grouping.py         # manual_group() (implemented); auto_group_shaped /
    │                       # auto_group_bounding_box (deferred stubs that raise
    │                       # NotImplementedError)
    ├── io_xml.py           # GameraXML read/write — export-authoritative; emits the legacy
    │                       # <features> block with version="ic-core/v1"
    ├── state.py            # ClassifierState enum + Session dataclass (direct-mutation
    │                       # state machine; calls ensure_features() before classify rounds)
    └── splitting.py        # Docstring-only stub; not wired into the pipeline
```

Tests live in `core/tests/` (sibling to the package) and currently cover: `test_classifier.py`, `test_features.py`, `test_grouping.py`, `test_ingest.py`, `test_io_xml.py`, `test_io_xml_writer.py`, `test_real_input_knn.py`, `test_state.py`.

Two fixture pools sit under `core/tests/`:
- `fixtures/` — legacy training XML files (`Hufnagel-example_training_data.xml`, `Square_notation-example_training_data.xml`) used as export-round-trip oracles and as canonical examples of the legacy `<features>` block shape that our writer emulates.
- `sample_input/` — real-world ingest pairs (`*.png` + `*_annotations.json`) used by `test_real_input_knn.py` and the visualisation/evaluation helpers under `sample_input/helpers/`.

**Out of scope — not built at all:**
- `splitting.py` (Gamera's `segmentation.cc_analysis` and friends) — the manual-split UX action that ran CCA on a single glyph to break it apart. **Not in scope** because the upstream detector is expected to produce one bbox per neume. If real data later shows crops that contain multiple neumes, that is a defect in the upstream detector and should be fixed there, not patched over in IC. `splitting.py` exists as a docstring-only file for documentation continuity.

**Deferred but stubbed — public API surface preserved:**
- Auto-grouping (`auto_group_shaped` / `auto_group_bounding_box` in `grouping.py`, and `POST /sessions/{id}/auto-group` in the API). The stubs raise `NotImplementedError` / return 501. The design call on which adjacency function to use, how to gate `max_graph_size`, and the criterion for accepting a merge has not been made. The new page+bbox ingest path *does* give us a shared page coordinate frame, so the deferral is no longer blocked on data shape — only on the design.

### Gamera replacement map

| Gamera surface | Replacement | Notes |
|---|---|---|
| `gamera.knn.kNNInteractive` | **Dependency-free hand-rolled kNN** in `classifier.py` (numpy only) | Pairwise Euclidean distance after per-feature standardisation; `np.argpartition` for top-k. `k=1` default for parity; `k` is exposed for experimentation. No `sklearn`, no `BallTree` — the dataset sizes don't justify the extra dependency. |
| Gamera feature vectors (computed internally on glyph images) | Custom `features.py` — see **Feature calculation** below | Re-implemented in `numpy` + `scipy.ndimage` + `skimage.measure`. Versioned via `FEATURE_VERSION`, cached on `Glyph` via optional `feature_vector` / `feature_version` fields and reused across classify rounds. Feature vectors are a **clean break** from Gamera; do not attempt parity at the vector level. |
| `gamera.classify.ShapedGroupingFunction` | `ic_core.grouping.auto_group_shaped` — **deferred stub** | Raises `NotImplementedError`. The page+bbox ingest path provides the page-coordinate frame this needs, so the deferral is on design (adjacency function choice, graph-size gating) not data shape. |
| `gamera.classify.BoundingBoxGroupingFunction` | `ic_core.grouping.auto_group_bounding_box` — **deferred stub** | Same status as `auto_group_shaped`. |
| `gamera.plugins.image_utilities.union_images` | `ic_core.grouping.manual_group` — **implemented** | Bitwise-OR over the input masks into a shared canvas spanning the union of their bounding boxes. The result is a new `Glyph` with `id_state_manual=True`, `confidence=1.0`, and a fresh UUID — joins the training pool immediately. |
| `gamera.plugins.segmentation.<plugin>` | **Not implemented — out of scope** | Splitting is a defect in the upstream detector if it ever becomes necessary; do not bring it into IC. |
| `gamera.gamera_xml` read/write | Hand-written parser using `lxml` | Used for **export only** (no longer read at ingestion). XML stays as the on-disk export format because downstream MEI-encoded pipelines consume it. Keep schema-identical output so existing pipelines accept the files. See [intermediary/gamera_xml.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/gamera_xml.py) for the structure. |
| Gamera `ONEBIT/DENSE` image | `numpy.ndarray` with dtype `bool` (ONEBIT) or `uint8` (DENSE) | Add adapters if you need to maintain XML compatibility. |

### Feature calculation — diff between old and new

The legacy pipeline delegated feature extraction entirely to Gamera: `cknn.generate_features_on_glyphs(...)` computed Gamera's full built-in feature suite on every glyph, and an optional `GameraXML - Feature Selection` file (loaded via `classifier.load_settings(features_file_path)`) restricted the active subset per run. The set of features, their dimensionality, their internal weighting (`perform_splits=True`), and the on-disk format were all opaque to the IC code — IC just asked Gamera to compute them and embedded the result in the exported XML via `WriteXMLFile(..., with_features=True)`.

The new pipeline implements feature extraction directly in [features.py](../core/ic_core/src/ic_core/features.py). Concrete differences:

| Aspect | Legacy (Gamera) | New (`ic_core.features`) |
|---|---|---|
| Where features are computed | Inside Gamera (`cknn.generate_features_on_glyphs`) — opaque C++ implementation | Pure Python: `numpy` + `scipy.ndimage` + `skimage.measure` |
| Which features | Gamera's full built-in suite, optionally filtered by a Feature Selection XML | Fixed Phase-1 set: `aspect_ratio`, `volume`, `nrows_feature`, `ncols_feature`, `compactness`, `nholes`, `volume16regions_*` (16 cells), `hu_moment_*` (7 invariants) — **29 dimensions total** |
| Dimensionality | Variable (depends on Gamera version + selection file) | Fixed at 29 |
| Versioning | None — implicit in Gamera version | Explicit string `FEATURE_VERSION = "ic-core/v1"`; bumped whenever the set or order changes |
| Feature Selection XML | Supported (`classifier.load_settings`) | **Not supported.** If you need a subset, version-bump `FEATURE_VERSION` and ship a new feature set instead |
| `perform_splits=True` (Gamera-internal feature reweighting during training) | On | **Not replicated.** Instead, every feature dimension is standardised (zero mean / unit variance) on the training set; the standardisation parameters are reused at predict time. This is a simpler, well-understood substitute |
| Distance metric | Whatever Gamera's kNN configured (effectively weighted Euclidean) | Plain Euclidean over standardised features |
| Embedded in exported XML? | Yes, via `with_features=True` | Yes — a `<features version="ic-core/v1" scaling="1.0">` block per glyph, with one `<feature name=...>` per logical feature (single value for 1-d, space-separated floats for `volume16regions`/`hu_moment`). Matches the Square_notation fixture shape; downstream consumers **must gate on `version`** before interpreting numbers because the *set* of features differs from Gamera's |
| Class-assignment agreement with legacy | n/a | Expect ≥ 90% on a shared glyph set, **not** bit-equality. See "Risks and gotchas" (2) |

The bottom line: the *schema* of the exported GameraXML is preserved (so MEI-encoded downstream pipelines accept the files), but the *numerical contents* of the feature blobs and the *decision boundary* of the classifier are intentionally non-equivalent. Treat the legacy XML as an export-format spec, not as a source of feature-vector ground truth.

### Algorithm semantics to preserve verbatim

These are documented in `KNN_ALGORITHM.md` and must round-trip identically, or your existing GameraXML test fixtures will fail:

1. **Full re-train every round** — discard and rebuild the classifier on each user submission.
2. **`k=1`** as default — winner-takes-all, no voting.
3. **Confidence sort order** — frontend sorts ascending by confidence; the API must return it that way or the frontend re-sort must replicate it.
4. **Special prefixes `_group`, `_delete`** — stripped by `filter_parts` before training and before export. ([interactive_classifier.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/interactive_classifier.py)) Recognised on ingestion (e.g. when loading legacy GameraXML for round-trip tests) even though no current UI action emits a `_group`-prefixed class name. The legacy `_split` prefix is dropped along with the deferred split action.
5. **Manual glyphs feed training, not classification** — the `id_state_manual` flag is the boundary.
6. **UUIDs survive round-trips** — glyphs carry an `id` field generated in `GameraGlyph.__init__` ([intermediary/gamera_glyph.py:10](../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/gamera_glyph.py#L10)). New glyphs (from ingestion or manual grouping) get fresh UUIDs; existing ones preserve theirs.
7. **`manual_group` sets `id_state_manual=True, confidence=1`** — the union'd glyph becomes training data immediately rather than waiting for the next classify round. Implemented in [core/ic_core/src/ic_core/grouping.py](../core/ic_core/src/ic_core/grouping.py) and exposed via `POST /sessions/{id}/group`.
8. ~~**Manual split outputs `UNCLASSIFIED`, `confidence=0`, `id_state_manual=False`**~~ — dropped with the split action; not implemented.

### Verification for Phase 1

- Unit tests for each of the semantics above, using small synthetic glyphs. See [core/tests/](../core/tests/) — `test_classifier.py`, `test_features.py`, `test_grouping.py`, `test_ingest.py`, `test_io_xml.py`, `test_io_xml_writer.py`, `test_state.py`.
- **Ingestion tests:** point `ingest_page()` at a `(page_image_bytes, annotations_bytes)` pair and verify it produces well-formed `Glyph` objects. MOTHRA-JSON inputs preserve the per-annotation UUID into the glyph id (idempotent re-ingest); YOLO inputs receive fresh UUIDs. Initial state is `id_state_manual=False`, `confidence=0`, `class_name="UNCLASSIFIED"` for every glyph — the upstream detector's class id is intentionally ignored at ingest time.
- **End-to-end smoke:** `test_real_input_knn.py` runs the page+JSON pairs under `core/tests/sample_input/` through ingest → classify → export and writes visualisations to `sample_input/visualization/` for eyeballing.
- **Golden-file tests for export:** the writer tests (`test_io_xml_writer.py`) assert the GameraXML round-trip — including the new `<features version="ic-core/v1">` block — against synthetic glyphs. The legacy fixtures `Hufnagel-example_training_data.xml` and `Square_notation-example_training_data.xml` serve as the on-disk shape oracle for the writer. Class-assignment agreement with the old Gamera-based code on a shared glyph set should be ≥ 90% — track this as a regression metric, not an exact-equality check.

---

## Phase 2 — Build the API layer (FastAPI)

Replaces [wrapper.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py). Rodan's job-and-settings dict is replaced by an explicit `Session` model in `ic_core.state` plus an in-memory session store in `ic_api.store`.

### Endpoints as implemented in [api/src/ic_api/main.py](../api/src/ic_api/main.py)

```
POST   /sessions                          multipart upload: page_image (file) + annotations (file)
                                          + annotations_format ("json" | "yolo")
                                          + optional class_names (JSON-encoded list[str])
                                          → 201, returns SessionDTO with fresh ingest
GET    /sessions/{id}                     fetch current SessionDTO
DELETE /sessions/{id}                     discard the session (204)
POST   /sessions/{id}/classify            run a classify round; body: {"k": int}
POST   /sessions/{id}/glyphs/{gid}        partial update (class_name?, id_state_manual?)
DELETE /sessions/{id}/glyphs/{gid}        drop a glyph from the working set
POST   /sessions/{id}/group               manual group N selected glyphs into one new manual glyph
POST   /sessions/{id}/auto-group          501 — deferred (see Phase 1 "Deferred but stubbed")
POST   /sessions/{id}/classes/{name}/rename    rename across glyphs + training + autocomplete
DELETE /sessions/{id}/classes/{name}      drop a class from the autocomplete list
POST   /sessions/{id}/save                no-op for the in-memory store; returns current state
POST   /sessions/{id}/complete            transition CLASSIFYING → EXPORT, return GameraXML body
```

Errors use a uniform `{"detail": "...", "code": "..."}` envelope (`not_found` → 404, `state_conflict` → 409, `validation_error` → 400, `deferred` → 501). The numpy classifier is fast enough on the sizes we care about that all endpoints respond synchronously — no WebSocket progress stream in v1.

> **Intentionally not implemented:** `POST /sessions/{id}/split` — splitting is an upstream-detector concern. See Phase 1 "Out of scope".

### State persistence

Rodan's settings dict accumulates `@changed_glyphs`, `@grouped_glyphs`, `@deleted_glyphs`, `@renamed_classes`, etc. across user interactions (see wrapper.py lines 389–476). In the new system:

- **Don't** keep that mutation-log pattern. It exists because Rodan re-invokes the task with a fresh dict each round.
- **Do** store authoritative session state directly. `ic_core.state.Session` is the in-memory authority; each endpoint mutates the session directly, no batched mutation queue. The current `ic_api.store.InMemorySessionStore` is process-local — sessions vanish on restart, which is fine for the single-user / local-tool target. Swap for a SQLite/Postgres-backed store when persistence becomes a requirement.
- Grouping does still accumulate state (a `manual_group` call deletes the originals and appends the union'd glyph), but it happens inline in the endpoint handler under a per-session lock — there is no separate `@grouped_glyphs` queue.
- For long-running operations (none currently — auto-classify is fast), the migration path is FastAPI BackgroundTasks for single-user, or Celery/Dramatiq if we ever need cross-process work.

### Auth

The current frontend hits `auth/Authenticator.js` every 5 seconds to refresh a Rodan token. In a non-Django context, pick one of:
- JWT with refresh token (if multi-user)
- Plain session cookie (if internal tool)
- No auth (if local-only desktop-style tool)

Don't carry the 5-second-refresh ping over — it's a Rodan quirk, not a requirement.

---

## Phase 3 — Frontend rewrite (React + Vite recommended)

The existing SPA (83 JS files) is a thoughtful piece of code despite being old. **Treat it as a UX spec, not a codebase to port.**

### What to keep

- The grid-of-glyphs interaction model
- Class panel with rename/delete
- Modal flows for delete confirmation and manual group (the split modal is dropped along with the split action; the auto-group modal is parked behind the 501 endpoint until that work is unblocked)
- Undo stack
- Ascending-confidence sort order
- Keyboard shortcuts (read `ic_frontend/public/js/app/views/` for the inventory)

### What to discard or replace

| Old | New |
|---|---|
| Backbone.Model / Backbone.Collection | React state + Zustand or TanStack Query |
| Backbone.Radio (pub/sub channels) | React Context, or just lifting state up |
| Marionette views | React components |
| `events/*.js` constant files | TypeScript discriminated unions / event types |
| Gulp + webpack 1 + Karma + Jest | Vite + Vitest |
| `Authenticator.js` (5-sec token ping) | Standard auth, see Phase 2 |

### Concrete recommendations

- **React + Vite + TypeScript.** Vite gives you a usable dev server in 1 minute vs. the existing gulp pipeline.
- **Use a virtualized grid** (`@tanstack/react-virtual`) — large pages produce thousands of glyphs and the old SPA rendered them all into the DOM. Performance is the main UX upgrade you get for free.
- **Display layer for glyphs:** base64-decode the RLE on the backend (or in a Web Worker on the frontend) and render to `<canvas>`. Don't try to round-trip RLE through React props.
- **Vue is equally viable** if you prefer it — the architecture above maps cleanly. Pick whichever your target project already uses.

---

## Critical files to reference (do not edit — they are the spec)

- [interactive_classifier.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/interactive_classifier.py) — algorithm core, port semantics from here
- [wrapper.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/wrapper.py) — state machine + user-input handling; especially lines 389–476 for the input action vocabulary
- [intermediary/gamera_xml.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/gamera_xml.py) — XML schema you must read/write
- [intermediary/gamera_glyph.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/gamera_glyph.py) — Glyph dict shape
- [intermediary/run_length_image.py](../Rodan-lite/backend/django/code/jobs/interactive_classifier/intermediary/run_length_image.py) — RLE format
- [KNN_ALGORITHM.md](../Rodan-lite/backend/django/code/jobs/interactive_classifier/KNN_ALGORITHM.md) — algorithm spec (excellent doc; lean on it)
- [CLAUDE.md](../Rodan-lite/backend/django/code/jobs/interactive_classifier/CLAUDE.md) — architecture overview
- [../Rodan-lite/backend/django/code/test/files/](../Rodan-lite/backend/django/code/test/files/) — real GameraXML fixtures for regression tests
- `ic_frontend/public/js/app/views/` — UI behavior spec for the new frontend

## Files to ignore (Rodan-specific, not reusable)

- `__init__.py`, `resource_types.yaml`, `gamera_xml_distributor.py`, `interfaces/interactive_classifier.html`
- Any `module_loader`, `RodanTask`, `input_port_types`, `output_port_types` patterns

---

## Risks and gotchas

1. **Feature vector incompatibility — by design.** Gamera computes feature vectors as part of its XML output (`with_features=True` in `WriteXMLFile`). Our new system's feature vectors are a different set, computed by different code, in a fixed 29-dimensional layout (see "Feature calculation" above). The exported XML carries a `<features version="ic-core/v1" scaling="1.0">` block per glyph so downstream consumers can detect the break before interpreting numbers; the structure mirrors the legacy `<feature name=...>` element shape so strict legacy parsers still accept the file. Downstream MEI pipelines that consumed Gamera's specific feature values must be re-pointed at the class assignments rather than the embedded numbers.

2. **Classification accuracy will differ.** Gamera's kNN with `perform_splits=True` and selected features is not a stock 1-NN. Our hand-rolled dependency-free kNN on our own 29-dimension feature vectors will produce different decisions. Plan for accuracy validation against a held-out set early — don't discover this at the end. Target ≥ 90% class-assignment agreement on a shared glyph set as a regression signal, not exact equality.

3. **`perform_splits=True` is not replicated.** Gamera's training-time feature reweighting has no direct analogue here. We rely on per-feature standardisation (zero-mean / unit-variance) computed on the training set instead. Flag this for the domain expert as part of accuracy validation; if results diverge unacceptably, the right response is feature engineering inside `features.py`, not bringing back Gamera-style internals.

4. **Auto-grouping is deferred, but no longer blocked on data shape.** With the new page+bbox ingest path, every `Glyph` carries `(ulx, uly, ncols, nrows)` in a shared page coordinate frame, which is exactly what spatial grouping needs. What is still pending is the *design call*: which adjacency function (`ShapedGroupingFunction` vs. `BoundingBoxGroupingFunction`), how to gate `max_graph_size` to keep dense pages from blowing up the graph, and what acceptance criterion to use for proposed merges. The stubs in `grouping.py` and the 501 endpoint preserve the API surface so the frontend can wire affordances without conditional imports.

5. **GameraXML XML schema is undocumented except by example.** Use the fixtures in `../Rodan-lite/backend/django/code/test/files/` and `core/tests/fixtures/` as your ground truth; write a parser test for each variant you encounter.

6. **Session size.** The Rodan `settings` dict holds *all glyphs* including base64-encoded images. For large pages this is megabytes per session. In FastAPI/Postgres, store the heavy image data on disk or object storage, keep only references in the DB. The current in-memory store is fine for single-user.

7. **The state machine has implicit ordering.** Look at the order of operations inside each state in wrapper.py: `add_grouped_glyphs → update_changed_glyphs → remove_deleted_glyphs → remove_deleted_classes → update_renamed_classes → filter_parts`. Under direct mutation the ordering is the user's call order, but the *semantics* still apply (e.g. renaming a class after deleting it skips the delete). Read the wrapper.py ordering as a guide to which operations interact, not as a sequence to replay.

8. **`gamera_xml_distributor.py` is dead weight.** It's a workflow-fanout helper specific to Rodan pipelines. Do not migrate it.

9. **Test fixtures.** Three sources live under [core/tests/](../core/tests/):
   - `fixtures/Hufnagel-example_training_data.xml`, `fixtures/Square_notation-example_training_data.xml` — legacy GameraXML training files, used as the canonical shape for the writer's `<features>` block and as oracle data for export round-trip tests.
   - `sample_input/*.png` + `sample_input/*_annotations.json` — real page+JSON ingest pairs, used by `test_real_input_knn.py` and the helper scripts.
   - `sample_input/visualization/` — generated diagnostic images, *not* test inputs; safe to delete and regenerate.

---

## Verification end-to-end

After all three phases, you should be able to:

1. **Algorithm-only:** `cd core/ic_core && uv run pytest ../tests` — all unit tests pass; class-assignment agreement with old Gamera code is ≥ 90% on the regression fixtures.
2. **API smoke test:** `cd api && uv run ic-api`, then `POST /sessions` with a page image + bbox JSON (or YOLO TXT), walk through `/classify` → manual `/glyphs/{gid}` corrections → optional `/group` → `/complete`, get a valid output GameraXML.
3. **Manual UI test:** load the new React/Vue frontend in a browser, upload a page+bbox pair, perform: auto-classify, manual reassignment, manual group, save, complete. Compare visually against the old SPA running on the existing Rodan deployment, accounting for the input-format difference (split is not exercised — it is out of scope; auto-group is parked behind the 501 endpoint).
4. **Regression against Rodan:** keep the old Rodan instance running in parallel for a few weeks; run the same input through both, diff the classified glyph output. Investigate any disagreement above the noise floor before retiring the old system.

---

## TL;DR migration order

1. Stand up `core/ic_core/` as a pure-Python package. Port semantics from `../Rodan-lite/.../interactive_classifier.py` and `intermediary/`. **Ingest a page image + bbox annotation file (MOTHRA JSON / YOLO TXT); skip the CCA and splitting modules; implement manual grouping but leave auto-grouping as a deferred stub.** Implement the kNN with numpy directly — no `sklearn`. Implement the feature set as described in "Feature calculation" above and embed the result in the exported XML behind a `version="ic-core/v1"` attribute. Unit-test everything. Validate accuracy vs. Gamera on fixtures (using a shared glyph set so the comparison is meaningful despite the input-format and feature-vector changes).
2. Wrap it in a FastAPI service. Replace the Rodan state machine with direct-mutation endpoints + an in-memory session store; defer DB-backed persistence until single-user is no longer enough. `/group` works; `/auto-group` returns 501; no `/split`.
3. Build a fresh React/Vite frontend using the old SPA as a UX spec. Manual-group UI is in scope; auto-group and split modals are not.
4. Run the new and old systems side-by-side on real data until you're confident, then retire the Rodan job. If real data turns out to contain multi-neume crops, that is a defect in the upstream detector — fix it there, not in IC.
