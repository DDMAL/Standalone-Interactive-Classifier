# Migrating the Interactive Classifier to a Modern Stack

## Context

The current Interactive Classifier lives in `backend/django/code/jobs/interactive_classifier/` as a Rodan job (Django + Celery), uses Gamera's `kNNInteractive` for classification, and ships a Backbone.Marionette SPA frontend (83 JS files, gulp/webpack build).

You want to move it into a **non-Django Python web app, without Gamera, with a React/Vue frontend, and modernize the algorithm**. That makes this a **ground-up rewrite, not a port**. Almost every layer changes; what survives is the *behavioral contract* documented in [KNN_ALGORITHM.md](backend/django/code/jobs/interactive_classifier/KNN_ALGORITHM.md) and [CLAUDE.md](backend/django/code/jobs/interactive_classifier/CLAUDE.md), and the *data structures* in the intermediary layer.

The single most important piece of advice up front: **treat the existing code as a specification, not a base.** Trying to incrementally rewrite in-place will mire you in Gamera shims and Rodan plumbing. Start fresh, port the algorithm semantics, and use the existing system only as a behavior oracle.

---

## Recommended Architecture (target project)

Three loosely coupled layers, each independently testable:

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (React + Vite)         ← replaces ic_frontend/        │
│    glyph grid, class panel, modal flows, undo, group/split      │
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
│    feature extraction, kNN, grouping, splitting, serialization  │
└─────────────────────────────────────────────────────────────────┘
```

Build the core first as a standalone `pip install`-able package with full unit tests. Only then add the API. Only then add the frontend.

---

## Phase 1 — Build the algorithm core (no web, no Django)

This is the bulk of the work and the highest risk.

**New package layout:**
```
interactive_classifier_core/
├── pyproject.toml
├── interactive_classifier_core/
│   ├── glyph.py            # Glyph dataclass (replaces intermediary/gamera_glyph.py)
│   ├── image.py            # RLE ↔ numpy ↔ PIL conversion (replaces run_length_image.py)
│   ├── features.py         # Feature extraction (replaces Gamera's internal features)
│   ├── classifier.py       # kNN training + classify (replaces prepare_classifier, run_correction_stage)
│   ├── grouping.py         # Spatial grouping (replaces group_and_correct + Gamera grouping funcs)
│   ├── splitting.py        # Connected-component splitting (replaces gamera.segmentation)
│   ├── io_xml.py           # GameraXML read/write (authoritative on-disk format)
│   └── state.py            # ClassifierStateEnum + session dataclass
└── tests/
    ├── test_features.py
    ├── test_classifier.py
    ├── test_grouping.py
    ├── test_splitting.py
    └── fixtures/           # real glyph XML files copied from this repo's test/files/
```

### Gamera replacement map

| Gamera surface | Replacement | Notes |
|---|---|---|
| `gamera.knn.kNNInteractive` | `sklearn.neighbors.KNeighborsClassifier` (or `BallTree` for speed) | Keep `k=1` initially for parity; expose `k` as a parameter. |
| Gamera feature vectors (computed internally on glyph images) | Custom `features.py` using `scikit-image` + `numpy` | Reimplement the subset Gamera used: aspect ratio, area, moments, projection histograms. See `KNN_ALGORITHM.md` for which features matter. Document and version the feature vector — old XML files won't be feature-compatible. |
| `gamera.classify.ShapedGroupingFunction` | Custom: pairwise pixel-distance using `scipy.ndimage.distance_transform_edt` | Builds adjacency for graph grouping. |
| `gamera.classify.BoundingBoxGroupingFunction` | Pure numpy bounding-box distance check | Trivial. |
| `gamera.plugins.image_utilities.union_images` | `np.logical_or` over aligned binary images | Trivial — recompute the bounding box and OR the masks. |
| `gamera.plugins.segmentation.<plugin>` | `scipy.ndimage.label` + `skimage.measure.regionprops` for connected components; expose other splitters as needed | Most-used plugin is `cc_analysis`; others can be added on demand. |
| `gamera.gamera_xml` read/write | Hand-written parser using `lxml` | XML is the authoritative on-disk format — we stay in XML because this work feeds into MEI-encoded pipelines downstream. Keep schema-identical output so existing pipelines accept the files. See [intermediary/gamera_xml.py](backend/django/code/jobs/interactive_classifier/intermediary/gamera_xml.py) for the structure. |
| Gamera `ONEBIT/DENSE` image | `numpy.ndarray` with dtype `bool` (ONEBIT) or `uint8` (DENSE) | Add adapters if you need to maintain XML compatibility. |

### Algorithm semantics to preserve verbatim

These are documented in `KNN_ALGORITHM.md` and must round-trip identically, or your existing GameraXML test fixtures will fail:

1. **Full re-train every round** — discard and rebuild the classifier on each user submission.
2. **`k=1`** as default — winner-takes-all, no voting.
3. **Confidence sort order** — frontend sorts ascending by confidence; the API must return it that way or the frontend re-sort must replicate it.
4. **Special prefixes `_split`, `_group`, `_delete`** — stripped by `filter_parts` before training and before export. ([interactive_classifier.py](backend/django/code/jobs/interactive_classifier/interactive_classifier.py))
5. **Manual glyphs feed training, not classification** — the `id_state_manual` flag is the boundary.
6. **UUIDs survive round-trips** — glyphs carry an `id` field generated in `GameraGlyph.__init__` ([intermediary/gamera_glyph.py:10](backend/django/code/jobs/interactive_classifier/intermediary/gamera_glyph.py#L10)). New glyphs (manual group, split) get fresh UUIDs; existing ones preserve theirs.
7. **`union_images` on manual group** sets `id_state_manual=True, confidence=1` — the grouped glyph becomes training data immediately.
8. **Manual split outputs `UNCLASSIFIED`, `confidence=0`, `id_state_manual=False`** — re-classified on the next round.

### Verification for Phase 1

- Unit tests for each of the eight semantics above, using small synthetic glyphs.
- **Golden-file tests:** take 2–3 real GameraXML files from `backend/django/code/test/files/` (e.g. `Interactive_Classifier_GameraXML_TrainingData.xml` referenced in `gamera_xml_distributor.py:43`), run the same training data through both the old Gamera-based code and the new core, and diff the outputs. They will not be byte-identical (different feature vectors), but the *class assignments* should agree on a high fraction of glyphs. Use this as a regression metric, not an exact-equality check.

---

## Phase 2 — Build the API layer (FastAPI)

Replaces [wrapper.py](backend/django/code/jobs/interactive_classifier/wrapper.py). Rodan's job-and-settings dict gets replaced by an explicit session model.

### Suggested endpoints

```
POST   /sessions                       create session, upload connected-components XML +
                                       optional training XML + class-names text
                                       → returns session_id, initial glyph set
GET    /sessions/{id}                  fetch current state (glyphs, classes, state enum)
POST   /sessions/{id}/classify         run a CLASSIFYING round (auto-classify non-manual glyphs)
POST   /sessions/{id}/group            manual group: takes glyph IDs + class_name
POST   /sessions/{id}/split            manual split: takes glyph_id + split_type
POST   /sessions/{id}/auto-group       triggers GROUP_AND_CLASSIFY with user_options
POST   /sessions/{id}/glyphs/{gid}     update a glyph (class assignment, delete flag)
POST   /sessions/{id}/save             persist to DB without exporting
POST   /sessions/{id}/complete         EXPORT_XML, return final GameraXML
DELETE /sessions/{id}                  cleanup
WS     /sessions/{id}/stream           push progress events for long auto-classify rounds
```

### State persistence

Rodan's settings dict accumulates `@changed_glyphs`, `@grouped_glyphs`, `@deleted_glyphs`, `@renamed_classes`, etc. across user interactions (see wrapper.py lines 389–476). In the new system:

- **Don't** keep that mutation-log pattern. It exists because Rodan re-invokes the task with a fresh dict each round.
- **Do** store authoritative session state in a database table (Postgres + SQLAlchemy, or just SQLite for a single-user tool). Each endpoint mutates the session directly; no batched mutation queue.
- For long-running auto-classify (the slow operation), use a background task (Celery, Dramatiq, or just FastAPI BackgroundTasks for single-user) and stream progress over WebSocket.

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
- Modal flows for group, split, delete confirmation
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

- [interactive_classifier.py](backend/django/code/jobs/interactive_classifier/interactive_classifier.py) — algorithm core, port semantics from here
- [wrapper.py](backend/django/code/jobs/interactive_classifier/wrapper.py) — state machine + user-input handling; especially lines 389–476 for the input action vocabulary
- [intermediary/gamera_xml.py](backend/django/code/jobs/interactive_classifier/intermediary/gamera_xml.py) — XML schema you must read/write
- [intermediary/gamera_glyph.py](backend/django/code/jobs/interactive_classifier/intermediary/gamera_glyph.py) — Glyph dict shape
- [intermediary/run_length_image.py](backend/django/code/jobs/interactive_classifier/intermediary/run_length_image.py) — RLE format
- [KNN_ALGORITHM.md](backend/django/code/jobs/interactive_classifier/KNN_ALGORITHM.md) — algorithm spec (excellent doc; lean on it)
- [CLAUDE.md](backend/django/code/jobs/interactive_classifier/CLAUDE.md) — architecture overview
- [backend/django/code/test/files/](backend/django/code/test/files/) — real GameraXML fixtures for regression tests
- `ic_frontend/public/js/app/views/` — UI behavior spec for the new frontend

## Files to ignore (Rodan-specific, not reusable)

- `__init__.py`, `resource_types.yaml`, `gamera_xml_distributor.py`, `interfaces/interactive_classifier.html`
- Any `module_loader`, `RodanTask`, `input_port_types`, `output_port_types` patterns

---

## Risks and gotchas

1. **Feature vector incompatibility.** Gamera computes feature vectors as part of its XML output (`with_features=True` in `WriteXMLFile`). Your new system's feature vectors will be different. Because we are staying in the GameraXML format to feed MEI-encoded downstream pipelines, the output XML must remain schema-compatible — but the embedded feature vectors themselves are versioned and treated as a clean break (downstream consumers should not depend on them). Document the feature-vector version in the XML so old/new files can be distinguished.

2. **Classification accuracy may differ.** Gamera's kNN with `perform_splits=True` and selected features is not a stock 1-NN. Stock sklearn `KNeighborsClassifier(n_neighbors=1)` on your own feature vectors will produce different decisions. Plan for accuracy validation against a held-out set early — don't discover this at the end.

3. **`perform_splits=True` semantics.** This Gamera option lets the classifier split feature weights during training. There is no direct sklearn equivalent. Most likely the right answer is to drop it and rely on better feature engineering / standardization; flag this for your domain expert.

4. **Grouping function correctness is fiddly.** `ShapedGroupingFunction` builds a graph based on per-pixel proximity, not bounding-box overlap. Naive bounding-box-distance implementations will produce noticeably worse groupings for diacritics and adjacent characters. Allocate test fixtures specifically for this.

5. **`max_graph_size` parameter** in `cknn.group_list_automatic` exists because the grouping graph blows up on large pages. Your replacement needs the same kind of cap or it will hang.

6. **GameraXML XML schema is undocumented except by example.** Use the fixtures in `backend/django/code/test/files/` as your ground truth; write a parser test for each variant you encounter.

7. **Session size.** The Rodan `settings` dict holds *all glyphs* including base64-encoded images. For large pages this is megabytes per session. In FastAPI/Postgres, store the heavy image data on disk or object storage, keep only references in the DB.

8. **The state machine has implicit ordering.** Look at the order of operations inside each state in wrapper.py: `add_grouped_glyphs → update_changed_glyphs → remove_deleted_glyphs → remove_deleted_classes → update_renamed_classes → filter_parts`. Reproduce this ordering exactly — reordering causes subtle data loss (e.g., renaming a class after deleting it skips the delete).

9. **`gamera_xml_distributor.py` is dead weight.** It's a workflow-fanout helper specific to Rodan pipelines. Do not migrate it.

10. **Test fixtures live in the Rodan tree.** Copy `backend/django/code/test/files/Interactive_Classifier_*` into the new repo as part of Phase 1; they're your regression oracle.

---

## Verification end-to-end

After all three phases, you should be able to:

1. **Algorithm-only:** `pytest interactive_classifier_core/tests/` — all unit tests pass; class-assignment agreement with old Gamera code is ≥ 90% on the regression fixtures.
2. **API smoke test:** start FastAPI locally, `POST /sessions` with a real GameraXML, walk through CLASSIFY → manual corrections → COMPLETE, get a valid output XML.
3. **Manual UI test:** load the new React/Vue frontend in a browser, upload the same fixture, perform: auto-classify, manual reassignment, manual group, manual split, auto-group, save, complete. Compare visually against the old SPA running on the existing Rodan deployment.
4. **Regression against Rodan:** keep the old Rodan instance running in parallel for a few weeks; run the same input through both, diff the classified glyph output. Investigate any disagreement above the noise floor before retiring the old system.

---

## TL;DR migration order

1. Stand up `interactive_classifier_core/` as a pure-Python package. Port semantics from `interactive_classifier.py` and `intermediary/`. Unit-test everything. Validate accuracy vs. Gamera on fixtures.
2. Wrap it in a FastAPI service. Replace the Rodan state machine with explicit endpoints + DB-backed sessions.
3. Build a fresh React/Vite frontend using the old SPA as a UX spec.
4. Run the new and old systems side-by-side on real data until you're confident, then retire the Rodan job.
