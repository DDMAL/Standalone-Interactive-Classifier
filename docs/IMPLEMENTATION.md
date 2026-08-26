# Interactive Classifier — implementation reference

Status of this document: written against `main` @ `553c1e6` (2026-08-24), the
commit mothra pins as its `ic/` submodule. It describes **what the code does
now**, then the divergence between `main` and `develop`.

Read this in preference to `docs/CLAUDE.md` and `docs/migration_plan.md`,
which are the original rewrite-planning documents and have drifted (see
[Known documentation drift](#known-documentation-drift) at the end).

---

## 1. What this service is

A three-layer replacement for the legacy Rodan Interactive Classifier job
(Django + Celery + Gamera + Backbone.Marionette). Neumes on a manuscript page
are labelled by a k-nearest-neighbour classifier that is re-trained from the
user's own corrections after every round; the deliverable is GameraXML, which
downstream MEI encoders already consume.

| Layer | Path | What it is |
|---|---|---|
| Algorithm core | `core/ic_core/` | Pure-Python package (`uv`-managed). No Django, no Gamera, no scikit-learn. |
| HTTP API | `api/` | FastAPI service. Thin translation of HTTP → `ic_core.state.Session` methods. Also serves the built SPA in production. |
| Frontend | `frontend/` | React 18 + Vite + TypeScript + Tailwind SPA. |

Two deployment shapes, both from the same image:

- **Standalone** — the SPA's own upload screen; export downloads a GameraXML
  file to the browser.
- **Embedded in mothra** — mothra iframes the SPA as pipeline step 2, stages the
  page server-side, and drives export over its own server-to-server bridge.
  Everything embedding-specific is described in §6.

### Deliberate departures from the legacy IC

1. **Input is a page image + a bounding-box document**, not page-level
   GameraXML with pre-cropped glyph images. Upstream connected-component
   analysis is gone; `ic_core.ingest.ingest_page()` crops each glyph out of the
   page on the fly. GameraXML is **export-only** on the page path — it is still
   *read* for training sets and presets.
2. **kNN is hand-rolled numpy** (`core/ic_core/src/ic_core/classifier.py`), not
   Gamera's `kNNInteractive` and not scikit-learn.
3. **Features are a clean break from Gamera** — a fixed 29-dimensional set,
   explicitly versioned (`FEATURE_VERSION = "ic-core/v1"`). The exported
   `<features>` block keeps the legacy *element shape* so strict parsers accept
   the file, but the numbers are not interchangeable with Gamera's. Downstream
   consumers must gate on the `version` attribute.
4. **Splitting is manual only.** User-drawn rectangles slice a glyph; there is
   no CCA-based auto-splitter, and adding one is explicitly out of scope —
   touching strokes, ligatures and binarisation noise defeat it on real neumes.
5. **Grouping is manual only.** `auto_group_shaped` / `auto_group_bounding_box`
   are `NotImplementedError` stubs and `POST /sessions/{id}/auto-group` returns
   **501** (not 404, so the UI can say "not available yet"). The page-coordinate
   frame auto-grouping needs *does* now exist — what's missing is the design
   decision (adjacency function, graph-size gating).
6. **No mutation log.** The legacy job accumulated `@changed_glyphs`,
   `@grouped_glyphs`, `@deleted_glyphs` … in a settings dict because Rodan
   re-invoked the task with a fresh dict each round. Sessions here mutate
   directly, so the legacy ordering gotcha does not exist at the API surface.

---

## 2. Algorithm core (`core/ic_core/`)

### 2.1 `glyph.py` — the unit of work

`Glyph` is a frozen, slotted dataclass: stable 32-char hex `id`, `class_name`,
the binary mask as an RLE string (`image_rle`) plus `ncols`/`nrows`, the
page-coordinate origin `ulx`/`uly`, the `id_state_manual` flag, `confidence`,
a `category`, `is_training`, and an optional feature cache.

**`category` is the MOTHRA detector's coarse class**, carried end to end
(ingest → state → API → UI): `Text` / `Neumes` / `Staves`, from `classId`
1/2/3 in the MOTHRA JSON. **Only `Neumes` are classified.** Text and Staves
pass through untouched so the UI can group and hide them, and they are held out
of the training pool — otherwise a real neume can land nearest a "text" or
"staff" exemplar and inherit that label. Anything born outside the JSON path
(YOLO ingest, manual grouping) defaults to `Neumes`.

`id_state_manual` is the training boundary: `True` → the glyph trains the
classifier and is never overwritten by a round; `False` → it is a candidate for
auto-classification.

The feature cache (`feature_vector` / `feature_version`) is declared
`compare=False, repr=False`, so ndarray-valued fields don't break dataclass
equality/hashing or spam logs. `classify_manual` / `classify_automatic` use
`dataclasses.replace`, so the cache survives a label change — which is what
makes "full re-train every round" cheap for a stable pool. `to_dict()` is
hand-built rather than `asdict()` precisely so the ndarray isn't deep-copied
just to be thrown away.

### 2.2 `image.py` — RLE

Row-major, whitespace-separated run lengths, alternating white/black, starting
with a white run (a leading `0` is emitted when the first pixel is black).
`True` = black = foreground. `array_to_png_base64` produces the display-only
preview the API ships to the browser; the RLE is the authoritative form and is
what gets persisted.

### 2.3 `ingest.py` — page + bboxes → glyphs

Inputs are **raw bytes**, never filesystem paths — the API layer hands multipart
payloads straight through, so a client can never trick the server into opening a
server-side file.

Two annotation formats, chosen by an explicit `format` argument (no suffix
guessing):

- **`"json"` — MOTHRA JSON.** Pixel `bbox: [ulx, uly, w, h]`, a `classId`, and a
  stable per-annotation `id` that becomes the `Glyph.id`, so re-ingesting the
  same document yields the same glyph ids.
- **`"yolo"` — YOLO TXT.** `<class_id> <cx> <cy> <w> <h>`, normalised. Carries no
  ids, so every ingest mints fresh UUIDs; `class_id` is ignored and the glyph
  defaults to the `Neumes` category.

Every ingested glyph starts `UNCLASSIFIED`.

**Binarisation** happens at ingest and is baked into each mask. Three methods:

| `method` | What it does |
|---|---|
| `global` | Fixed cutoff: intensity ≤ 127 is ink. The default, kept so existing callers/tests are byte-for-byte unchanged. |
| `otsu` | One global cutoff derived from the page histogram. |
| `sauvola` | Local adaptive threshold (window 25, k 0.2, tuned by A/B on `core/data/test`). Recovers faint ink and staff lines on unevenly-lit parchment. |

Sauvola is memory-dangerous done naively: `threshold_sauvola` holds roughly six
page-sized float64 arrays at peak (~48 bytes/pixel — over 1.5 GB on a 32 MP
folio, which OOM-kills a 1 GiB container). `_sauvola_mask` therefore processes
horizontal strips sized to a ~96 MiB budget with a ~26-row halo, so peak memory
is a function of the budget rather than of the page.

### 2.4 `features.py` — 29 dimensions, versioned

Eight logical features, 29 float64 dimensions, computed with numpy + scipy.ndimage
+ skimage.measure:

`aspect_ratio`, `volume`, `nrows_feature`, `ncols_feature`, `compactness`,
`nholes` (1-d each), `volume16regions` (16-d), `hu_moment` (7-d).

`LOGICAL_FEATURES` is the single source of truth: the flat `FEATURE_NAMES`, the
vector layout, and the XML writer's one-`<feature>`-element-per-logical-feature
output are all derived from it. There is no Feature-Selection-XML equivalent —
the replacement for changing the feature set is bumping `FEATURE_VERSION` and
re-training. `get_features` reuses the per-glyph cache only when
`feature_version` matches the current `FEATURE_VERSION`, so a bump invalidates
every stale cache automatically.

### 2.5 `classifier.py` — the kNN

```
compute features → standardise per-dimension (zero mean / unit variance,
std clamped to 1e-12) → pairwise Euclidean → np.argpartition top-k
→ k=1: nearest wins; k>1: majority vote, ties broken by the closest neighbour
```

`filter_parts()` strips glyphs whose class name begins with `_group` or
`_delete` — transient user-intent markers — before every training round and
before every export. (The legacy `_split` prefix is deliberately absent; the
prefix is still *recognised* when loading legacy GameraXML.)

**Confidence is not Gamera-comparable.** It is `1 / (1 + distance)` against the
nearest neighbour in standardised feature space — monotonic in distance, so the
ordering the UI depends on is preserved, but absolute values must not be
compared against legacy XML.

`sort_by_confidence_ascending()` implements the review-queue invariant: the
least-certain glyph surfaces first.

**Note the `k` drift:** `DEFAULT_K` is **3**, while the constant's own comment,
the module docstring, and `docs/KNN_ALGORITHM.md` all still say k=1 is the
default and "must remain the default for parity". The code is what ships (the
toolbar also offers k as a user choice, and `ClassifyRequest.k` defaults to 3) —
the prose is stale. See §7.

### 2.6 `grouping.py` / `splitting.py`

`manual_group(glyph_ids, class_name)` bitwise-ORs the selected masks into one
canvas spanning the union of their bboxes, and — critically — fills the gaps
between the child bboxes from the session's retained **full-page mask**, since
ink falling between two tight bboxes was never copied into either crop. The
result is `id_state_manual=True, confidence=1.0` with a fresh UUID: it becomes
training data immediately rather than waiting for the next round.

`manual_split(glyph, regions)` slices the parent's mask along axis-aligned
rectangles given as `[ulx, uly, ncols, nrows]` in **page** coordinates (the same
frame as the bbox). Children are `UNCLASSIFIED`, `confidence=0`,
`id_state_manual=False`, each with a fresh UUID; the parent is dropped from the
working set. Children are *not* training data — the next classify round labels
them.

### 2.7 `state.py` — `Session` and the lifecycle

```
IMPORT ──ingest──▶ CLASSIFYING ──complete──▶ EXPORT
                     ▲     │
                     └─────┘  classify / update / group / split / delete /
                              rename / delete-class / rebinarize
```

`EXPORT` is terminal: mutations from it raise `StateTransitionError`, which the
API maps to **409**. Re-export from `EXPORT` is always allowed.

Session fields worth knowing:

| Field | Why it exists |
|---|---|
| `glyphs` | The working set, mixed manual/auto/unclassified, held in display order (ascending confidence for neumes, non-neumes trailing). |
| `training_glyphs` | The external pool: presets + uploaded GameraXML. One combined list, because that's what the classifier trains on. |
| `imported_class_names` | Autocomplete vocabulary; survives rounds even when no glyph currently uses a name. |
| `page_mask` | Full-page binarised mask. Needed by `manual_group` (gap fill) and by `rebinarize`. |
| `page_bytes` / `page_media_type` | The original upload, retained so `GET /sessions/{id}/page` can serve the page to a frontend that did not perform the upload — i.e. every embedded/deep-linked case. |
| `annotations_bytes` / `annotations_format` | Retained for completeness; **not** re-run by `rebinarize` (see below). |
| `binarization_method` | Which method produced the current masks; surfaced so the toolbar toggle reflects reality. |
| `preset_training_ids` / `uploaded_training_ids` | Provenance by glyph id, so the export screen can toggle the two training pools independently. Ids survive rename and export hygiene, so partitioning the live list by membership stays correct. |
| `source_name` | Stem of the uploaded bbox document; names the export artefact. |

`classify()` materialises the feature cache, splits `Neumes` from the rest,
filters the training pool to `Neumes` as well, runs one round, re-sorts by
ascending confidence, and puts non-neumes after.

**`rebinarize()` deserves attention** — it is the subtlest method in the core.
Every mask in a session is by construction a slice of the full-page mask at the
glyph's bbox: an ingested glyph literally *is* `page_mask[bbox]`; a split child
is its parent's mask restricted to a sub-rectangle of the parent's bbox, i.e.
the same pixels; a manual group is the OR of its members plus page-mask gap
fill. So changing method needs nothing but the new page mask — re-slice each
glyph's own bbox out of it. Ids, labels, manual flags, categories and order all
survive, **including nested derived glyphs** (a group of split children, a split
of a group).

An earlier implementation re-ran ingest and carried labels across by id. Split
children and grouped glyphs hold fresh ids no re-ingest can produce, so they
silently vanished and the pre-split parent came back. The re-ingest was also
pointless — the bboxes come from a document that never changes. Dropping it
fixed two more bugs for free: deleted glyphs no longer resurrect, and
YOLO-ingested sessions keep their labels (YOLO has no ids, so the old
carry-by-id matched nothing and a method switch wiped every label). Feature
caches are dropped since the pixels changed; auto labels are kept but are stale,
so callers normally chain a classify round — the toolbar does.

Clamping detail: the re-slice measures from `max(0, ulx)` rather than the
declared origin, because a stored mask for a bbox the detector rounded past the
page edge already starts there. Measuring from the declared (possibly negative)
origin would shave a few columns off such a glyph on *every* switch instead of
being idempotent.

`complete()` runs the export hygiene pass in place: strip transient
`_group`/`_delete` parts, drop `UNCLASSIFIED` training entries, transition to
`EXPORT`.

### 2.8 `io_xml.py` — GameraXML

Hand-written `lxml` reader and writer. The writer is the authoritative export
path and emits `<features version="ic-core/v1" scaling="1.0">` with one
`<feature name=...>` element per logical feature (single value for 1-d,
space-separated floats for the multi-dimensional ones), mirroring
`core/tests/fixtures/Square_notation-example_training_data.xml`. The reader
exists for round-trip tests **and** for the real training-set path (presets and
uploaded `.xml` files are parsed with `load_glyphs_bytes`).

The fixtures under `core/tests/fixtures/` are **writer-shape oracles**, not
ingestion samples.

---

## 3. HTTP API (`api/`)

FastAPI, `uv`-managed, depends on `ic-core` by relative path
(`api/pyproject.toml` `[tool.uv.sources]` → `../core/ic_core`), which is why any
build must preserve the sibling layout of `core/` and `api/`.

### 3.1 Session storage — two backends, selected by environment

`store.py` defines a narrow structural `SessionStore` protocol
(`create` / `get` / `session` / `delete` / `clear` / `lookup` / `list_sessions`).

- **`InMemorySessionStore`** — process-local dict, a registry lock guarding the
  dicts plus a lazily-created per-session `threading.Lock`. Sessions are lost on
  restart.
- **`PersistentSessionStore`** (`db_store.py`) — Postgres, table `ic_sessions`,
  a lazily-created `psycopg2.pool.ThreadedConnectionPool`, `connect_timeout=5`
  (without it libpq waits out the OS TCP timeout — minutes — and a single
  unreachable-DB request or `/healthz` probe hangs a worker thread).

`build_default_store()` picks Postgres when `IC_DATABASE_URL` or `DATABASE_URL`
(shared with the mothra backend) is set, otherwise in-memory. Both branches
announce themselves on stderr.

**Why every handler wraps its work in `with store.session(id) as s:`** — the
registry lock makes the dict thread-safe, but a retrieved `Session` is a plain
mutable object. Two requests on the same id (a double-click, concurrent UI
calls, a retry) would otherwise interleave their mutations. The context manager
holds the per-session lock, so each handler's read-mutate-serialise sequence is
atomic; different ids never block each other. A missing id raises `KeyError`,
which the exception handler maps to 404.

The persistent store hooks **the exit of that same context manager** to flush,
so every mutating endpoint auto-persists with zero endpoint changes. A pure
`GET` re-flushes too — a small idempotent `UPDATE`, acceptable for this
workload. Immutable blobs (page bytes, annotation bytes) are written once at
`create` and never rewritten. Glyphs serialise via their RLE string, not the
base64 PNG; the feature cache is dropped and lazily recomputed after a hydrate.

`(project_id, image_id)` is the resume key, enforced by a **partial** unique
index so the many key-less `(NULL, NULL)` sessions from IC's own upload screen
never collide.

Deliberately *not* verified at import: the DSN. `build_default_store()` runs at
module import, so a real connection attempt would block startup on the network —
and worse, "couldn't reach Postgres right now" must not silently downgrade the
process to in-memory for the rest of its life. A brief DB outage during startup
is routine; trading it for a permanent invisible loss of persistence in a
deployment that explicitly asked for persistence is far worse than loud
request-time errors that heal themselves.

### 3.2 `GET /healthz`

Returns `{"status": "ok", "store": {...}, "sessions": n}`.

`backend` / `persistent` report what the **environment asked for**. Holding a
Postgres store proves nothing on its own — it connects lazily, so a typo'd DSN
looks identical to a working one at construction time. `reachable` closes that
gap by round-tripping `SELECT 1`: `true` = sessions really are persisted,
`false` = the deployment believes it configured persistence but hasn't got it,
`null` = the in-memory store, which has nothing to reach.

`status` stays `"ok"` even when the database is unreachable, so wiring this up
as a liveness probe can't turn a DB hiccup into a restart loop — the diagnosis
belongs in the payload, not the status code.

### 3.3 Endpoint reference

Discovery / catalogue:

| Method | Path | Notes |
|---|---|---|
| `GET` | `/healthz` | See above. |
| `GET` | `/training-presets` | Filenames of built-in GameraXML training sets under `core/data/presets` (`IC_PRESETS_DIR` overrides). |
| `GET` | `/vocabularies` | Vocabulary CSV filenames under `core/data/train` (`IC_TRAIN_DIR` overrides) — only files that actually have a `classification` column. |
| `GET` | `/vocabularies/{name}/classes` | Distinct class names from that CSV, for preview. |

Path-traversal note: both catalogues only ever open a file the API itself
enumerated. A client-supplied name is validated against that listing before any
disk access, so `../secrets.xml` cannot escape the directory.

Session creation:

| Method | Path | Notes |
|---|---|---|
| `POST` | `/sessions` | `multipart/form-data`: `page_image`, `annotations`, `annotations_format`, optional `binarization_method`, `class_names`, `training_files[]`, `training_presets`, `vocabulary`. Returns the session in `CLASSIFYING`. |
| `POST` | `/staging` | Host stages a page + bboxes (+ optional `project_id`/`image_id`); returns a single-use `staging_id`. |
| `GET` | `/staging/{id}` | Staged metadata (page name, format, box count) for the upload screen. |
| `GET` | `/staging/{id}/page` | The staged page image, so the upload screen can preview it. |
| `POST` | `/sessions/from-staging` | Pairs a staged page with the user's training/vocabulary choices. Consumes the staging entry. |

Two multipart quirks are worked around in `main.py` and are load-bearing:

- Starlette 1.x caps each multipart part at 1 MB and FastAPI calls
  `request.form()` without `max_part_size`, so high-res scans would 413.
  `Request._get_form` is monkey-patched to raise the cap to 50 MB
  (`MAX_UPLOAD_BYTES` overrides).
- `class_names` and `training_presets` are **JSON-encoded strings**, not
  `list[str]`. FastAPI 0.136 treats any `list[X]` Form parameter sharing an
  endpoint with `UploadFile` as a JSON body, which then makes every multipart
  field look missing. For the same family of reasons these two endpoints use
  `Depends(get_store)` inline rather than the `Store` annotated alias — the
  alias preceding File/Form parameters mis-classifies the body.

Both creation paths funnel into `_finalize_session()`, which ingests, builds the
page mask **with the same method as the glyphs** (a mismatch would make the mask
disagree with them), records preset vs uploaded provenance, retains page and
annotation bytes, and — if any training data was supplied — **runs the first
classify round before responding**, so the client lands on an already-labelled
page.

Session lifecycle:

| Method | Path | Notes |
|---|---|---|
| `GET` | `/sessions` | Summaries, most-recent first; `?project_id=` scopes to one mothra project. Omits masks and page bytes. |
| `DELETE` | `/sessions` | Wipe every session; returns `{"deleted": n}`. No undo. |
| `GET` | `/sessions/lookup?project_id=&image_id=` | Resumable session id, or 404. Declared **before** `/sessions/{id}` so `lookup` isn't captured as a session id. An `EXPORT` session is treated as not resumable. |
| `GET` | `/sessions/{id}` | Full state. |
| `GET` | `/sessions/{id}/page` | Original page bytes. |
| `DELETE` | `/sessions/{id}` | Discard one. |
| `POST` | `/sessions/{id}/save` | Explicit checkpoint. With Postgres it flushes on context exit — the same write-through every mutation already does; in-memory it just echoes state. |
| `POST` | `/sessions/{id}/complete` | Export. See §3.4. |

Classification and editing:

| Method | Path | Notes |
|---|---|---|
| `POST` | `/sessions/{id}/classify` | Body `{"k": 3}`. Re-train, re-classify every non-manual neume. |
| `POST` | `/sessions/{id}/binarization` | Body `{"method": ...}`. Re-binarise + re-slice every mask. 400 for a session with no retained page. |
| `POST` | `/sessions/{id}/glyphs/{gid}` | Partial update: `class_name`, `id_state_manual`, `category`. |
| `DELETE` | `/sessions/{id}/glyphs/{gid}` | Drop from the working set. |
| `DELETE` | `/sessions/{id}/training-glyphs/{gid}` | Drop one glyph from the training pool. Returns the **session** (not 204) so the UI refreshes count + panel from one response. Does not re-classify. |
| `POST` | `/sessions/{id}/group` | Union N glyphs into one manual glyph. |
| `POST` | `/sessions/{id}/glyphs/{gid}/split` | Body `{"regions": [[ulx, uly, ncols, nrows], …]}`. Returns the children. |
| `POST` | `/sessions/{id}/auto-group` | **501**, deferred. Touches the session lookup first so an unknown id still 404s rather than 501s. |
| `POST` | `/sessions/{id}/classes/{name}/rename` | Rename across working set, training set and autocomplete. |
| `DELETE` | `/sessions/{id}/classes/{name}` | Drop a class and its dotted-namespace subclasses from autocomplete. |

Error model: every non-2xx is `{"detail": "...", "code": "..."}` with `code`
drawn from a finite enum, so the frontend dispatches on the code rather than
parsing prose. `StateTransitionError` → 409 `state_conflict`; `KeyError` → 404
`not_found`; `ValueError` → 400.

### 3.4 Export — `POST /sessions/{id}/complete`

Four independent boolean flags choose what is folded into one GameraXML
document, concatenated in this order and de-duplicated by glyph id (so
selecting both `page` and `manual_neumes` never emits a glyph twice). At least
one must be set, else 400.

- `page` — every working glyph on the annotated page
- `manual_neumes` — only the working neumes the user labelled by hand
- `preset_training` — training glyphs from a built-in preset
- `uploaded_training` — training glyphs the user uploaded

The response is `application/xml` with a `Content-Disposition` filename tagged
by the chosen sections (`ic-session-<stem>-page-manual-neumes.xml`), the stem
being the sanitised source name or, failing that, the session id.

**`finalize` (default `true`) is the important parameter.** With `finalize=true`
the first call transitions `CLASSIFYING → EXPORT` (idempotent; re-export always
works, but mutations then 409). With **`finalize=false`** the same hygiene pass
— strip transient `_group`/`_delete` parts, drop `UNCLASSIFIED` training entries
— is applied to the *exported copy only*, leaving the live session in
`CLASSIFYING`, editable and resumable.

That flag exists because an embedding host treats the XML as an *intermediate*
artefact: mothra feeds the export to its MEI encoder and still expects the user
to reopen and correct the page. Since `lookup` refuses to resume an `EXPORT`
session, finalising on export would silently strand every correction behind it —
re-entering the page would hand back a blank new session. This is exactly the
bug the flag was added to fix, and mothra now refuses to start against an IC
whose OpenAPI schema lacks it (see §6).

### 3.5 Serving the SPA

After every API route, `main.py` mounts `api/src/ic_api/static/` at `/` with
`html=True`, if the directory exists. Order matters: `/sessions` and friends
still resolve to their handlers, everything else falls through to `index.html`
so client-side routing works. In local dev the directory is absent and the mount
is simply skipped. `scripts/build-frontend.sh` reproduces what the Dockerfile
does for a production-style run without Docker.

---

## 4. Frontend (`frontend/`)

React 18, Vite 5, TypeScript, Tailwind 3, TanStack Query (server state),
Zustand (`store/uiStore.ts`, UI state), TanStack Virtual (the glyph grid),
Radix primitives, Biome for lint/format.

`App.tsx` is the whole router — no react-router. It branches on query
parameters and on whether a session is open:

| Query param | Behaviour |
|---|---|
| *(none)* | `UploadView` — IC's own create-session screen. |
| `?session=<id>` | Open straight into `SessionView`. Also posts `ic:session-created` upward, because resuming skips `useCreateSession`'s notify and the host would otherwise wait forever before enabling its encode action. |
| `?staged=<id>` | `UploadView` with the page + bboxes pre-filled and locked; the user only picks training data and vocabulary. |
| `?manage=1&project_id=<id>` | Full-width `SessionResumeList` scoped to that project. |

`SessionView` is a four-pane layout under a toolbar: `ClassTreePanel` (left
rail), `PageImagePane` (zoom/pan page with the bbox overlay and lasso),
`GlyphGrid` (virtualised tiles, ascending confidence), `RightDock`
(`EditPanel` / `MultiEditPanel` / `TrainingDataPanel`).

Notable UI behaviour, all of it encoded in the store or hooks:

- **Soft delete.** `deletedGlyphIds` hides glyphs from grid, overlay and lasso
  but keeps them restorable; the actual `DELETE`s are committed at export time
  (`useComplete` fires them in parallel before the export request).
- **Modal guard.** `modalOpenCount` is a *counter*, not a boolean, so
  overlapping open/close transitions can't desync it. While > 0, window-level
  shortcuts (Enter-to-classify, type-to-focus, zoom/pan, Esc) stand down.
- **Hotkeys.** `+`/`-`/`0` zoom, arrows pan, `Esc` clears selection, `z` undoes
  an apply, hold `h` to hide all bboxes. A bare alphanumeric key is
  "type-to-focus" for the class-name field when a glyph is selected — `h` is
  excluded from that set so the hide hotkey can never seed the field.
- **Undo** covers `class_name` applies only (max 5 entries), not split, group or
  rebinarize.
- **Glyph view toggle** (`glyphImageMode`): binarised mask or the original page
  crop. It is a display preference that persists across sessions, and it is
  honoured by the grid tiles (a live canvas per tile), the edit panel, and the
  split canvas — the last two go through `useGlyphImageSrc`, which crops the
  region onto an offscreen canvas and exports a data URI. That export costs a
  real encode, so it is only for the one-glyph cases; the grid keeps drawing
  straight onto a live canvas.
- **`knnK`** is a user-selectable toolbar preference that also persists across
  sessions.
- **Training-data panel and edit panel share one dock slot**, so they are
  mutually exclusive by construction: expanding one clears the other's input.
- **Presets are mutually exclusive** — checking one unchecks the rest.

API access goes through `api/client.ts`. In dev `BASE` is `""` and Vite proxies
to `127.0.0.1:8000`; in a production build it falls back to
`VITE_API_BASE ?? ""`, which is empty for the single-origin image — same-origin
requests, no CORS, nothing to configure.

---

## 5. Build, run, test

```bash
# core
cd core/ic_core && uv sync && uv run pytest ../tests     # tests live in core/tests/
uv run python ../scripts/run_pipeline.py                  # train → classify → overlays
uv run ruff check .

# api  (start before the frontend; binds 127.0.0.1:8000, HOST/PORT override)
cd api && uv sync && uv run ic-api && uv run pytest

# frontend
cd frontend && npm install && npm run dev                 # :5173
npm run check      # biome
npm run build      # tsc --noEmit && vite build
```

`core/scripts/paths.py` centralises every data path and each default is
overridable via an `IC_*` env var (`IC_TRAINING_XML`, `IC_TEST_PAGE`, …), so
scripts and tests can run against alternate inputs with no code edits.
`core/data/derived/` is gitignored and regenerable; `test_real_input_knn.py`
self-bootstraps its training XML into a temp dir when the default path is
absent, so a clean checkout passes `pytest` without running
`convert_hufnagel_csv.py` by hand.

**Docker (single origin).** Two stages: node builds the SPA, then a
`uv:python3.12-bookworm-slim` stage copies `core/` and `api/`, runs
`uv sync --frozen --no-dev`, and drops the built SPA into
`api/src/ic_api/static/`. Build context is the `ic/` directory, and it must
contain both `core/` and `api/` with their relative layout intact.

**CI** (`.github/workflows/run-tests.yml`) runs the `core` and `api` pytest
suites on pull requests targeting `main`/`master`. There is no frontend job.

Environment variables the service reads:

| Variable | Effect |
|---|---|
| `HOST` / `PORT` | uvicorn bind (default `127.0.0.1:8000`; the image sets `0.0.0.0:8000`). |
| `IC_DATABASE_URL` / `DATABASE_URL` | Selects the Postgres session store. |
| `IC_PRESETS_DIR` | Where `/training-presets` looks. |
| `IC_TRAIN_DIR` | Where `/vocabularies` looks. |
| `MAX_UPLOAD_BYTES` | Multipart part-size cap (default 50 MB). |

---

## 6. Embedding contract (mothra)

mothra iframes this SPA as pipeline step 2. The two sides talk over `postMessage`
plus a server-to-server REST bridge (`landing-page/scripts/ic_api.py`).

Server-to-server, mothra → IC:

1. `GET /sessions/lookup?project_id=&image_id=` — is there a session to resume?
2. `POST /staging` with the page image, the bboxes, and the `project_id` /
   `image_id`, then deep-link the iframe to `/?staged=<id>`.
3. `POST /sessions/{id}/complete?page=true&finalize=false` — take the GameraXML
   snapshot without retiring the session.
4. `GET /sessions?project_id=` — back mothra's own per-project "saved sessions"
   picker.
5. In mothra's *auto* mode, `POST /sessions` directly (not `/staging`) plus one
   classify round; those sessions **do** finalise, deliberately — `/sessions`
   takes no project/image id, so IC could never map them back to a page for
   resume, and leaving them in `CLASSIFYING` would only accumulate unreachable
   sessions.

`postMessage`, iframe → host:

| Message | When |
|---|---|
| `ic:ready` | `UploadView` mounted and listening. |
| `ic:session-created` | A session exists (created, or deep-link-resumed). |
| `ic:auto-export` | The one-click classify-then-export shortcut ran; the host should run its own queue path. |
| `ic:resume-session` | A row was clicked in the `?manage=1` list. |

`postMessage`, host → iframe:

| Message | Effect |
|---|---|
| `ic:prefill-training` | Adopt the host's batch-level training selection. |

Three behaviours flip on `window.parent !== window`, and each one matters:

- **Export sends `finalize=false`.** Standalone, the download really is the end
  of the session's life, so the default finalise is right. Embedded, it would
  strand every correction (§3.4).
- **The one-click shortcut doesn't download.** `useAutoExport` posts
  `ic:session-created` then `ic:auto-export` and returns, so the host drives
  completion through its own server-side bridge — the GameraXML stays
  server-side and the page lands in the host's encode queue exactly like a
  manually-classified one, with no stray browser download. The button is worded
  "Queue page" instead of "Auto-export".
- **`?manage=1` clicks hand off rather than navigate.** Swapping the list for a
  `SessionView` *inside the host's modal* would strand the user in a pane with
  no filmstrip, clef controls or encode queue. So the click posts
  `ic:resume-session` and mothra navigates; it re-opens the same session, since
  sessions are unique per `(project_id, image_id)`. Standalone there is no host
  to hand off to, so it resumes in place.

`ic:prefill-training` data is treated as untrusted input from another window:
`normalizePresets` drops non-strings and truncates to the one-preset invariant,
warning to the console when it truncates — a silent truncation would look like
the selection was simply lost.

---

## 7. Known documentation drift

Recorded so the next reader doesn't trust the wrong file. None of these are
code bugs.

| Where | Says | Actually |
|---|---|---|
| `classifier.py:77-79`, its module docstring, `docs/KNN_ALGORITHM.md` | k=1 is the default and "must remain the default for parity" | `DEFAULT_K = 3`; `ClassifyRequest.k` defaults to 3; the toolbar exposes k as a user choice |
| `docs/CLAUDE.md` | "`frontend/` is currently empty" / "not yet started"; store is "in-memory only" | The SPA is complete; Postgres persistence exists (`db_store.py`) |
| `api/src/ic_api/main.py` module docstring, "What's deliberately missing in v1" | "The default store is in-memory only; swap `ic_api.store` for a SQLite-backed implementation" | Already done, as Postgres, selected by `DATABASE_URL` |
| `README.md` status table | API is "FastAPI + in-memory store" | Backend is environment-selected |
| `store.py` module docstring | "This is **the** Phase-2 storage layer" | It is now one of two |

Two small live gaps, also not bugs but worth knowing:

- `frontend/vite.config.ts`'s dev proxy covers `/sessions`, `/training-sets`,
  `/training-presets`, `/vocabularies` — but **not `/staging`**. So the
  `?staged=<id>` flow does not work against the Vite dev server on :5173; it
  works in the single-origin build (`:8000`), which is how mothra actually
  runs IC. Add `/staging` to the proxy to exercise that flow in dev.
- `/training-sets` in that same proxy list is vestigial — no such endpoint
  exists in `main.py`.
