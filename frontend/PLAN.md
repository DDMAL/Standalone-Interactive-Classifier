# Frontend — `ic_new/frontend/`

This document describes the frontend **as built**. It began as a phased plan
(Phase A MVP → B page overlay/multi-select → C multi-edit/class management) and
has since grown past all three; the sections below reflect the current
implementation rather than the original roadmap. A short [phase history](#how-it-grew)
is kept at the end for context.

## Context

`ic_new` rewrites the legacy Rodan Interactive Classifier (Backbone.Marionette
+ Django + Gamera) as a non-Django Python service (`core/ic_core/` algorithm
core + `api/` FastAPI surface) plus this React UI. The UI lets a user upload a
manuscript page + a detection (bbox) file, review glyphs the classifier is least
sure about, correct labels one at a time or in bulk, split/group/delete glyphs,
manage the class vocabulary, and export GameraXML.

### Stack

- **React 18 + Vite 5 + TypeScript 5**, path alias `@ → src`.
- **Tailwind 3** for styling (`clsx` for conditional classes); **Radix**
  headless primitives — `@radix-ui/react-dialog` (Split / Group / batch-confirm /
  confirm dialogs) and `@radix-ui/react-popover` (the class-name autocomplete).
- **TanStack Query 5** for server state, **Zustand 5** for UI state.
- **Biome** for lint + format (`npm run check` / `npm run format`).
- The page image is held in browser memory as an object URL — never re-uploaded.
- `@tanstack/react-virtual`, `@radix-ui/react-label`, and `@radix-ui/react-slot`
  are still listed in `package.json` but are **not currently imported** — the
  glyph grid renders plain DOM tiles (see [GlyphGrid](#glyph-grid--the-working-set)),
  not a virtualized list.

### Dev wiring

- `vite.config.ts` proxies `/sessions`, `/training-sets`, `/training-presets`,
  and `/vocabularies` to `http://127.0.0.1:8000`, so app code calls
  `fetch("/sessions/...")` with no base prefix. `VITE_API_BASE` is a fallback
  for non-proxied prod builds ([api/client.ts](src/api/client.ts)).
- The API adds `CORSMiddleware` for `http://localhost:5173` /
  `http://127.0.0.1:5173` ([api/src/ic_api/main.py](../api/src/ic_api/main.py#L117-L122))
  so ad-hoc cross-origin fetch/`curl` works too.
- `main.tsx` mounts a `QueryClient` with `retry: false` and
  `refetchOnWindowFocus: false`.

## Two screens

[App.tsx](src/App.tsx) renders exactly one of two screens based on
`uiStore.sessionId`:

- **UploadView** — before a session exists.
- **SessionView** — the classification workspace.

Two deep-link params let an embedding host (mothra) jump in:

- `?session=<id>` — open straight into an existing session. The page image is
  served by the API at `/sessions/<id>/page` (this frontend never uploaded it).
- `?staged=<id>` — open the upload screen with the page + bboxes already staged;
  the user only adds training data + a vocabulary, then starts the session
  (created via `POST /sessions/from-staging`).

When embedded in an iframe, `useCreateSession` posts
`{ type: "ic:session-created", sessionId }` to `window.parent` so the host can
drive completion ([hooks/useCreateSession.ts](src/hooks/useCreateSession.ts)).

## File layout

```
frontend/
├── index.html, package.json, tsconfig.json, vite.config.ts
├── tailwind.config.ts, postcss.config.js, biome.json
├── .env.development                # VITE_API_BASE fallback
└── src/
    ├── main.tsx                    # React root + QueryClientProvider
    ├── App.tsx                     # UploadView | SessionView + deep-link parsing
    ├── index.css
    ├── api/
    │   ├── client.ts               # fetch wrapper → ApiError{code,detail,status}; postForBlob for export
    │   └── sessions.ts             # one function per endpoint
    ├── types/
    │   └── api.ts                  # SessionDTO, GlyphDTO, enums, ApiError
    ├── store/
    │   └── uiStore.ts              # Zustand: selection set, deletes, modal count, prefs, undo
    ├── hooks/                      # one hook per mutation/query + interaction hooks
    ├── lib/                        # format, keymap, classTree, bbox, tileRefs, download
    └── components/
        ├── UploadView.tsx
        ├── SessionView.tsx         # layout shell + global keyboard
        ├── Toolbar.tsx             # binarize, glyph view, k/reclassify, undo, save, export menu
        ├── ClassTreePanel.tsx / ClassTreeNode.tsx   # left rail
        ├── PageImagePane.tsx / PageOverlay.tsx / BBoxLayer.tsx / LassoLayer.tsx / ZoomPanContainer.tsx
        ├── GlyphGrid.tsx / ClassSection.tsx / GlyphTile.tsx / GlyphImage.tsx / DeletedSection.tsx
        ├── EditPanel.tsx / MultiEditPanel.tsx / ClassNameInput.tsx
        ├── SplitDialog.tsx / GroupDialog.tsx / BatchConfirmDialog.tsx / ConfirmDialog.tsx
        └── ui/                     # Button, icons
```

## Data model — [types/api.ts](src/types/api.ts)

```ts
type ClassifierState    = "import" | "classifying" | "export";
type AnnotationFormat   = "json" | "yolo";
type BinarizationMethod = "global" | "otsu" | "sauvola";
type GlyphCategory      = "Text" | "Neumes" | "Staves";   // only Neumes get classified

interface GlyphDTO {
  id: string; class_name: string; confidence: number;
  id_state_manual: boolean; category: GlyphCategory;
  ulx: number; uly: number; ncols: number; nrows: number;
  image_b64: string;                                        // binarized glyph mask PNG
}

interface SessionDTO {
  id: string; state: ClassifierState;
  glyphs: GlyphDTO[]; training_glyphs: GlyphDTO[]; class_names: string[];
  binarization_method: BinarizationMethod;
  preset_training_count: number;                            // training glyphs from a built-in preset
  uploaded_training_count: number;                          // training glyphs from an uploaded file
}
```

`CATEGORY_ORDER` = `["Neumes", "Text", "Staves"]` and only Neumes is open by
default (`CATEGORY_DEFAULT_OPEN`). `ApiError` carries a structured
`code` (`not_found | state_conflict | validation_error | deferred |
internal_error | unknown`) and HTTP `status`.

## API client — [api/sessions.ts](src/api/sessions.ts)

| Function | Call |
| --- | --- |
| `createSession` | `POST /sessions` (multipart: page image, annotations, format, optional class names, training files, `training_presets`, `vocabulary`) |
| `getStaging` / `createSessionFromStaging` | `GET /staging/{id}` · `POST /sessions/from-staging` |
| `listTrainingPresets` / `listVocabularies` / `getVocabularyClasses` | `GET /training-presets` · `GET /vocabularies` · `GET /vocabularies/{name}/classes` |
| `getSession` / `deleteSession` | `GET` · `DELETE /sessions/{id}` |
| `classify` | `POST /sessions/{id}/classify` `{k}` |
| `rebinarize` | `POST /sessions/{id}/binarization` `{method}` |
| `updateGlyph` | `POST /sessions/{id}/glyphs/{gid}` `{class_name?, id_state_manual?, category?}` |
| `deleteGlyph` | `DELETE /sessions/{id}/glyphs/{gid}` |
| `manualGroup` | `POST /sessions/{id}/group` `{glyph_ids, class_name}` |
| `splitGlyph` | `POST /sessions/{id}/glyphs/{gid}/split` `{regions: [ulx,uly,ncols,nrows][]}` |
| `renameClass` / `deleteClass` | `POST .../classes/{name}/rename` · `DELETE .../classes/{name}` |
| `saveSession` | `POST /sessions/{id}/save` |
| `completeSession` | `POST /sessions/{id}/complete?page&manual_neumes&preset_training&uploaded_training` → XML blob |

## Screens & components

### Upload view — [UploadView.tsx](src/components/UploadView.tsx)

The entry form.

- **Page + annotations** — a page image and a detection file, with a format
  select (`MOTHRA JSON` / `YOLO TXT`). In the staged (mothra) flow these inputs
  are replaced by a read-only summary of the staged page + detection count, and
  the session is created from staging.
- **Training data (optional)** — two combinable sources: built-in **presets**
  (checkboxes, from `listTrainingPresets`) and **uploaded** GameraXML `.xml`
  files. A running total tells the user how many training sets will be combined
  and classified on start.
- **Vocabulary (optional)** — a select from `listVocabularies` that seeds the
  available class names; picking one shows a read-only preview of its classes.
- Submit stashes the page File as an object URL (or points at `/sessions/{id}/page`
  in the staged flow) and flips into the session view.

### Session view — [SessionView.tsx](src/components/SessionView.tsx)

The layout shell: a `Toolbar` over a four-column row —
**ClassTreePanel · PageImagePane · GlyphGrid · EditPanel**. The `EditPanel` only
mounts when ≥1 glyph is selected (so the workspace is three columns when nothing
is selected). It wraps everything in `PageImageProvider` (see
[usePageImage](src/hooks/usePageImage.tsx)) so the original-crop renderer can
reach the loaded `<img>`.

SessionView also owns the **global keyboard model** via window listeners:

| Key | Action |
| --- | --- |
| `+` / `=` · `-` / `_` · `0` | Zoom in / out (container center) · reset |
| Arrow keys | Pan the page |
| `Esc` | Clear selection |
| `Cmd/Ctrl+Z` | Undo the last label apply |
| hold `h` | Temporarily hide all page bboxes (press-and-hold; blur-safe) |

All of these stand down while a modal is open (`isModalOpen()`), while a text
input is focused (`isEditableTarget`), or — for zoom/pan number keys — while a
glyph is selected and the edit panel claims type-to-focus
([lib/keymap.ts](src/lib/keymap.ts)).

### Toolbar — [Toolbar.tsx](src/components/Toolbar.tsx)

Glyph count on the left; classifier controls + save/export on the right.

- **Binarize** (Global / Otsu / Sauvola) → `rebinarize`. Rebuilds every glyph
  mask; manual groups/splits reset, labels are kept. Re-run classify to refresh
  auto labels.
- **Glyphs** (Binarized / Original) → `glyphImageMode`. Display-only toggle for
  what tiles/previews show; doesn't touch underlying data.
- **k** (1 / 3 / 5 / 7) + **↺ Reclassify** → `classify(k)`. A `k` is disabled
  when the training pool has fewer than `k` glyphs, and the active `k` auto-falls
  back if the pool shrinks. Shows the training-glyph count
  (`trainingPoolSize`, which counts in-session manual corrections **plus**
  external training glyphs) and a ⚠ tooltip under 10 training glyphs.
- **Undo** → `useUndoApply` (up to 5 deep; restores label snapshots but does
  **not** auto-reclassify; does not cover split/group/rebinarize).
- **New session** → confirm dialog, then `clearSession()`.
- **Save** → `saveSession`.
- **Complete & Export ▾** → a checkbox menu (`ExportMenu`) choosing which
  sections to fold into one GameraXML: whole page, manual neumes, uploaded
  training, preset training. Empty sections are disabled and force-unchecked; at
  least one must be selected. Export also commits any pending soft-deletes.

### Class tree panel — [ClassTreePanel.tsx](src/components/ClassTreePanel.tsx) · [ClassTreeNode.tsx](src/components/ClassTreeNode.tsx)

Left rail (200px expanded / 24px collapsed, state in `uiStore.classTreeCollapsed`).
Parses `session.class_names` into a `.`-separated tree
([lib/classTree.ts](src/lib/classTree.ts)); each node shows a working-set count
badge and, on hover, an action row.

- **Click a class name** → applies that class to all currently-selected Neumes
  and reclassifies (the bulk counterpart to the EditPanel apply; pushes an undo
  entry). No-op — reads as a hint — when nothing applicable is selected.
- **Select** → `useClassSelection`: selects every glyph on that class, or the
  whole subtree for interior nodes.
- **Rename** → inline edit; `renameClass` on the full dotted path.
- **Delete** → confirm dialog; `deleteClass`, resetting affected glyphs
  (including soft-deleted carriers, so the backend's union-derived `class_names`
  can't re-derive the class) to `UNCLASSIFIED`. Skips the confirm when no
  *present* glyph carries the class.

### Page image pane — [PageImagePane.tsx](src/components/PageImagePane.tsx)

The manuscript page (left ~third) inside a `ZoomPanContainer`, with an
absolutely-positioned SVG overlay sibling of the `<img>`.

- **[PageOverlay](src/components/PageOverlay.tsx)** uses the image's natural
  size as its `viewBox`, so bbox coordinates draw verbatim — no per-rect scaling.
  It filters out soft-deleted glyphs and hides everything while `bboxesHidden`
  (hold-`h`).
- **[BBoxLayer](src/components/BBoxLayer.tsx)** draws one rect per glyph with
  `vectorEffect="non-scaling-stroke"` (1px strokes at any zoom). **Neumes** are
  interactive (hover/click/`Shift`·`Cmd`-click toggle); **Text/Staves** are
  non-interactive decor that render **only when selected or hovered from the
  grid**, so the page isn't cluttered with unclassified outlines. Manual glyphs
  are green; auto glyphs go slate → amber (hover) → blue (selected), matching the
  tiles.
- **[LassoLayer](src/components/LassoLayer.tsx)** + **[useLasso](src/hooks/useLasso.ts)**
  — drag on empty space draws a marquee (rAF-throttled); on release it hit-tests
  Neumes only and commits (`extendSelection` with `Shift`/`Cmd`, else
  `setSelection`). A zero-motion background click clears the selection.
- **[useZoomPan](src/hooks/useZoomPan.ts)** — scale clamped to `[0.25, 8]`,
  step 1.2. Trackpad pinch (ctrl/meta-wheel) zooms at the cursor; plain wheel
  pans. Clicking a grid tile re-centers the page on that glyph when zoomed in
  (via `pendingFocusGlyphId` → `centerOnImagePoint`).

### Glyph grid — the working set — [GlyphGrid.tsx](src/components/GlyphGrid.tsx)

All glyphs as thumbnail tiles, partitioned in one pass into MOTHRA-category
sections plus a deleted bucket.

- **[ClassSection](src/components/ClassSection.tsx)** — collapsible **Neumes /
  Text / Staves** sections (responsive CSS grid, ~88px tiles; **not** virtualized).
  Only **Neumes** carries a sort dropdown (`conf-asc` default, plus `conf-desc`,
  `name-asc`, `name-desc` — [lib/format.ts](src/lib/format.ts)).
- **[GlyphTile](src/components/GlyphTile.tsx)** — glyph image
  ([GlyphImage](src/components/GlyphImage.tsx) honors the binarized/original
  toggle), class name, confidence %, and an **M/A** badge. Manual tiles are
  green; selected are ringed. Click selects + re-centers the page
  (`focusGlyph`); `Shift`/`Cmd`-click toggles. Registers itself in
  [lib/tileRefs.ts](src/lib/tileRefs.ts) so `useSelectionSync` can scroll the
  primary tile into view. Hover is two-way linked with the page overlay.
- **[DeletedSection](src/components/DeletedSection.tsx)** — an amber recycle bin
  under Staves holding soft-deleted glyphs, each with a **Put back** button.
  Deletes are UI-only (`deletedGlyphIds`) until **Complete & Export** actually
  `DELETE`s them ([useComplete](src/hooks/useComplete.ts)).

### Edit panel — [EditPanel.tsx](src/components/EditPanel.tsx)

Right rail; branches on selection size (keyed so local state resets on every
transition): **1 → SingleEditor**, **≥2 → MultiEditPanel**.

**SingleEditor** (1 glyph):
- Enlarged preview + metadata (category, confidence, source, position, size).
- For Neumes: a **ClassNameInput** autocomplete + **Apply & reclassify**
  (`updateGlyph` → `classify` → invalidate → push undo). Type any letter
  anywhere to seed the field; **Enter** from anywhere applies (both via window
  listeners gated by `isEditableTarget`); ↑/↓ walk suggestions.
- **Move to class** — reassign MOTHRA category (backend resets the label).
- **Split glyph…** — opens the split dialog.
- **Delete glyph** — soft-delete.

**MultiEditPanel** — [MultiEditPanel.tsx](src/components/MultiEditPanel.tsx) (≥2):
- Header: "N selected · K Neumes, M non-Neumes". Non-Neumes are filtered out of
  applies (with a "Skipping M" hint).
- **Apply to K Neumes** — one class to all (`useUpdateGlyphs`, returns
  `{applied, failed}` for a "K of N applied" status). Seeded with the dominant
  class in the selection.
- **Apply each in own class…** — opens `BatchConfirmDialog`.
- **Move to class** (whole batch, no reclassify), **Group as new glyph**
  (`Cmd/Ctrl+G`), **Delete N glyphs**. `Cmd/Ctrl+E` focuses the class input.

**[ClassNameInput](src/components/ClassNameInput.tsx)** — shared combobox
(Radix Popover anchor): filters the class list (max 50), supports ↑/↓ + Enter to
apply the highlight, `onMouseDown` preventDefault to keep focus, and free text.

### Dialogs

While any is open, `useModalGuard` increments `modalOpenCount` so page-level
shortcuts stand down ([hooks/useModalGuard.ts](src/hooks/useModalGuard.ts)).

- **[SplitDialog](src/components/SplitDialog.tsx)** — draw axis-aligned
  rectangles over a glyph's image (SVG with a drawable margin, `image-rendering:
  pixelated`, per-rect numbered badge + delete). Each rect becomes one new
  **UNCLASSIFIED** child (confidence 0 → surfaces at the top of the queue).
  Coordinates are glyph-local, translated to page coords on submit; degenerate
  rects are snapped/clipped away. Offers **Split** and **Split & Reclassify**;
  the CCA auto-splitter was rejected in favor of this manual flow.
- **[GroupDialog](src/components/GroupDialog.tsx)** — merge selected Neumes into
  one new manual glyph, with a **live before/after preview** (canvas
  reconstruction mirroring the backend's union-of-bboxes group, honoring the
  binarized/original toggle).
- **[BatchConfirmDialog](src/components/BatchConfirmDialog.tsx)** — the "apply
  each in own class" mosaic: every selected Neume shown grouped by its predicted
  class with an editable label, committed per-glyph (`useUpdateGlyphsPerGlyph`)
  then classify. UNCLASSIFIED entries are skipped.
- **[ConfirmDialog](src/components/ConfirmDialog.tsx)** — small reusable yes/no
  for destructive actions (new session, delete class).

## State model

### UI state — [store/uiStore.ts](src/store/uiStore.ts)

Set-first selection plus a "primary" id for framing/scroll.

```ts
selectedGlyphIds: Set<string>;  primaryGlyphId: string | null;  hoverGlyphId: string | null;
deletedGlyphIds: Set<string>;                 // soft delete; committed at export
pendingFocusGlyphId: string | null;           // tile click → re-center page; consumed by PageImagePane
bboxesHidden: boolean;                        // hold-h
modalOpenCount: number;                       // openModal/closeModal; isModalOpen()
classTreeCollapsed: boolean;
knnK: number;                                 // default 3 — persists across sessions (a preference)
glyphImageMode: "binarized" | "original";     // default binarized — persists
undoStack: UndoEntry[];                       // max 5; label snapshots only
// actions: selectGlyph/toggleGlyph/setSelection/extendSelection/clearSelection,
//          focusGlyph/consumeFocus, softDeleteGlyphs/restoreGlyph/clearDeleted,
//          pushUndo/popUndo, setSession/clearSession (revoke object URL + reset transient state)
```

`setSession`/`clearSession` reset selection, hover, deletes, focus, the class-tree
collapse, and the undo stack; `knnK` and `glyphImageMode` deliberately persist as
preferences.

### Server state — TanStack Query

- Single query key `['session', id]` ([hooks/useSession.ts](src/hooks/useSession.ts),
  `staleTime: 0`).
- Mutations live one-per-file under `hooks/` and `invalidateQueries(['session',
  id])` on success so the tree, page, and grid all redraw from one refetch.
  `useCreateSession(FromStaging)` writes the returned DTO straight into the cache
  with `setQueryData` and skips the refetch.
- Bulk operations fan out `Promise.allSettled` and report `{applied, failed}` so
  a partial failure surfaces "K of N applied" rather than aborting the batch
  ([hooks/useUpdateGlyphs.ts](src/hooks/useUpdateGlyphs.ts)).

## Deferred / not built

- **Auto-grouping** — backend `POST /auto-group` returns 501; no UI surface.
  The `ApiError{code:"deferred"}` shape already renders a meaningful message if a
  future placeholder ever hits it.
- **Auto-splitting** — rejected (touching strokes/ligatures/noise defeat
  connected-components); the manual `SplitDialog` covers both easy and hard cases.
- **Grid virtualization** — the CSS-grid sections render fine at the expected
  few-hundred-glyphs-per-page scale; `@tanstack/react-virtual` remains a
  dependency for if/when profiling justifies it.

## Verification

End-to-end manual smoke test (all steps pass = the app is healthy):

1. `cd api && uv sync && uv run ic-api` (→ `http://127.0.0.1:8000`); `cd
   frontend && npm install && npm run dev` (→ `http://localhost:5173`).
2. Upload a page + JSON/YOLO annotations from `core/data/train/` or
   `core/data/test/`, optionally pick a preset + vocabulary, start the session.
3. SessionView renders: class tree, page with bbox overlay, glyph grid (Neumes
   sorted lowest-confidence-first), no edit panel until a selection exists.
4. Click the lowest-confidence Neume, type a class, Enter → tile turns green with
   an **M** badge, classify runs, grid resorts.
5. Lasso several Neumes → MultiEditPanel; apply one class, or "apply each in own
   class", or Group; verify a mixed selection skips non-Neumes.
6. Split a glyph into pieces; the children appear UNCLASSIFIED at the top.
7. Delete a glyph → Deleted section; Put back restores it.
8. Toggle Binarize methods and the Binarized/Original glyph view; change `k` and
   Reclassify; Undo a label apply.
9. Use the class tree: Select a subtree, Rename a class, Delete a class.
10. Complete & Export with a section selection → downloads GameraXML reflecting
    labels, groups, splits, renames, and the still-soft-deleted exclusions.

If a step fails, localize via the Network tab + the API's structured
`{code, detail}` errors, then to: coordinate space (overlay `viewBox`), selection
wiring (set vs. primary), input focus (a keyboard listener leaking into the
autocomplete), or the lazy-delete pass in `useComplete`.

## How it grew

The UI was built in three phases, all shipped:

- **Phase A (MVP)** — upload → grid sorted by ascending confidence → single-glyph
  reclassify → save → export GameraXML.
- **Phase B** — first-class page image: bbox overlay, two-way hover/selection
  linkage, zoom/pan/lasso, soft-delete recycle bin, Neumes sort options,
  manual-vs-auto coloring.
- **Phase C** — multi-edit, the class-tree sidebar (select/rename/delete), and
  manual grouping; plus the manual split flow.

Beyond the phases, the app added: binarization-method switching, the
binarized/original glyph view, toolbar `k` selection + explicit reclassify, an
undo stack, move-to-category, "apply each in own class" batch confirmation,
split-&-reclassify, the group before/after preview, training presets + uploaded
training sets + vocabularies on upload, the checkbox export-section menu, and the
staging / deep-link / `postMessage` embedding path for mothra.
