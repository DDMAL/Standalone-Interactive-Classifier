# Frontend Plan — `ic_new/frontend/`

## Context

The `ic_new` project rewrites the legacy Rodan Interactive Classifier (Backbone.Marionette + Django + Gamera) as a non-Django Python service plus a fresh React UI. The algorithm core (`core/ic_core/`) and the FastAPI surface (`api/`) are implemented; this `frontend/` directory is empty. This plan covers building it.

Per the migration plan, the new UI is React + Vite + TypeScript. To get end-to-end value fast and avoid over-designing, we plan in **three phases** and implement only **Phase A (MVP)** now. Phase A is the smallest UI that lets a user upload a page + bbox file, see classified glyphs sorted by ascending confidence (lowest first), reclassify one glyph at a time, save, and export GameraXML. Phases B and C are sketched so the structure of Phase A doesn't paint us into a corner.

Stack choices already confirmed: **Tailwind + Radix (headless)**, **TanStack Query** for server state + **Zustand** for UI state, **page image held in browser memory as an object URL** (no API change needed for that). One small backend change is in scope: adding CORS to the FastAPI app for `http://localhost:5173`.

## Phase A — MVP (implement now)

### Workflow Phase A must support

1. User selects a page-image file and an annotation file (`json` or `yolo`), picks the format, submits.
2. Frontend `POST /sessions` (multipart), stashes the page File as an object URL, navigates to the session view.
3. Glyph grid renders all `session.glyphs` sorted ascending by `confidence`; each tile shows the embedded base64 PNG, predicted class, confidence %, and a manual-vs-auto badge.
4. Clicking a glyph opens the edit panel with a class-name autocomplete drawn from `session.class_names`. Submitting:
   - `POST /sessions/{id}/glyphs/{gid}` with `{class_name, id_state_manual:true}`
   - then `POST /sessions/{id}/classify` `{k:1}`
   - then invalidate the session query so the grid resorts.
5. **Save** → `POST /sessions/{id}/save` (no-op snapshot on the in-memory store, but exercises the contract).
6. **Complete & Export** → `POST /sessions/{id}/complete`, downloads response body as `ic-session-{id}.xml`.

### File layout

Create under `frontend/`:

```
frontend/
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── tailwind.config.ts
├── postcss.config.js
├── biome.json
├── .env.development           # VITE_API_BASE=http://127.0.0.1:8000 (fallback only)
├── .gitignore
└── src/
    ├── main.tsx               # React root + QueryClientProvider
    ├── App.tsx                # UploadView | SessionView switch on uiStore.sessionId
    ├── index.css              # Tailwind directives
    ├── api/
    │   ├── client.ts          # fetch wrapper; throws ApiError{code,detail,status}
    │   └── sessions.ts        # one function per endpoint
    ├── types/
    │   └── api.ts             # SessionDTO, GlyphDTO, ClassifierState, ApiError
    ├── hooks/
    │   ├── useSession.ts      # useQuery(['session', id])
    │   ├── useCreateSession.ts
    │   ├── useUpdateGlyph.ts
    │   ├── useClassify.ts
    │   ├── useSave.ts
    │   └── useComplete.ts     # triggers XML download on success
    ├── store/
    │   └── uiStore.ts         # Zustand: sessionId, pageObjectUrl, selectedGlyphId
    ├── lib/
    │   ├── download.ts        # blob -> <a download> helper
    │   └── format.ts          # confidence percent, sort comparator
    └── components/
        ├── UploadView.tsx
        ├── SessionView.tsx    # 3-pane layout shell + Toolbar
        ├── PageImagePane.tsx  # <img src={objectUrl}> only in Phase A
        ├── GlyphGrid.tsx      # virtualized via @tanstack/react-virtual
        ├── GlyphTile.tsx
        ├── EditPanel.tsx
        ├── ClassNameInput.tsx # Radix Popover autocomplete from class_names
        ├── Toolbar.tsx
        └── ui/                # small Radix wrappers (Button, Dialog, Popover, Label)
```

### TypeScript types — `src/types/api.ts`

```ts
export type ClassifierState = "import" | "classifying" | "export";

export interface GlyphDTO {
  id: string;
  class_name: string;
  confidence: number;
  id_state_manual: boolean;
  ulx: number; uly: number; ncols: number; nrows: number;
  image_b64: string;
}

export interface SessionDTO {
  id: string;
  state: ClassifierState;
  glyphs: GlyphDTO[];
  training_glyphs: GlyphDTO[];
  class_names: string[];
}

export interface ApiError { code: string; detail: string; status: number }
```

These mirror `SessionDTO` / `GlyphDTO` in `api/src/ic_api/schemas.py`.

### Component responsibilities

- **UploadView** — two `<input type="file">` (page image, annotations), `<select>` for `json|yolo`, optional class-names textarea (parsed to JSON list). On submit: `useCreateSession`; on success, stash the page File as `URL.createObjectURL(file)` in `uiStore` and set `sessionId`.
- **SessionView** — Tailwind flex layout: left `PageImagePane`, center `GlyphGrid`, right `EditPanel` (visible iff `selectedGlyphId`). Owns `Toolbar`. Subscribes to `useSession(sessionId)`.
- **PageImagePane** — Phase A is just `<img src={pageObjectUrl} class="max-w-full">` in an overflow-scroll container. No overlay or zoom.
- **GlyphGrid** — `useVirtualizer({count, estimateSize: ~110, lanes: floor(containerWidth / tileWidth)})`. Sorts via `useMemo(() => [...glyphs].sort((a,b) => a.confidence - b.confidence), [glyphs])`.
- **GlyphTile** — `<img src={"data:image/png;base64,"+image_b64}>`, class label, confidence %, "M" or "A" badge from `id_state_manual`. Click → `uiStore.selectGlyph(id)`.
- **EditPanel** — larger view of selected glyph; `ClassNameInput` seeded from current `class_name`. Submit handler: `updateGlyph.mutateAsync(...) → classify.mutateAsync({k:1}) → queryClient.invalidateQueries(['session', id])`. Disable submit while either mutation is pending.
- **ClassNameInput** — Radix `Popover` + filtered list from `session.class_names`; free-text allowed (legacy IC behavior).
- **Toolbar** — "Save" (`useSave`), "Complete & Export" (`useComplete`, which downloads `ic-session-{id}.xml`).

### Query keys / hooks

- Single key: `['session', sessionId]`.
- `useSession(id)`: `enabled: !!id`, `staleTime: 0`.
- All mutation `onSuccess` handlers call `queryClient.invalidateQueries(['session', id])`.
- `useCreateSession.onSuccess` writes the returned `SessionDTO` directly via `queryClient.setQueryData(['session', dto.id], dto)` and updates `uiStore`.

### Zustand store — `src/store/uiStore.ts`

```ts
interface UiState {
  sessionId: string | null;
  pageObjectUrl: string | null;       // revoked on clearSession
  selectedGlyphId: string | null;
  selectedGlyphIds: Set<string>;      // unused in Phase A; reserved for B/C
  isEditPanelOpen: boolean;
  setSession(id: string, objectUrl: string): void;
  clearSession(): void;
  selectGlyph(id: string | null): void;
}
```

### Vite config — dev proxy (not pure CORS)

```ts
// vite.config.ts
server: {
  port: 5173,
  proxy: { "/sessions": "http://127.0.0.1:8000" }
}
```

Frontend code calls `fetch("/sessions/...")` with no base prefix. Keeps dev and prod origins symmetric. `VITE_API_BASE` exists only as a fallback for non-dev builds. We *still* add CORS on the API (below) so direct `curl` / fetch from other origins works for ad-hoc testing.

### CORS change — `api/src/ic_api/main.py`

Insert immediately after the `app = FastAPI(...)` block (current line ~83 in `main.py`):

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

No dependency change — `CORSMiddleware` ships with FastAPI/Starlette.

### Packages

```
react ^18.3, react-dom ^18.3, @types/react ^18.3, @types/react-dom ^18.3
typescript ^5.6, vite ^5.4, @vitejs/plugin-react ^4.3
tailwindcss ^3.4, postcss ^8.4, autoprefixer ^10.4
@tanstack/react-query ^5.59
@tanstack/react-virtual ^3.10
zustand ^5.0
@radix-ui/react-popover ^1.1, @radix-ui/react-dialog ^1.1,
@radix-ui/react-label ^2.1, @radix-ui/react-slot ^1.1
clsx ^2.1
# dev
@biomejs/biome ^1.9
```

### Lint/format — Biome

Single binary, one config file, ~10× faster than eslint+prettier, native TS/JSX. No Phase A imports need eslint-only plugins.

## Phase B — Page overlay, multi-select, recycle bin, faster editor (shipped)

### Goals

Make the page image first-class and tighten the labelling loop. Phase A treated the page as a passive thumbnail and required a click-into-input round-trip per edit; Phase B lights up bboxes, links the page pane and the glyph grid through a shared selection set, adds zoom/pan/lasso, and grows a small set of editor affordances that came out of usage testing:

- **Non-interactive Text/Staves bboxes** — only Neumes are clickable on the page.
- **Manual-vs-auto coloring** — manually-corrected glyphs are visually green-themed in both the grid and the page overlay.
- **Keyboard-driven autocomplete** — ↑/↓ to walk suggestions, Enter applies the highlighted choice.
- **Apply-from-anywhere** — Enter outside any input applies the current class name to the selected glyph.
- **Soft delete with put-back** — a "Deleted" recycle-bin section sits below Staves; deletes commit to the backend only at Complete & Export.
- **Sort options on Neumes** — confidence asc/desc, name asc/desc.

Multi-edit is still deferred to Phase C: when ≥2 glyphs are selected the `EditPanel` shows a count + a "Delete N glyphs" button, not an editor. The single `POST /sessions/{id}/glyphs/{gid}` mutation hasn't changed; the only new mutation called during Phase B is `DELETE /sessions/{id}/glyphs/{gid}`, fanned out from `useComplete` just before `POST /complete`.

### Workflow Phase B supports

1. Page image renders with one bbox per glyph drawn from `ulx/uly/ncols/nrows`. Hover highlights, click selects.
2. Text and Staves bboxes paint as muted dashed outlines underneath the Neume layer with `pointer-events: none` — visible context, not selectable.
3. Click a tile in the grid highlights its bbox on the page; click a bbox scrolls the matching tile into view in the grid and opens (or keeps) the `EditPanel`.
4. Pointer-drag on empty page area draws a lasso; release commits intersecting Neume glyph ids to `selectedGlyphIds`. `Shift`/`Cmd`-drag unions with the current selection; plain drag replaces. Non-Neume and soft-deleted ids are filtered out of the hit set.
5. `Shift`/`Cmd`-click on either a tile or a bbox toggles that id in the selection set.
6. `+` / `=` zooms in, `-` zooms out (anchored at the cursor), `0` resets, arrow keys pan, `Esc` clears selection.
7. Selecting 2+ glyphs shows a "N selected — multi-edit in Phase C" placeholder with a single "Delete N glyphs" action; selecting exactly 1 shows the Phase A editor.
8. With one glyph selected, **Enter** anywhere outside an input applies the current `class_name` and re-classifies — same effect as clicking "Apply & reclassify".
9. Inside the class-name input, **↑/↓** walks the suggestion list and **Enter** picks the highlighted suggestion and applies in one shot (Esc closes the popover).
10. **Delete** moves a glyph to the "Deleted" section. Put-back from there restores it. The actual `DELETE` call is deferred to Complete & Export.
11. The Neumes section header carries a sort dropdown: confidence (low→high / high→low) and name (A→Z / Z→A).
12. Manually-classified glyphs (`id_state_manual === true`) keep a green theme on their tile and their bbox across hover / selected / idle states.

### Selection model — set-first

`uiStore` is set-first. `primaryGlyphId` is the last-touched id, used for `EditPanel` framing and `useSelectionSync` scroll.

```ts
interface UiState {
  // ... session, hover, etc.
  selectedGlyphIds: Set<string>;
  primaryGlyphId: string | null;
  hoverGlyphId: string | null;

  selectGlyph(id: string | null): void;             // replace with {id}, or clear
  toggleGlyph(id: string): void;                    // shift/cmd-click
  setSelection(ids: Iterable<string>): void;        // lasso commit (replace)
  extendSelection(ids: Iterable<string>): void;     // lasso commit + modifier (union)
  clearSelection(): void;
  setHover(id: string | null): void;

  // Recycle bin — see "Soft delete" below.
  deletedGlyphIds: Set<string>;
  softDeleteGlyphs(ids: Iterable<string>): void;
  restoreGlyph(id: string): void;
  clearDeleted(): void;
}
```

`ClassSection`, `BBoxLayer`, and `EditPanel` read `selectedGlyphIds` (membership test) and `primaryGlyphId` (for the panel). `softDeleteGlyphs` also strips deleted ids from `selectedGlyphIds`/`primaryGlyphId`/`hoverGlyphId` to keep the three sets consistent.

### Files

New under `src/`:

```
components/
├── PageOverlay.tsx         # absolutely-positioned <svg> sibling of the <img>
├── BBoxLayer.tsx           # decor pass (Text/Staves) + interactive Neume rects, memoized
├── LassoLayer.tsx          # dashed marquee, mounted during pointer-drag
├── ZoomPanContainer.tsx    # CSS-transform wrapper; consumes useZoomPan
├── DeletedSection.tsx      # recycle-bin section at the bottom of the grid
hooks/
├── useZoomPan.ts           # scale + translate state, wheel/keyboard handlers
├── useLasso.ts             # pointerdown→move→up state machine, rAF-throttled
├── useSelectionSync.ts     # scrolls primary tile into view when primary changes
lib/
├── bbox.ts                 # Rect type, intersect, screenToImage
├── keymap.ts               # key→action table + isEditableTarget guard
└── tileRefs.ts             # module-scoped {glyphId → HTMLElement} registry
```

Removed: the `useDeleteGlyph` hook stub — `useComplete` calls `deleteGlyph` from the API client directly.

Modified:

- `components/PageImagePane.tsx` — replaces the bare `<img>` with `ZoomPanContainer > {img, PageOverlay}`; captures `naturalWidth/Height` from `img.onload` for coordinate scaling. Receives `zoomPan` from `SessionView`.
- `components/SessionView.tsx` — owns `useZoomPan`, the window `keydown` listener, and `useSelectionSync()`; passes raw `session.glyphs` to `PageImagePane` and a pre-sorted (confidence asc) copy to `GlyphGrid` for the default order.
- `components/GlyphTile.tsx` — selected via set membership; `onClick` reads `event.shiftKey || event.metaKey` to call `toggleGlyph` vs `selectGlyph`; registers itself in `tileRefs` for selection-sync; carries the green palette when `id_state_manual`.
- `components/ClassSection.tsx` — reads `selectedGlyphIds`; takes `sortable?: boolean` so only the Neumes section renders the sort dropdown.
- `components/GlyphGrid.tsx` — partitions glyphs into category groups + deleted bucket in one pass; appends `<DeletedSection>` after Staves; passes `sortable={category === "Neumes"}`.
- `components/EditPanel.tsx` — branches on `selectedGlyphIds.size`: 0 → no panel, 1 → `SingleEditor` (with delete + Enter-from-anywhere + autocomplete-driven apply), ≥2 → `MultiSelectionPanel` (count + bulk delete + clear hint).
- `components/ClassNameInput.tsx` — combobox: active-index state + ↑/↓ navigation + Enter apply (via `onApply`) + `onMouseDown` preventDefault on items to keep input focus.
- `hooks/useComplete.ts` — drains `deletedGlyphIds` via parallel `DELETE` calls and `clearDeleted()` before calling `completeSession`.
- `api/sessions.ts` — adds `deleteGlyph(id, glyphId)`.
- `lib/format.ts` — adds `SortMode`, `SORT_LABELS`, and `sortGlyphs(glyphs, mode)`.
- `store/uiStore.ts` — see "Selection model" above; resets `hoverGlyphId` and `deletedGlyphIds` on `setSession`/`clearSession`.

### `PageOverlay` and coordinate space

Page image is drawn at its natural size scaled by CSS (`max-w-full`). The overlay `<svg>` mounts as a sibling with `position: absolute; inset: 0` and `viewBox="0 0 naturalWidth naturalHeight"` so bbox coordinates can be used **as-is** — no per-rect math. `preserveAspectRatio="xMinYMin meet"` keeps it locked to the image while the zoom wrapper scales the outer transform.

```tsx
<svg
  viewBox={`0 0 ${naturalW} ${naturalH}`}
  preserveAspectRatio="xMinYMin meet"
  className="absolute inset-0 h-full w-full"
  role="presentation" aria-hidden
  onPointerDown={lasso.onPointerDown}
  onPointerMove={lasso.onPointerMove}
  onPointerUp={lasso.onPointerUp}
>
  <title>Glyph bounding box overlay</title>
  <BBoxLayer glyphs={visibleGlyphs} selectedIds={...} hoverId={...} />
  <LassoLayer rect={lasso.rect} />
</svg>
```

`vectorEffect="non-scaling-stroke"` is the key trick — strokes stay 1 CSS pixel wide regardless of zoom level, so deep zoom doesn't paint fat outlines.

`visibleGlyphs` excludes `deletedGlyphIds`. Inside `BBoxLayer` glyphs are split into two passes:

- **Decor pass** — `category !== "Neumes"`. Rendered first inside a `<g className="pointer-events-none">` with a dashed `stroke-slate-300/60` outline. Visible context, no hover/click/lasso pickup.
- **Interactive pass** — `category === "Neumes"`. Per-rect `onPointerEnter`/`onClick`; the `onPointerDown` on each rect `stopPropagation()`s so it doesn't trip the lasso.

`classFor(selected, hovered, isManual)` produces the rect's class:

- Manual → green palette (idle/hover/selected get progressively deeper green).
- Non-manual → slate idle, amber hover, blue selected.

### `useZoomPan`

State: `{ scale: number; tx: number; ty: number }`. Applied as `transform: translate(tx, ty) scale(scale)` with `transform-origin: 0 0` on `ZoomPanContainer`'s inner div.

- Wheel with `ctrlKey`/`metaKey` (trackpad pinch) → zoom around cursor: compute new scale (clamped to `[0.25, 8]`), then adjust `tx/ty` so the cursor's local point stays under the cursor.
- Plain wheel → pan (`tx -= deltaX; ty -= deltaY`).
- `+`/`=`/`-`/`0` from `keymap.ts` call into the same zoom-at-anchor helper, using the container center as the anchor.
- Arrow keys nudge `tx/ty` by a constant step (40 px).

### `useLasso`

State machine; commits Neume hits only. Phases:

1. `pointerdown` on overlay background (`e.target === e.currentTarget`) → record anchor in image coords, set capture, remember `event.shiftKey || event.metaKey`.
2. `pointermove` → update the cached `{x,y}`; schedule a `requestAnimationFrame` to redraw the marquee at most once per frame.
3. `pointerup` → walk `glyphs`, skip non-Neumes (and implicitly skip deleted ids because `PageOverlay` already filtered them out before passing the list in), and `extendSelection(ids)` if modifier else `setSelection(ids)`. A zero-motion drag clears the selection unless a modifier was held.

Intersection helper in `lib/bbox.ts`:

```ts
export function intersects(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}
```

Marquee redraw is rAF-throttled; hit testing runs only on `pointerup`, so there's no per-move scan over the glyph list.

### Keyboard map

| Key                          | Action                                                                |
| ---------------------------- | --------------------------------------------------------------------- |
| `+` / `=`                    | Zoom in, anchored at container center                                 |
| `-` / `_`                    | Zoom out                                                              |
| `0`                          | Reset zoom + pan                                                      |
| `Esc`                        | `clearSelection()` (outside input); closes the autocomplete (inside) |
| Arrow keys                   | Pan (outside input); walk suggestions (inside the class-name input)   |
| `Enter`                      | Apply current class name (outside input) / apply highlighted (inside) |
| `Shift`/`Cmd`-click          | Toggle id in selection (tile *or* bbox)                               |
| `Shift`/`Cmd`-drag on page   | Lasso union with current selection                                    |

Handlers live in two places:

- A single `useEffect` in `SessionView` for the global shortcuts; `isEditableTarget(e.target)` short-circuits when an `<input>`/`<textarea>` has focus so the autocomplete isn't intercepted.
- An `onKeyDown` on the input inside `ClassNameInput` for combobox navigation; this listener `preventDefault`s on Enter when `onApply` is wired so the surrounding form doesn't double-submit.

There's also a sibling window listener mounted by `SingleEditor` for **Enter outside any input** — it reads `applyRef.current` (always pointing at the latest `applyClassName`) and calls it. This is what makes "click a bbox → press Enter" feel like one motion.

### `useSelectionSync`

Subscribes to `primaryGlyphId`. When it changes, defers one rAF tick (so a just-mounted tile is in `tileRefs`) and calls `tileEl?.scrollIntoView({ block: "nearest", behavior: "smooth" })`. Tile refs are kept in a module-scoped `Map<glyphId, HTMLElement>` in `lib/tileRefs.ts` populated by `GlyphTile`'s `ref` callback — no `useImperativeHandle` plumbing.

### Soft delete + lazy commit

Backend has `DELETE /sessions/{id}/glyphs/{gid}` but no resurrect endpoint, so a one-shot delete would be irreversible. Phase B handles that as a frontend recycle bin:

- "Delete glyph" / "Delete N glyphs" call `softDeleteGlyphs([...ids])` — adds to `deletedGlyphIds`, drops from selection/hover. **No network call**.
- `GlyphGrid` partitions session glyphs into category groups vs deleted bucket. Deleted ones render in `<DeletedSection>` (amber-tinted, collapsed by default) under Staves. Each tile carries a "Put back" button that calls `restoreGlyph`.
- `PageOverlay` filters `deletedGlyphIds` out of both `BBoxLayer` and `useLasso`'s hit list — deleted glyphs vanish from the page image and can't be lassoed back in.
- `useComplete` is the choke point that actually persists deletes: `Promise.all(ids.map(id => deleteGlyph(sessionId, id)))` → `clearDeleted()` → `completeSession()`. The downloaded XML reflects the deletions.

Caveat: aborting via "New session" without exporting drops the deletions only from the UI; the backend still has those glyphs. That's fine — the session is being thrown away anyway.

### Sort options

`lib/format.ts` adds `SortMode = "conf-asc" | "conf-desc" | "name-asc" | "name-desc"`, `SORT_LABELS` for the option text, and `sortGlyphs(glyphs, mode)`. `ClassSection` carries a local `sortMode` state (default `"conf-asc"`, matching the previous behaviour) and applies `useMemo`-ed `sortGlyphs` only when its `sortable` prop is true. `GlyphGrid` passes `sortable={category === "Neumes"}`, so Text/Staves are untouched.

### Manual highlight

`id_state_manual === true` switches both the tile and the bbox to a green palette so manually-corrected glyphs are visible at a glance across all states:

- `GlyphTile` — green border + faint green bg by default; deeper green on hover and on selected (a green ring marks the selected manual tile distinctly from the blue ring used for unmanual selected tiles).
- `BBoxLayer.classFor(selected, hovered, isManual)` — green fills/strokes mirror the tile palette.

The "M" badge on each tile stays, but is no longer the only signal.

### Keyboard-driven autocomplete (`ClassNameInput`)

Combobox state:

- `activeIndex` defaults to `-1` (nothing highlighted) and resets when the filtered list changes (so typing more characters can't fire-and-forget the wrong item).
- `↑` / `↓` open the popover (if closed) and cycle `activeIndex` over `filtered`; the active item scrolls into view.
- `Enter` with `activeIndex >= 0` writes `filtered[activeIndex]` via `onChange`, closes the popover, and calls `onApply(option)` — `preventDefault` keeps the form from double-submitting. With no highlight, `Enter` just calls `onApply(value)`.
- `Esc` closes the popover.
- Mouse hover over an item also moves `activeIndex`, so kbd and mouse don't disagree.
- `onMouseDown={(e) => e.preventDefault()}` on each suggestion blocks the focus shift that would otherwise pop the user out of typing flow on click.

`EditPanel.SingleEditor`'s `applyClassName` takes an optional `override` so the apply can use the just-picked suggestion synchronously (no stale-state round-trip through React state).

### Performance budget

Sessions we expect (a few hundred glyphs per page) render fine as plain SVG. Defer the canvas split unless profiling justifies it:

- ≤ ~2,000 rects: keep all-SVG; one DOM node per glyph is cheap.
- > ~2,000: split layers — render unselected bboxes into an offscreen `<canvas>` (single draw per zoom/data change), keep selected/hover on SVG (small N, easy hit testing).

### Packages

No new runtime deps. `vectorEffect`, pointer events, and CSS transforms are baseline. `clsx` carries the class-name tables.

### Verification

End-to-end manual test (Phase B complete = all steps pass):

1. Run `ic-api` + `vite dev` and import a page as in Phase A.
2. One bbox per glyph, aligned to the underlying art at every zoom level (no drift on zoom — that's the `vectorEffect` + `viewBox` test). Text/Staves render as dashed muted outlines and ignore clicks/hover/lasso.
3. Hover a Neume bbox: it highlights, and the corresponding grid tile highlights in turn. Hover a grid tile: same in reverse. (Two-way hover linkage.)
4. Click a bbox: the matching tile scrolls into view in the grid, `EditPanel` shows that glyph.
5. Drag a lasso across several glyphs: on release, all overlapping Neume tiles show selected styling; Text/Staves are skipped. `Shift`-drag again over a different region: the new ids union with the previous selection. `Esc` clears.
6. Press `+` three times then `-` once, then `0`. The image scales and resets; bbox strokes stay 1px throughout. Arrow keys pan within the container; trackpad pinch zoom anchors at the cursor.
7. With 3 glyphs selected, the `EditPanel` shows "3 selected — multi-edit in Phase C" with a "Delete 3 glyphs" button. Click it: the three glyphs disappear from the grid and the page overlay, and appear in the "Deleted" section under Staves.
8. Expand "Deleted" and click "Put back" on one tile: it reappears in its original category section and in the page overlay.
9. Click "Complete & Export". The downloaded `ic-session-{id}.xml` excludes the still-deleted glyphs; the recycle bin is empty.
10. Click one Neume bbox so its tile is the only selected one. Without clicking back into the input, press `Enter`: it applies the current class name and re-classifies the rest.
11. Click into the class-name input. Press `↓` a few times — the highlighted suggestion advances. Press `Enter`: the highlighted suggestion is applied and the grid resorts.
12. Open the Neumes sort dropdown and pick "Name (A→Z)". Tiles reorder alphabetically. Pick "Confidence (high → low)": reorders accordingly. Text/Staves order is unchanged.
13. Apply a class name to a tile. Both the tile and its bbox on the page become green, and stay green across hover and (de)selection.

If a step fails, isolate to: coordinate space (overlay viewBox), selection wiring (set vs. primary), input focus (keyboard listener leaking into the autocomplete), or the lazy-delete pass in `useComplete`.

If a step fails, isolate to: coordinate space (overlay viewBox), selection wiring (set vs. primary), or input focus (keyboard listener leaking into the autocomplete).

## Phase C — Multi-edit, class management, manual grouping (implement next)

### Goals

Phase B made the page+grid interactive and built up a real selection set, but the editor still only knows what to do with one glyph at a time, and the class vocabulary is read-only. Phase C closes both gaps:

- **Multi-edit** — apply one class name to N selected glyphs in one round-trip pattern, replacing the `MultiSelectionPanel`'s "multi-edit in Phase C" placeholder with a real editor.
- **Class management sidebar** — surface `session.class_names` as a tree the user can navigate, rename, delete, and use as a selector ("select all glyphs in `neume.punctum`").
- **Manual grouping** — merge a multi-selection into a single new glyph with a user-supplied class name, exercising `POST /sessions/{id}/group`.
- **Auto-grouping** (DEFFERED) stays hidden — backend still returns 501 from `POST /auto-group`; no UI surface for it.

All three features share the same selection set built in Phase B, so the selection model itself doesn't change; only what we do with it does.

### Workflow Phase C supports

1. With **2+ glyphs selected**, the right-side panel shows a `MultiEditPanel` (not a placeholder): one `ClassNameInput`, an "Apply to N glyphs" button, a "Group as new glyph" button, and the existing bulk-delete button. Apply fires N parallel `updateGlyph` calls, one `classify`, one invalidation. Non-Neume members of the selection are skipped server-side-safely (the panel filters them out before mutating and shows a "Skipping K non-Neume glyphs" hint).
2. With **exactly 1 glyph selected**, the Phase B `SingleEditor` is unchanged.
3. **Group N glyphs** opens a `GroupDialog` (Radix `Dialog`) with a `ClassNameInput` seeded from the most common class in the selection. Submit calls `POST /sessions/{id}/group`, invalidates the session, then selects only the new glyph (its id is in the response).
4. A new left rail, **`ClassTreePanel`**, mounts between the toolbar and `PageImagePane` (collapsible — `<` toggle on its header collapses it to a 24 px strip so users who don't need it can hide it). It renders `session.class_names` parsed as a tree on `.`-separated namespaces.
5. Per tree node:
   - Hover → a small action row: **Select**, **Rename**, **Delete**.
   - **Select** populates `selectedGlyphIds` with every working-set glyph whose `class_name` equals the node's full path (or starts with `node.path + "."` if the node is an interior node — i.e. the whole subtree).
   - **Rename** opens an inline `<input>` on the node; Enter calls `POST /sessions/{id}/classes/{name}/rename` with `{new_name: <full path>}`, invalidates session.
   - **Delete** opens a confirm `Dialog` ("Delete class 'X' and N descendant classes?"); confirm calls `DELETE /sessions/{id}/classes/{name}`, invalidates session.
6. Manually-classified glyphs that no longer match any class in the tree (e.g. after a `Delete`) still appear in the grid with their old `class_name` — the backend keeps them but the tree no longer offers that class for autocomplete. The grid tile shows the orphaned class name in italic slate so it's discoverable.
7. The "Apply to N glyphs" and "Group" actions live behind the same Enter-from-anywhere wiring as the `SingleEditor` apply (keymap honors selection size).

### Selection model — no change

`uiStore`'s set-first selection from Phase B is exactly what Phase C needs. `selectFromClass(name)` is a new store action that calls `setSelection(ids)` after walking `session.glyphs` for matches — it's a thin wrapper that lives on the `ClassTreePanel`'s side rather than in the store (the store has no session reference), see `useClassSelection` below.

### Files

New under `src/`:

```
components/
├── MultiEditPanel.tsx         # replaces MultiSelectionPanel for size>=2
├── GroupDialog.tsx            # Radix Dialog wrapping ClassNameInput; used by both apply-group and tree-rename's confirm
├── ClassTreePanel.tsx         # left rail: tree + per-node actions + collapse toggle
├── ClassTreeNode.tsx          # one recursive row inside ClassTreePanel
├── ConfirmDialog.tsx          # small Radix Dialog wrapper used by delete-class
hooks/
├── useUpdateGlyphs.ts         # bulk update: N updateGlyph + 1 classify + invalidate
├── useGroup.ts                # POST /group; on success, replace selection with returned glyph id
├── useRenameClass.ts          # POST /classes/{name}/rename
├── useDeleteClass.ts          # DELETE /classes/{name}
├── useClassSelection.ts       # walks session.glyphs for a class/prefix, calls setSelection
lib/
├── classTree.ts               # parses string[] of class names into ClassNode tree (dot-separated)
```

Modified:

- `api/sessions.ts` — adds `updateGlyphsBulk(...)` helper (not strictly required — `Promise.all(updateGlyph(...))` works fine and keeps the surface minimal; only add a single-call helper if profiling later shows N tiny POSTs are the bottleneck), `manualGroup(id, body)`, `renameClass(id, name, newName)`, `deleteClass(id, name)`.
- `components/SessionView.tsx` — adds `<ClassTreePanel />` as the first child of the row that currently holds `PageImagePane | GlyphGrid | EditPanel`; manages its open/collapsed state in `uiStore` so it survives session-view re-renders.
- `components/EditPanel.tsx` — branch on selection size now routes `>=2` to `MultiEditPanel` instead of `MultiSelectionPanel`; the file keeps `SingleEditor` exactly as is.
- `components/Toolbar.tsx` — no new button (group lives in `MultiEditPanel`); only adds a small `class_names.length` chip next to the glyph count if helpful for orienting the user against the tree.
- `store/uiStore.ts` — adds `classTreeCollapsed: boolean` + `setClassTreeCollapsed(v: boolean)`; resets on `setSession`/`clearSession`.
- `lib/format.ts` — no change.
- `lib/keymap.ts` — no new keys; the Enter-from-anywhere handler in `MultiEditPanel` mirrors the one in `SingleEditor` (separate window listener, same `applyRef` trick).

Removed:

- `components/EditPanel.tsx`'s `MultiSelectionPanel` (deleted — `MultiEditPanel` subsumes it). The bulk-delete button moves with it.

### `MultiEditPanel`

Same width and chrome as `SingleEditor` so the layout doesn't reflow when selection size crosses the 1↔2 boundary.

Top: "N selected · K Neumes, M non-Neumes" header with a Clear (`×`) button. Below, a stacked layout:

- **Class-name editor** — `ClassNameInput` seeded with the most common `class_name` across the K Neume members (`""` if none classified yet). `onApply` calls `applyToMany(value)`.
- **Apply button** — `Apply to K Neumes` (disabled when K is 0 or `pending` or empty input). Spinner text mirrors `SingleEditor`.
- **Group button** — `Group as new glyph` opens `GroupDialog`. Disabled when fewer than 2 glyphs are selected (the panel itself only renders for `>=2`, so the gate is really only the Neume-count check — grouping rejects empty/single sets on the backend with a 400, which we let bubble).
- **Delete button** — `Delete N glyphs` (same red-bordered ghost as today, calls `softDeleteGlyphs`).

`applyToMany(name)`:
```
const ids = neumeIds(selectedGlyphIds, session.glyphs);
if (ids.length === 0 || !name.trim()) return;
await Promise.all(ids.map(id =>
  updateGlyph.mutateAsync({ glyphId: id, patch: { class_name: name.trim(), id_state_manual: true } })
));
await classify.mutateAsync(1);
queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
```

Local state mirrors `SingleEditor` (a `useUpdateGlyphs` hook can wrap this, but the inline shape above is fine and keeps the mutation accounting visible — pick whichever after the first build).

**Enter-from-anywhere**: a window `keydown` listener identical in shape to the one in `SingleEditor`, gated by `isEditableTarget`, calls `applyRef.current()` on Enter. Both listeners coexist safely: only one of `SingleEditor` / `MultiEditPanel` is mounted at a time because `EditPanel` branches on `selectionSize`.

### `GroupDialog`

Radix `Dialog.Root` controlled by `MultiEditPanel`. Body: a `ClassNameInput` (seeded with the dominant class), a Submit button, a Cancel. Submit calls `useGroup`:

```
const newGlyph = await groupMutation.mutateAsync({ glyph_ids: ids, class_name: name });
queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
selectGlyph(newGlyph.id);   // replace selection with the merged glyph
```

The dialog uses `Dialog.Portal + Dialog.Overlay + Dialog.Content` and Tailwind `data-[state=open]`/`data-[state=closed]` selectors for fade in/out (no extra animation library — Radix sets the data attribute and Tailwind's `data-*` variants handle the rest). Focus management is Radix's default (focus trap on Content, return on close).

### `ClassTreePanel` + `ClassTreeNode`

Width: 200 px expanded, 24 px collapsed. The collapse toggle is a chevron button in the header. State lives in `uiStore` (`classTreeCollapsed`).

Parses `session.class_names` once into a tree via `lib/classTree.ts`:

```ts
export interface ClassNode {
  segment: string;          // e.g. "punctum"
  path: string;             // e.g. "neume.punctum"
  children: ClassNode[];
  /** True iff `path` itself appears in the input list (vs. only as a prefix). */
  isLeafClass: boolean;
}
export function buildClassTree(names: string[]): ClassNode[];
```

Memoize via `useMemo(() => buildClassTree(session.class_names), [session.class_names])`. The default sort is alphabetical per level.

Each `ClassTreeNode` row:
- `<button>` with twist-arrow + segment label + a small count badge showing how many working-set glyphs match (`useMemo` against `session.glyphs`). Clicking the row toggles expand/collapse.
- On hover (or focus-within for keyboard), an inline action group slides in to the right with `Select`, `Rename`, `Delete` icons (small Lucide-style SVG inlines kept in `components/ui/icons.tsx` — no icon package needed for three icons).
- Rename uses an inline edit-in-place: the segment becomes an `<input>` initialized with the current segment. Enter dispatches `useRenameClass` with the full path (renaming only happens on the leaf segment; the API renames the full string in the class list, and re-classifies references — that's a property of the backend's `session.rename_class`, see [api/src/ic_api/main.py#L548-L561](api/src/ic_api/main.py#L548-L561)). Esc cancels.
- Delete pops `ConfirmDialog`. On confirm, dispatches `useDeleteClass`.
- Select fires `useClassSelection(node.path, includeSubtree=!node.isLeafClass || node.children.length > 0)`.

Empty state ("No classes yet — apply a class to a glyph to start populating this tree") shows when `class_names` is empty.

### `useClassSelection`

```ts
export function useClassSelection() {
  const { data: session } = useSession(/* current sessionId from store */);
  const setSelection = useUiStore((s) => s.setSelection);
  return (path: string, includeSubtree: boolean) => {
    if (!session) return;
    const prefix = path + ".";
    const ids = session.glyphs
      .filter(g => g.class_name === path || (includeSubtree && g.class_name.startsWith(prefix)))
      .map(g => g.id);
    setSelection(ids);
  };
}
```

Walking the full glyph list per click is fine — sessions in the working size targeted by Phase B (a few hundred glyphs per page) make this a sub-millisecond pass; no need to index by class name.

### API client additions — `api/sessions.ts`

```ts
export const manualGroup = (id: string, body: { glyph_ids: string[]; class_name: string }) =>
  http.post<GlyphDTO>(`/sessions/${id}/group`, body);

export const renameClass = (id: string, name: string, new_name: string) =>
  http.post<SessionDTO>(`/sessions/${id}/classes/${encodeURIComponent(name)}/rename`, { new_name });

export const deleteClass = (id: string, name: string) =>
  http.delete<SessionDTO>(`/sessions/${id}/classes/${encodeURIComponent(name)}`);
```

(The backend returns `SessionDTO` for rename/delete, so the mutation hooks can also `setQueryData(['session', id], dto)` directly and skip an invalidate-refetch round trip — match the Phase A `useCreateSession` pattern.)

### Auto-grouping

Hidden in the UI. Keep `auto-group` out of the API client until the core ships `auto_group_shaped`. If a user ever lands on a 501 (e.g. from a future placeholder button), the existing `ApiError{code:"deferred"}` shape already prints a meaningful message via the toolbar's error state.

### Error handling

All four new mutations route through the existing `http`/`ApiError` plumbing, so:

- `409 state_conflict` (e.g. trying to rename a class on a completed session) bubbles a `state_conflict` toast and the panel becomes read-only.
- `400 validation_error` for renaming a class to `UNCLASSIFIED` (forbidden by core, see [core/ic_core/state.py](core/ic_core/state.py)) shows inline next to the rename input.
- `404 not_found` for renaming/deleting a class that's no longer in the list is silently swallowed after a session invalidate (the tree refreshes and the node disappears).
- A failed bulk-apply leaves partial state intact (the mutations are independent on the backend). The hook collects per-id errors and shows a small "K of N applied" status under the apply button; the user can hit Apply again to re-run on the still-mismatched glyphs.

### Keyboard map (additions)

| Key            | Action                                                                  |
| -------------- | ----------------------------------------------------------------------- |
| `Enter`        | Apply current class to all Neumes in the multi-selection (outside input)|
| `Cmd/Ctrl+G`   | Open `GroupDialog` (when `selectionSize >= 2` and Neume-count >= 2)     |
| `Cmd/Ctrl+E`   | Focus the multi-edit `ClassNameInput`                                   |

Wired in `keymap.ts` (new `KeyAction` variants `openGroupDialog` / `focusClassEditor`) and dispatched from `SessionView`'s existing window listener. Both gate on `selectionSize >= 2` and `isEditableTarget` short-circuit so they don't fire while typing in the input.

### Packages

No new runtime deps. `@radix-ui/react-dialog` is already in Phase A's package list and is exactly what `GroupDialog` and `ConfirmDialog` use. Three small inline SVG icons live in `components/ui/icons.tsx`; no `lucide-react` import.

### Performance budget

- Bulk apply with 200 glyphs is 200 small POSTs. At ~5 ms/req on localhost, that's ~1 s total; well inside the user's expected feel for a bulk action. If the dataset grows past that, the right move is a backend `PATCH /sessions/{id}/glyphs` taking an array — not orchestration tricks on the client. Don't add it pre-emptively.
- The class tree rebuild on `session.class_names` change is `O(N log N)` for the sort + `O(N · avg depth)` for the tree build. Sessions have on the order of ~10²-10³ classes max; trivial.

### Verification

End-to-end manual test (Phase C complete = all steps pass):

1. Run `ic-api` + `vite dev`, import a page as in Phase A, do a Phase B lasso to grab 5 Neume glyphs.
2. `MultiEditPanel` shows in the right rail with "5 selected · 5 Neumes, 0 non-Neumes". Type `punctum`, press Enter (outside any input — i.e. with focus on a tile or bbox). All five glyphs flip to `punctum` with the green manual badge; the grid resorts; the classifier ran once.
3. Lasso a mixed set (3 Neumes, 2 Text). Header reads "5 selected · 3 Neumes, 2 non-Neumes". Apply: only the 3 Neumes change class; the 2 Text glyphs are untouched and the "Skipping 2 non-Neume glyphs" hint shows.
4. With 4 glyphs selected, click **Group as new glyph**. `GroupDialog` opens with the dominant class pre-filled; type `clivis`, submit. The 4 source glyphs disappear; one new glyph appears with class `clivis` and the union bbox; selection now contains only the new glyph's id; the right panel switches to `SingleEditor`.
5. Open `ClassTreePanel`. With `neume.punctum`, `neume.clivis`, `neume.virga`, `text.lyrics` in the class list, the tree shows `neume/` (3 children) and `text/` (1 child). Click the `neume` row's `Select`: every working-set glyph whose class starts with `neume.` is selected (multi-edit panel reappears).
6. Hover `neume.punctum`, click `Rename`, change to `punctum`, Enter. The tree refreshes (`punctum` is now top-level; the previously-classified glyphs' tiles show the new class name).
7. Hover `text.lyrics`, click `Delete`. `ConfirmDialog` opens; confirm. The class disappears from the tree; any glyph that had `class_name === "text.lyrics"` keeps its label visible in italic slate but the autocomplete no longer offers it.
8. Press `Cmd/Ctrl+G` with 2+ glyphs selected — opens `GroupDialog`. Press `Cmd/Ctrl+E` — focuses the multi-edit `ClassNameInput`. Both keys do nothing while typing inside any input.
9. Collapse the `ClassTreePanel` via its chevron; the page image and grid expand. Reopen the session (or refresh): the panel is back open (state lives in `uiStore` and resets on `setSession`, which is the right behaviour — page-specific, not browser-persisted).
10. Click "Complete & Export". XML reflects the renamed classes, the merged group glyph, and the still-soft-deleted exclusions from Phase B.

If a step fails, isolate to: bulk-mutation orchestration (`Promise.all` ordering vs. invalidate timing), class-tree parsing (off-by-one on `isLeafClass`), or the `selectionSize` branching in `EditPanel`.

### Future — split one bbox into many (sketch)

Inverse of manual grouping: take one glyph whose detector bbox spans several true glyphs (a common MOTHRA failure mode) and cut it into N replacement glyphs.

- **Trigger** — toolbar/`SingleEditor` button "Split…" visible when exactly one glyph is selected. Opens a `SplitDialog` (Radix `Dialog`, larger than `GroupDialog`).
- **Dialog content** — the page-image region covered by the source glyph's bbox, cropped and zoomed to fill the dialog (reuse `PageImagePane`'s `<svg viewBox>` trick so coordinates stay in image space). User drags one or more lassos inside the crop; each release commits a sub-rect to a local `parts: Rect[]` list, drawn in a distinct color and labelled by index. Per-part: a small inline `ClassNameInput` seeded from the source glyph's `class_name`, and an `×` to drop the part.
- **Submit** — calls a new `POST /sessions/{id}/glyphs/{gid}/split` endpoint with `{parts: [{ulx, uly, ncols, nrows, class_name}, ...]}`. Backend responsibility: crop each rect out of the source glyph's image, create N new glyphs (manual, since the user drew them), delete the source. Frontend invalidates the session and selects the new glyph ids.
- **Lasso reuse** — the same `useLasso` machine from Phase B drives the inside-dialog drag, just bound to a smaller container and with no Neume-only filter (it's drawing rects, not picking glyphs). Marquee styling is shared via `LassoLayer`.
- **Why deferred** — the backend endpoint and the pixel-cropping in `ic_core` don't exist yet. The MOTHRA category model also has no place for "child of split source", which we may or may not need to track for provenance. Hold for Phase D / a dedicated split spike.

Until that lands, users work around the case by deleting the over-spanning bbox and re-running MOTHRA upstream — not great, but no data is lost.

## Critical files

- New: `frontend/src/api/sessions.ts`, `frontend/src/components/SessionView.tsx`, `frontend/src/components/GlyphGrid.tsx`, `frontend/src/components/EditPanel.tsx`, plus the rest of the layout above.
- Modify: `api/src/ic_api/main.py` — add `CORSMiddleware` block.

## Verification

End-to-end manual test (Phase A complete = all six steps pass):

1. From `api/`: `uv sync && uv run ic-api`. Confirm `Uvicorn running on http://127.0.0.1:8000`.
2. From `frontend/`: `npm install && npm run dev`. Vite serves on `http://localhost:5173`.
3. In the browser, pick a page image and a JSON or YOLO annotation file from `core/data/train/` or `core/data/test/`, set format, submit.
4. `SessionView` renders: page image on the left, glyph grid populated, tiles sorted ascending by confidence (lowest first).
5. Click the lowest-confidence tile, type a class name in the autocomplete, submit. Confirm the tile updates with the new class and an "M" badge; the grid resorts because classify ran on the rest.
6. Click **Save** — confirm 200 in the Network tab (no visible change is expected).
7. Click **Complete & Export** — confirm `ic-session-{id}.xml` downloads and contains GameraXML.

If a step fails, the Network tab plus the API's structured `{code, detail}` errors should localize the cause.
