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

## Phase C — Multi-edit + class management + manual grouping (sketch only)

- **Multi-edit modal** (Radix `Dialog`) opens when `selectedGlyphIds.size > 1`; one `ClassNameInput`; submit fires N parallel `updateGlyph` mutations via `Promise.all`, then one `classify`, then invalidate.
- **Class tree sidebar** — parse `session.class_names` on dot-separated namespaces (e.g. `neume.punctum`) into a tree; per-node actions: Rename (`POST /classes/{name}/rename`), Delete (`DELETE /classes/{name}`), Select-all-with-this-class (populates `selectedGlyphIds`).
- **Manual grouping** — toolbar button enabled when `selectedGlyphIds.size >= 2`; prompts for class name, then `POST /group`; invalidate session.
- `POST /auto-group` is 501 — keep its UI hidden until the core implements it.

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
