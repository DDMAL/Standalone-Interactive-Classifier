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

## Phase B — Page-image lasso + zoom (sketch only)

- Replace `<img>` with `<canvas>` (or layered `<img>` + absolutely-positioned `<svg>` for crisp bbox strokes); wrap in a CSS-transform zoom container.
- Render one rect per glyph from `ulx/uly/ncols/nrows`; fill on hover, stroke on `selectedGlyphIds.has(id)`.
- Pointer-down on empty area starts a lasso rect; pointer-up commits intersecting ids into `uiStore.selectedGlyphIds`. Shift-drag = union, plain drag = replace.
- Keyboard: `+`/`=` zoom in, `-` zoom out (anchor at cursor), `0` reset, arrow keys pan.
- `GlyphGrid` reads the same selection set and scrolls the first selection into view via the virtualizer's `scrollToIndex`.

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
