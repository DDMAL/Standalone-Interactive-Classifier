import { create } from "zustand";

/** What the glyph-grid tiles render: the binarized foreground mask, or the
 * original page crop. A display preference, not session state. */
export type GlyphImageMode = "binarized" | "original";

/** Which classifier the next classify round uses. "knn" (default) is the
 * handcrafted-feature k-nearest-neighbours classifier -- unchanged
 * behaviour. "ssl_fusion" is the optional SSL+handcrafted fused
 * logistic-regression classifier; requires the server to have the ssl
 * extra installed and IC_SSL_CHECKPOINT configured. */
export type ClassifierBackend = "knn" | "ssl_fusion";

export interface UndoEntry {
  description: string;
  snapshots: { id: string; class_name: string; id_state_manual: boolean }[];
}

interface UiState {
  sessionId: string | null;
  pageObjectUrl: string | null;

  // Selection is a set; primaryGlyphId is the last-touched id, used for
  // framing the EditPanel and as the scroll-into-view target.
  selectedGlyphIds: Set<string>;
  primaryGlyphId: string | null;

  hoverGlyphId: string | null;

  // True only while the user holds the "h" hotkey — temporarily hides all
  // bboxes on the page so the underlying image is unobscured. Transient;
  // never persisted and reset on session change.
  bboxesHidden: boolean;
  setBboxesHidden: (v: boolean) => void;

  // Count of open modal dialogs (Split, Group, …). While > 0, window-level
  // keyboard shortcuts (Enter-to-classify, type-to-focus, zoom/pan, Esc) must
  // stand down so a keypress meant for the dialog doesn't also fire an action
  // on the page underneath. A counter rather than a boolean so overlapping
  // open/close transitions can't desync the flag.
  modalOpenCount: number;
  openModal: () => void;
  closeModal: () => void;

  // Set when a glyph is selected from the grid (tile click). PageImagePane
  // watches this to re-center the page on that glyph; cleared via
  // consumeFocus once handled. Selections from the page overlay or lasso
  // never set this — they originate from the image itself, so re-centering
  // would be jarring.
  pendingFocusGlyphId: string | null;

  // Soft-deleted ids — hidden from the grid/overlay/lasso but recoverable
  // via restoreGlyph. Committed to the backend at export time.
  deletedGlyphIds: Set<string>;

  // Whether the left-rail class tree is collapsed. Page-specific; resets
  // on setSession/clearSession.
  classTreeCollapsed: boolean;
  setClassTreeCollapsed: (v: boolean) => void;

  // Whether the right-hand training-data panel is expanded. It shares the
  // slot with the EditPanel, so the two are mutually exclusive: expanding it
  // clears the current selection (yielding the slot from the EditPanel), and
  // selecting any glyph collapses it again. Page-specific; resets on
  // setSession/clearSession.
  trainingPanelExpanded: boolean;
  expandTrainingPanel: () => void;
  collapseTrainingPanel: () => void;

  // Neighbour count for the kNN classifier. User-selectable from the
  // toolbar; persists across session changes as a preference.
  knnK: number;
  setKnnK: (k: number) => void;

  // Which classifier backend to use on the next classify round. See
  // ClassifierBackend. User-selectable from the toolbar; persists as a
  // preference, defaults to "knn" so behaviour is unchanged out of the box.
  classifierBackend: ClassifierBackend;
  setClassifierBackend: (backend: ClassifierBackend) => void;

  // Whether the glyph grid shows binarized masks or original page crops.
  // A display preference; like knnK it persists across session changes.
  glyphImageMode: GlyphImageMode;
  setGlyphImageMode: (mode: GlyphImageMode) => void;

  // Undo stack for apply operations (class_name changes). Max 5 entries;
  // each entry holds the before-state for every glyph affected. Cleared on
  // session change. Does not cover split / group / rebinarize.
  undoStack: UndoEntry[];
  pushUndo: (entry: UndoEntry) => void;
  popUndo: () => UndoEntry | undefined;

  setSession: (id: string, objectUrl: string) => void;
  clearSession: () => void;

  // Replace selection with {id}. Phase A call sites still work — passing null
  // clears.
  selectGlyph: (id: string | null) => void;
  // Shift/Cmd-click on a tile or bbox.
  toggleGlyph: (id: string) => void;
  // Lasso commit without modifier.
  setSelection: (ids: Iterable<string>) => void;
  // Lasso commit with shift/cmd modifier.
  extendSelection: (ids: Iterable<string>) => void;
  clearSelection: () => void;

  setHover: (id: string | null) => void;

  softDeleteGlyphs: (ids: Iterable<string>) => void;
  restoreGlyph: (id: string) => void;
  clearDeleted: () => void;

  // Select id + request that PageImagePane re-center on it. Used by
  // GlyphTile clicks.
  focusGlyph: (id: string) => void;
  consumeFocus: () => void;
}

export const useUiStore = create<UiState>((set, get) => ({
  sessionId: null,
  pageObjectUrl: null,
  selectedGlyphIds: new Set(),
  primaryGlyphId: null,
  hoverGlyphId: null,
  bboxesHidden: false,
  modalOpenCount: 0,
  pendingFocusGlyphId: null,
  deletedGlyphIds: new Set(),
  classTreeCollapsed: false,
  trainingPanelExpanded: false,
  knnK: 3,
  classifierBackend: "knn",
  glyphImageMode: "binarized",
  undoStack: [],

  setBboxesHidden: (v) => set({ bboxesHidden: v }),

  openModal: () => set((s) => ({ modalOpenCount: s.modalOpenCount + 1 })),
  closeModal: () =>
    set((s) => ({ modalOpenCount: Math.max(0, s.modalOpenCount - 1) })),

  setClassTreeCollapsed: (v) => set({ classTreeCollapsed: v }),

  // Expanding deselects: the training panel and EditPanel share one slot, so
  // the current selection must clear for the panel to take it (satisfies
  // "expanding while the EditPanel is active deselects the glyphs").
  expandTrainingPanel: () =>
    set({
      trainingPanelExpanded: true,
      selectedGlyphIds: new Set(),
      primaryGlyphId: null,
    }),

  collapseTrainingPanel: () => set({ trainingPanelExpanded: false }),

  setKnnK: (k) => set({ knnK: k }),

  setClassifierBackend: (backend) => set({ classifierBackend: backend }),

  setGlyphImageMode: (mode) => set({ glyphImageMode: mode }),

  pushUndo: (entry) =>
    set((s) => ({
      undoStack: [...s.undoStack.slice(-4), entry],
    })),

  popUndo: () => {
    const stack = get().undoStack;
    if (stack.length === 0) return undefined;
    const entry = stack[stack.length - 1];
    set({ undoStack: stack.slice(0, -1) });
    return entry;
  },

  setSession: (id, objectUrl) => {
    const prev = get().pageObjectUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      sessionId: id,
      pageObjectUrl: objectUrl,
      selectedGlyphIds: new Set(),
      primaryGlyphId: null,
      hoverGlyphId: null,
      bboxesHidden: false,
      pendingFocusGlyphId: null,
      deletedGlyphIds: new Set(),
      classTreeCollapsed: false,
      trainingPanelExpanded: false,
      undoStack: [],
    });
  },

  clearSession: () => {
    const prev = get().pageObjectUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      sessionId: null,
      pageObjectUrl: null,
      selectedGlyphIds: new Set(),
      primaryGlyphId: null,
      hoverGlyphId: null,
      bboxesHidden: false,
      pendingFocusGlyphId: null,
      deletedGlyphIds: new Set(),
      classTreeCollapsed: false,
      trainingPanelExpanded: false,
      undoStack: [],
    });
  },

  // Every selection entry point below also collapses the training panel when
  // it produces a non-empty selection — "selecting a glyph collapses the
  // training panel". A clear (null / empty) leaves the panel as-is.
  selectGlyph: (id) =>
    set(
      id === null
        ? { selectedGlyphIds: new Set(), primaryGlyphId: null }
        : {
            selectedGlyphIds: new Set([id]),
            primaryGlyphId: id,
            trainingPanelExpanded: false,
          },
    ),

  toggleGlyph: (id) => {
    const cur = get().selectedGlyphIds;
    const next = new Set(cur);
    if (next.has(id)) {
      next.delete(id);
      const primary =
        get().primaryGlyphId === id
          ? next.size === 0
            ? null
            : [...next][next.size - 1]
          : get().primaryGlyphId;
      set({ selectedGlyphIds: next, primaryGlyphId: primary });
    } else {
      next.add(id);
      set({
        selectedGlyphIds: next,
        primaryGlyphId: id,
        trainingPanelExpanded: false,
      });
    }
  },

  setSelection: (ids) => {
    const next = new Set(ids);
    const arr = [...next];
    set({
      selectedGlyphIds: next,
      primaryGlyphId: arr.length ? arr[arr.length - 1] : null,
      ...(next.size > 0 ? { trainingPanelExpanded: false } : {}),
    });
  },

  extendSelection: (ids) => {
    const cur = get().selectedGlyphIds;
    const next = new Set(cur);
    let last: string | null = get().primaryGlyphId;
    for (const id of ids) {
      next.add(id);
      last = id;
    }
    set({
      selectedGlyphIds: next,
      primaryGlyphId: last,
      ...(next.size > 0 ? { trainingPanelExpanded: false } : {}),
    });
  },

  clearSelection: () =>
    set({ selectedGlyphIds: new Set(), primaryGlyphId: null }),

  setHover: (id) => set({ hoverGlyphId: id }),

  softDeleteGlyphs: (ids) => {
    const toDelete = new Set(ids);
    if (toDelete.size === 0) return;
    const curDeleted = get().deletedGlyphIds;
    const nextDeleted = new Set(curDeleted);
    for (const id of toDelete) nextDeleted.add(id);

    // Drop deleted ids from selection / hover.
    const curSelected = get().selectedGlyphIds;
    const nextSelected = new Set<string>();
    for (const id of curSelected) {
      if (!toDelete.has(id)) nextSelected.add(id);
    }
    const curPrimary = get().primaryGlyphId;
    const nextPrimary =
      curPrimary && toDelete.has(curPrimary)
        ? nextSelected.size === 0
          ? null
          : [...nextSelected][nextSelected.size - 1]
        : curPrimary;
    const curHover = get().hoverGlyphId;
    const nextHover = curHover && toDelete.has(curHover) ? null : curHover;

    const curFocus = get().pendingFocusGlyphId;
    const nextFocus = curFocus && toDelete.has(curFocus) ? null : curFocus;

    set({
      deletedGlyphIds: nextDeleted,
      selectedGlyphIds: nextSelected,
      primaryGlyphId: nextPrimary,
      hoverGlyphId: nextHover,
      pendingFocusGlyphId: nextFocus,
    });
  },

  restoreGlyph: (id) => {
    const cur = get().deletedGlyphIds;
    if (!cur.has(id)) return;
    const next = new Set(cur);
    next.delete(id);
    set({ deletedGlyphIds: next });
  },

  clearDeleted: () => set({ deletedGlyphIds: new Set() }),

  focusGlyph: (id) =>
    set({
      selectedGlyphIds: new Set([id]),
      primaryGlyphId: id,
      pendingFocusGlyphId: id,
      trainingPanelExpanded: false,
    }),

  consumeFocus: () => set({ pendingFocusGlyphId: null }),
}));

/**
 * Non-reactive read for use inside window-level keydown listeners, which
 * read state imperatively rather than subscribing. Returns true while any
 * modal dialog is open, signalling page-level shortcuts to stand down.
 */
export const isModalOpen = (): boolean =>
  useUiStore.getState().modalOpenCount > 0;
