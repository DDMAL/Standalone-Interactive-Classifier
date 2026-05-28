import { create } from "zustand";

interface UiState {
  sessionId: string | null;
  pageObjectUrl: string | null;

  // Selection is a set; primaryGlyphId is the last-touched id, used for
  // framing the EditPanel and as the scroll-into-view target.
  selectedGlyphIds: Set<string>;
  primaryGlyphId: string | null;

  hoverGlyphId: string | null;

  // Soft-deleted ids — hidden from the grid/overlay/lasso but recoverable
  // via restoreGlyph. Committed to the backend at Complete & Export time.
  deletedGlyphIds: Set<string>;

  // Whether the left-rail class tree is collapsed. Page-specific; resets
  // on setSession/clearSession.
  classTreeCollapsed: boolean;
  setClassTreeCollapsed: (v: boolean) => void;

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
}

export const useUiStore = create<UiState>((set, get) => ({
  sessionId: null,
  pageObjectUrl: null,
  selectedGlyphIds: new Set(),
  primaryGlyphId: null,
  hoverGlyphId: null,
  deletedGlyphIds: new Set(),
  classTreeCollapsed: false,

  setClassTreeCollapsed: (v) => set({ classTreeCollapsed: v }),

  setSession: (id, objectUrl) => {
    const prev = get().pageObjectUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      sessionId: id,
      pageObjectUrl: objectUrl,
      selectedGlyphIds: new Set(),
      primaryGlyphId: null,
      hoverGlyphId: null,
      deletedGlyphIds: new Set(),
      classTreeCollapsed: false,
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
      deletedGlyphIds: new Set(),
      classTreeCollapsed: false,
    });
  },

  selectGlyph: (id) =>
    set(
      id === null
        ? { selectedGlyphIds: new Set(), primaryGlyphId: null }
        : { selectedGlyphIds: new Set([id]), primaryGlyphId: id },
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
      set({ selectedGlyphIds: next, primaryGlyphId: id });
    }
  },

  setSelection: (ids) => {
    const next = new Set(ids);
    const arr = [...next];
    set({
      selectedGlyphIds: next,
      primaryGlyphId: arr.length ? arr[arr.length - 1] : null,
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
    set({ selectedGlyphIds: next, primaryGlyphId: last });
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

    set({
      deletedGlyphIds: nextDeleted,
      selectedGlyphIds: nextSelected,
      primaryGlyphId: nextPrimary,
      hoverGlyphId: nextHover,
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
}));
