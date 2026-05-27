import { create } from "zustand";

interface UiState {
  sessionId: string | null;
  pageObjectUrl: string | null;
  selectedGlyphId: string | null;
  selectedGlyphIds: Set<string>; // reserved for Phase B/C multi-select
  setSession: (id: string, objectUrl: string) => void;
  clearSession: () => void;
  selectGlyph: (id: string | null) => void;
}

export const useUiStore = create<UiState>((set, get) => ({
  sessionId: null,
  pageObjectUrl: null,
  selectedGlyphId: null,
  selectedGlyphIds: new Set(),

  setSession: (id, objectUrl) => {
    const prev = get().pageObjectUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({ sessionId: id, pageObjectUrl: objectUrl, selectedGlyphId: null });
  },

  clearSession: () => {
    const prev = get().pageObjectUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      sessionId: null,
      pageObjectUrl: null,
      selectedGlyphId: null,
      selectedGlyphIds: new Set(),
    });
  },

  selectGlyph: (id) => set({ selectedGlyphId: id }),
}));
