import type { GlyphDTO } from "@/types/api";

export const formatConfidence = (c: number): string =>
  `${(c * 100).toFixed(1)}%`;

// Lowest-confidence glyphs first — they need the most attention.
export const byConfidenceAsc = (a: GlyphDTO, b: GlyphDTO): number =>
  a.confidence - b.confidence;

export const glyphDataUri = (glyph: GlyphDTO): string =>
  `data:image/png;base64,${glyph.image_b64}`;

export type SortMode = "conf-asc" | "conf-desc" | "name-asc" | "name-desc";

export const SORT_LABELS: Record<SortMode, string> = {
  "conf-asc": "Confidence (low → high)",
  "conf-desc": "Confidence (high → low)",
  "name-asc": "Name (A → Z)",
  "name-desc": "Name (Z → A)",
};

export function sortGlyphs(glyphs: GlyphDTO[], mode: SortMode): GlyphDTO[] {
  const out = [...glyphs];
  switch (mode) {
    case "conf-asc":
      return out.sort((a, b) => a.confidence - b.confidence);
    case "conf-desc":
      return out.sort((a, b) => b.confidence - a.confidence);
    case "name-asc":
      return out.sort((a, b) => a.class_name.localeCompare(b.class_name));
    case "name-desc":
      return out.sort((a, b) => b.class_name.localeCompare(a.class_name));
  }
}
