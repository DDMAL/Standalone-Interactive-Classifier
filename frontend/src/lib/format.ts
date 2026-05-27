import type { GlyphDTO } from "@/types/api";

export const formatConfidence = (c: number): string =>
  `${(c * 100).toFixed(1)}%`;

// Lowest-confidence glyphs first — they need the most attention.
export const byConfidenceAsc = (a: GlyphDTO, b: GlyphDTO): number =>
  a.confidence - b.confidence;

export const glyphDataUri = (glyph: GlyphDTO): string =>
  `data:image/png;base64,${glyph.image_b64}`;
