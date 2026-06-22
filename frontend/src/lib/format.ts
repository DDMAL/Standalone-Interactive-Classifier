import type { GlyphDTO, SessionDTO } from "@/types/api";

export const formatConfidence = (c: number): string =>
  `${(c * 100).toFixed(1)}%`;

// Mirrors the backend's `collect_training_set` so the UI can reason about the
// training pool the next classify round will actually use. A glyph counts as a
// label only if it has a real class name (not UNCLASSIFIED, no transient
// `_group`/`_delete` prefix).
const UNCLASSIFIED = "UNCLASSIFIED";
const TRANSIENT_PREFIXES = ["_group", "_delete"];

const isLabelled = (g: GlyphDTO): boolean =>
  g.class_name !== UNCLASSIFIED &&
  !TRANSIENT_PREFIXES.some((p) => g.class_name.startsWith(p));

// Size of the pool the classifier trains on: the user's in-session manual
// corrections (manual + labelled glyphs in the working set) plus any labelled
// glyphs from the external training database. Manual corrections are NOT in
// `training_glyphs` — they live in `glyphs` with `id_state_manual=true` — so
// counting `training_glyphs` alone undercounts the real pool.
export const trainingPoolSize = (session: SessionDTO): number => {
  const manual = session.glyphs.filter(
    (g) => g.id_state_manual && isLabelled(g),
  ).length;
  const external = session.training_glyphs.filter(isLabelled).length;
  return manual + external;
};

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
