import { ClassSection } from "@/components/ClassSection";
import {
  CATEGORY_DEFAULT_OPEN,
  CATEGORY_ORDER,
  type GlyphCategory,
  type GlyphDTO,
} from "@/types/api";
import { useMemo } from "react";

/**
 * The glyph display area: three collapsible class sections (Neumes, Text,
 * Staves) grouped by MOTHRA category. Incoming glyph order is preserved
 * within each section, so the ascending-confidence review order set by the
 * backend still holds for Neumes.
 */
export function GlyphGrid({ glyphs }: { glyphs: GlyphDTO[] }) {
  const byCategory = useMemo(() => {
    const groups: Record<GlyphCategory, GlyphDTO[]> = {
      Neumes: [],
      Text: [],
      Staves: [],
    };
    for (const glyph of glyphs) {
      (groups[glyph.category] ?? groups.Neumes).push(glyph);
    }
    return groups;
  }, [glyphs]);

  return (
    <div className="min-w-0 flex-1 overflow-auto bg-slate-50 p-2">
      {glyphs.length === 0 ? (
        <p className="p-4 text-sm text-slate-400">No glyphs in this session.</p>
      ) : (
        CATEGORY_ORDER.map((category) => (
          <ClassSection
            key={category}
            category={category}
            glyphs={byCategory[category]}
            defaultOpen={CATEGORY_DEFAULT_OPEN[category]}
          />
        ))
      )}
    </div>
  );
}