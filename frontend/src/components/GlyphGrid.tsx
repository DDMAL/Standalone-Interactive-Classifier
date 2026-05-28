import { ClassSection } from "@/components/ClassSection";
import { DeletedSection } from "@/components/DeletedSection";
import { useUiStore } from "@/store/uiStore";
import {
  CATEGORY_DEFAULT_OPEN,
  CATEGORY_ORDER,
  type GlyphCategory,
  type GlyphDTO,
} from "@/types/api";
import { useMemo } from "react";

/**
 * The glyph display area: three collapsible class sections (Neumes, Text,
 * Staves) grouped by MOTHRA category, plus a recycle-bin "Deleted" section
 * at the bottom. Incoming glyph order is preserved within each section
 * (the Neumes section's own sort dropdown can override it).
 */
export function GlyphGrid({ glyphs }: { glyphs: GlyphDTO[] }) {
  const deletedGlyphIds = useUiStore((s) => s.deletedGlyphIds);

  const { byCategory, deleted } = useMemo(() => {
    const groups: Record<GlyphCategory, GlyphDTO[]> = {
      Neumes: [],
      Text: [],
      Staves: [],
    };
    const del: GlyphDTO[] = [];
    for (const glyph of glyphs) {
      if (deletedGlyphIds.has(glyph.id)) {
        del.push(glyph);
        continue;
      }
      (groups[glyph.category] ?? groups.Neumes).push(glyph);
    }
    return { byCategory: groups, deleted: del };
  }, [glyphs, deletedGlyphIds]);

  return (
    <div className="min-w-0 flex-1 overflow-auto bg-slate-50 p-2">
      {glyphs.length === 0 ? (
        <p className="p-4 text-sm text-slate-400">No glyphs in this session.</p>
      ) : (
        <>
          {CATEGORY_ORDER.map((category) => (
            <ClassSection
              key={category}
              category={category}
              glyphs={byCategory[category]}
              defaultOpen={CATEGORY_DEFAULT_OPEN[category]}
              sortable={category === "Neumes"}
            />
          ))}
          <DeletedSection glyphs={deleted} />
        </>
      )}
    </div>
  );
}
