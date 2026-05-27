import { GlyphTile } from "@/components/GlyphTile";
import { useUiStore } from "@/store/uiStore";
import type { GlyphCategory, GlyphDTO } from "@/types/api";
import { clsx } from "clsx";
import { useState } from "react";

const TILE_MIN_WIDTH = 88;
const ROW_HEIGHT = 116;

interface ClassSectionProps {
  category: GlyphCategory;
  glyphs: GlyphDTO[];
  defaultOpen: boolean;
}

/**
 * One collapsible MOTHRA-category section in the glyph display area. The
 * header toggles fold state (seeded from `defaultOpen`); the body lays the
 * category's tiles out in a responsive grid.
 */
export function ClassSection({ category, glyphs, defaultOpen }: ClassSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const selectedGlyphId = useUiStore((s) => s.selectedGlyphId);
  const selectGlyph = useUiStore((s) => s.selectGlyph);

  return (
    <section className="mb-2 overflow-hidden rounded-lg border border-slate-200 bg-white">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 hover:bg-slate-50"
      >
        <span className="flex items-center gap-2 text-sm font-semibold text-slate-800">
          <span
            aria-hidden
            className={clsx(
              "inline-block text-slate-400 transition-transform",
              open && "rotate-90",
            )}
          >
            ▶
          </span>
          {category}
        </span>
        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
          {glyphs.length}
        </span>
      </button>

      {open &&
        (glyphs.length === 0 ? (
          <p className="px-3 pb-3 text-xs text-slate-400">
            No glyphs in this class.
          </p>
        ) : (
          <div
            className="grid gap-2 p-2"
            style={{
              gridTemplateColumns: `repeat(auto-fill, minmax(${TILE_MIN_WIDTH}px, 1fr))`,
              gridAutoRows: `${ROW_HEIGHT}px`,
            }}
          >
            {glyphs.map((glyph) => (
              <GlyphTile
                key={glyph.id}
                glyph={glyph}
                selected={glyph.id === selectedGlyphId}
                onSelect={selectGlyph}
              />
            ))}
          </div>
        ))}
    </section>
  );
}
