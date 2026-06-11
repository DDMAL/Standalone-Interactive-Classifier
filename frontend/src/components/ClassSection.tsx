import { GlyphTile } from "@/components/GlyphTile";
import { SORT_LABELS, type SortMode, sortGlyphs } from "@/lib/format";
import { useUiStore } from "@/store/uiStore";
import type { GlyphCategory, GlyphDTO } from "@/types/api";
import { clsx } from "clsx";
import { useMemo, useState } from "react";

const TILE_MIN_WIDTH = 88;
const ROW_HEIGHT = 116;
const SORT_MODES: SortMode[] = [
  "conf-asc",
  "conf-desc",
  "name-asc",
  "name-desc",
];

interface ClassSectionProps {
  category: GlyphCategory;
  glyphs: GlyphDTO[];
  defaultOpen: boolean;
  /** Show sort controls in the header. Only Neumes opts in today. */
  sortable?: boolean;
}

/**
 * One collapsible MOTHRA-category section in the glyph display area. The
 * header toggles fold state (seeded from `defaultOpen`); the body lays the
 * category's tiles out in a responsive grid.
 */
export function ClassSection({
  category,
  glyphs,
  defaultOpen,
  sortable = false,
}: ClassSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [sortMode, setSortMode] = useState<SortMode>("conf-asc");
  const selectedGlyphIds = useUiStore((s) => s.selectedGlyphIds);

  const orderedGlyphs = useMemo(
    () => (sortable ? sortGlyphs(glyphs, sortMode) : glyphs),
    [glyphs, sortable, sortMode],
  );

  return (
    <section className="mb-2 overflow-hidden rounded-lg border border-slate-200 bg-white">
      <div className="flex items-center justify-between px-3 py-2 hover:bg-slate-50">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex flex-1 items-center gap-2 text-left text-sm font-semibold text-slate-800"
        >
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
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-normal text-slate-500">
            {glyphs.length}
          </span>
        </button>
        {sortable && open && (
          <label className="flex items-center gap-1 text-xs text-slate-500">
            <span>Sort:</span>
            <select
              value={sortMode}
              onChange={(e) => setSortMode(e.target.value as SortMode)}
              className="rounded border border-slate-300 bg-white px-1 py-0.5 text-xs text-slate-700"
            >
              {SORT_MODES.map((mode) => (
                <option key={mode} value={mode}>
                  {SORT_LABELS[mode]}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {open &&
        (orderedGlyphs.length === 0 ? (
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
            {orderedGlyphs.map((glyph) => (
              <GlyphTile
                key={glyph.id}
                glyph={glyph}
                selected={selectedGlyphIds.has(glyph.id)}
              />
            ))}
          </div>
        ))}
    </section>
  );
}
