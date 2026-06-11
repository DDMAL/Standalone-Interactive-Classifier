import { Button } from "@/components/ui/Button";
import { glyphDataUri } from "@/lib/format";
import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import { clsx } from "clsx";
import { useState } from "react";

const TILE_MIN_WIDTH = 96;
const ROW_HEIGHT = 140;

/**
 * The recycle-bin section at the bottom of the glyph grid. Soft-deleted
 * glyphs land here; each tile carries a "Put back" button that restores
 * the glyph to its original category section. Deletes commit to the
 * backend only at Complete & Export time.
 */
export function DeletedSection({ glyphs }: { glyphs: GlyphDTO[] }) {
  const [open, setOpen] = useState(false);
  const restoreGlyph = useUiStore((s) => s.restoreGlyph);

  if (glyphs.length === 0) return null;

  return (
    <section className="mb-2 overflow-hidden rounded-lg border border-amber-200 bg-amber-50/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 hover:bg-amber-50"
      >
        <span className="flex items-center gap-2 text-sm font-semibold text-amber-900">
          <span
            aria-hidden
            className={clsx(
              "inline-block text-amber-500 transition-transform",
              open && "rotate-90",
            )}
          >
            ▶
          </span>
          Deleted
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-normal text-amber-700">
            {glyphs.length}
          </span>
        </span>
        <span className="text-xs font-normal text-amber-700">
          Removed on export
        </span>
      </button>

      {open && (
        <div
          className="grid gap-2 p-2"
          style={{
            gridTemplateColumns: `repeat(auto-fill, minmax(${TILE_MIN_WIDTH}px, 1fr))`,
            gridAutoRows: `${ROW_HEIGHT}px`,
          }}
        >
          {glyphs.map((glyph) => (
            <div
              key={glyph.id}
              className="flex h-full w-full flex-col items-center gap-1 rounded border border-amber-200 bg-white p-1 text-center"
            >
              <div className="flex h-16 w-full items-center justify-center overflow-hidden bg-white opacity-70">
                <img
                  src={glyphDataUri(glyph)}
                  alt={glyph.class_name}
                  className="max-h-16 max-w-full object-contain"
                />
              </div>
              <span
                className="w-full truncate text-xs text-slate-600"
                title={glyph.class_name}
              >
                {glyph.class_name}
              </span>
              <Button
                variant="secondary"
                onClick={() => restoreGlyph(glyph.id)}
                className="w-full px-1 py-0.5 text-[11px]"
              >
                Put back
              </Button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
