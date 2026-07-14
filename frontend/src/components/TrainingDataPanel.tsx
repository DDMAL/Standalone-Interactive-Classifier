import { Button } from "@/components/ui/Button";
import { glyphDataUri } from "@/lib/format";
import type { GlyphDTO } from "@/types/api";
import { clsx } from "clsx";
import { useMemo, useState } from "react";

interface TrainingDataPanelProps {
  glyphs: GlyphDTO[];
  onCollapse: () => void;
}

interface ClassGroup {
  className: string;
  glyphs: GlyphDTO[];
}

function groupByClass(glyphs: GlyphDTO[]): ClassGroup[] {
  const map = new Map<string, GlyphDTO[]>();
  for (const g of glyphs) {
    const bucket = map.get(g.class_name);
    if (bucket) bucket.push(g);
    else map.set(g.class_name, [g]);
  }
  return [...map.entries()]
    .map(([className, gs]) => ({ className, glyphs: gs }))
    .sort((a, b) => a.className.localeCompare(b.className));
}

/**
 * Read-only browser for the session's training pool, shown in the right-hand
 * slot in place of the EditPanel (see {@link RightDock}). Training glyphs are
 * grouped by class name into collapsible sections — the Square preset alone is
 * ~2400 glyphs, so sections start collapsed and only mount their tiles when
 * opened, keeping the DOM light.
 *
 * Tiles render the glyph's stored binarized mask directly rather than going
 * through {@link GlyphImage}: a training glyph's bbox indexes into its own
 * source page, which this frontend never loaded, so the "original" crop mode
 * would draw from the wrong image.
 */
export function TrainingDataPanel({
  glyphs,
  onCollapse,
}: TrainingDataPanelProps) {
  const groups = useMemo(() => groupByClass(glyphs), [glyphs]);

  return (
    <aside className="flex w-80 shrink-0 flex-col border-l border-slate-200 bg-white">
      <div className="flex items-start justify-between border-b border-slate-200 px-4 py-2">
        <div>
          <h2 className="text-sm font-semibold text-slate-800">
            Training data
          </h2>
          <p className="text-xs text-slate-500">
            {glyphs.length.toLocaleString()} glyphs · {groups.length}{" "}
            {groups.length === 1 ? "class" : "classes"}
          </p>
        </div>
        <Button
          variant="ghost"
          onClick={onCollapse}
          className="px-2 py-0.5"
          title="Collapse training data"
        >
          ▶
        </Button>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-2">
        {glyphs.length === 0 ? (
          <p className="p-4 text-sm text-slate-400">
            This session has no training glyphs. Pick a preset or upload a
            training set when creating the session to populate it.
          </p>
        ) : (
          groups.map((group) => (
            <TrainingClassSection
              key={group.className}
              className={group.className}
              glyphs={group.glyphs}
            />
          ))
        )}
      </div>
    </aside>
  );
}

function TrainingClassSection({ className, glyphs }: ClassGroup) {
  const [open, setOpen] = useState(false);

  return (
    <section className="mb-2 overflow-hidden rounded-lg border border-slate-200 bg-white">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-semibold text-slate-800 hover:bg-slate-50"
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
        <span className="flex-1 truncate" title={className}>
          {className}
        </span>
        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-normal text-slate-500">
          {glyphs.length}
        </span>
      </button>

      {open && (
        <div
          className="grid gap-2 p-2"
          style={{
            gridTemplateColumns: "repeat(auto-fill, minmax(64px, 1fr))",
            gridAutoRows: "64px",
          }}
        >
          {glyphs.map((glyph) => (
            <div
              key={glyph.id}
              title={className}
              className="flex items-center justify-center overflow-hidden rounded border border-slate-200 bg-white p-1"
            >
              <img
                src={glyphDataUri(glyph)}
                alt={className}
                className="h-full w-full object-contain"
              />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
