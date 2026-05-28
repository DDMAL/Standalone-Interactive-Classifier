import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import { clsx } from "clsx";
import { type PointerEvent as ReactPointerEvent, memo } from "react";

interface BBoxLayerProps {
  glyphs: GlyphDTO[];
  selectedIds: Set<string>;
  hoverId: string | null;
}

function classFor(
  selected: boolean,
  hovered: boolean,
  isManual: boolean,
): string {
  return clsx(
    "cursor-pointer transition-colors",
    isManual
      ? selected
        ? "fill-green-500/30 stroke-green-600"
        : hovered
          ? "fill-green-400/20 stroke-green-500"
          : "fill-green-500/10 stroke-green-500"
      : selected
        ? "fill-blue-500/25 stroke-blue-500"
        : hovered
          ? "fill-amber-300/20 stroke-amber-500"
          : "fill-transparent stroke-slate-400/70 hover:stroke-amber-500",
  );
}

function BBoxLayerImpl({ glyphs, selectedIds, hoverId }: BBoxLayerProps) {
  const toggleGlyph = useUiStore((s) => s.toggleGlyph);
  const selectGlyph = useUiStore((s) => s.selectGlyph);
  const setHover = useUiStore((s) => s.setHover);

  function onRectPointerDown(e: ReactPointerEvent<SVGRectElement>) {
    // Keep useLasso from treating this as a background drag.
    e.stopPropagation();
  }

  function onRectClick(id: string, e: React.MouseEvent<SVGRectElement>) {
    e.stopPropagation();
    if (e.shiftKey || e.metaKey) toggleGlyph(id);
    else selectGlyph(id);
  }

  // Two passes so the muted Text/Staves bboxes paint underneath the
  // interactive Neume bboxes (and selection highlights).
  const decor: GlyphDTO[] = [];
  const interactive: GlyphDTO[] = [];
  for (const g of glyphs) {
    if (g.category === "Neumes") interactive.push(g);
    else decor.push(g);
  }

  return (
    <g>
      <g className="pointer-events-none">
        {decor.map((g) => (
          <rect
            key={g.id}
            x={g.ulx}
            y={g.uly}
            width={g.ncols}
            height={g.nrows}
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
            strokeDasharray="3 3"
            className="fill-transparent stroke-slate-300/60"
          />
        ))}
      </g>
      {interactive.map((g) => {
        const selected = selectedIds.has(g.id);
        const hovered = hoverId === g.id;
        return (
          // biome-ignore lint/a11y/useKeyWithClickEvents: SVG rects act as a pointer overlay; keyboard selection runs through the focusable tile buttons in the grid.
          <rect
            key={g.id}
            x={g.ulx}
            y={g.uly}
            width={g.ncols}
            height={g.nrows}
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
            className={classFor(selected, hovered, g.id_state_manual)}
            onPointerEnter={() => setHover(g.id)}
            onPointerLeave={() => setHover(null)}
            onPointerDown={onRectPointerDown}
            onClick={(e) => onRectClick(g.id, e)}
          />
        );
      })}
    </g>
  );
}

export const BBoxLayer = memo(BBoxLayerImpl);
