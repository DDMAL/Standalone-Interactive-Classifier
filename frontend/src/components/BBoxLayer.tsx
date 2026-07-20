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
        ? "fill-green-500/40 stroke-green-700"
        : hovered
          ? "fill-green-400/20 stroke-green-500"
          : "fill-green-500/10 stroke-green-500"
      : selected
        ? "fill-violet-500/35 stroke-violet-600"
        : hovered
          ? "fill-amber-400/40 stroke-amber-600 animate-bbox-pulse motion-reduce:animate-none"
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

  // Two passes so Text/Staves bboxes paint underneath the interactive
  // Neume bboxes. Decor (non-Neume) bboxes only render when the glyph is
  // selected or hovered via the grid — they stay hidden on the page
  // otherwise so the image isn't cluttered with non-classified outlines.
  const decor: GlyphDTO[] = [];
  const interactive: GlyphDTO[] = [];
  for (const g of glyphs) {
    if (g.category === "Neumes") interactive.push(g);
    else if (selectedIds.has(g.id) || hoverId === g.id) decor.push(g);
  }

  return (
    <g>
      <g className="pointer-events-none">
        {decor.map((g) => {
          const selected = selectedIds.has(g.id);
          return (
            <rect
              key={g.id}
              x={g.ulx}
              y={g.uly}
              width={g.ncols}
              height={g.nrows}
              strokeWidth={selected ? 1 : 2}
              vectorEffect="non-scaling-stroke"
              strokeDasharray="3 3"
              className={
                selected
                  ? "fill-violet-500/20 stroke-violet-600"
                  : "fill-amber-400/30 stroke-amber-600 animate-bbox-pulse motion-reduce:animate-none"
              }
            />
          );
        })}
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
            strokeWidth={hovered && !selected ? 2.5 : 1}
            vectorEffect="non-scaling-stroke"
            className={classFor(selected, hovered, g.id_state_manual)}
            onPointerEnter={() => setHover(g.id)}
            onPointerLeave={() => setHover(null)}
            onPointerDown={onRectPointerDown}
            onClick={(e) => onRectClick(g.id, e)}
          >
            <title>{g.class_name}</title>
          </rect>
        );
      })}
    </g>
  );
}

export const BBoxLayer = memo(BBoxLayerImpl);
