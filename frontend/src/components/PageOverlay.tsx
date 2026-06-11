import { BBoxLayer } from "@/components/BBoxLayer";
import { LassoLayer } from "@/components/LassoLayer";
import { useLasso } from "@/hooks/useLasso";
import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import { useMemo } from "react";

interface PageOverlayProps {
  glyphs: GlyphDTO[];
  naturalWidth: number;
  naturalHeight: number;
}

/**
 * Absolute-positioned SVG sibling of the page <img>. Uses the image's
 * natural dimensions as the viewBox so bbox coordinates can be drawn
 * verbatim — no per-rect scaling math.
 */
export function PageOverlay({
  glyphs,
  naturalWidth,
  naturalHeight,
}: PageOverlayProps) {
  const selectedIds = useUiStore((s) => s.selectedGlyphIds);
  const hoverId = useUiStore((s) => s.hoverGlyphId);
  const deletedIds = useUiStore((s) => s.deletedGlyphIds);
  const visibleGlyphs = useMemo(
    () => glyphs.filter((g) => !deletedIds.has(g.id)),
    [glyphs, deletedIds],
  );
  const lasso = useLasso(visibleGlyphs, naturalWidth, naturalHeight);

  if (naturalWidth === 0 || naturalHeight === 0) return null;

  return (
    <svg
      viewBox={`0 0 ${naturalWidth} ${naturalHeight}`}
      preserveAspectRatio="xMinYMin meet"
      className="absolute inset-0 h-full w-full"
      style={{ pointerEvents: "auto" }}
      role="presentation"
      aria-hidden="true"
      onPointerDown={lasso.onPointerDown}
      onPointerMove={lasso.onPointerMove}
      onPointerUp={lasso.onPointerUp}
    >
      <title>Glyph bounding box overlay</title>
      <BBoxLayer
        glyphs={visibleGlyphs}
        selectedIds={selectedIds}
        hoverId={hoverId}
      />
      <LassoLayer rect={lasso.rect} />
    </svg>
  );
}
