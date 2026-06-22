import { formatConfidence, glyphDataUri } from "@/lib/format";
import { registerTile } from "@/lib/tileRefs";
import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import { clsx } from "clsx";
import { useCallback } from "react";

interface GlyphTileProps {
  glyph: GlyphDTO;
  selected: boolean;
}

export function GlyphTile({ glyph, selected }: GlyphTileProps) {
  const focusGlyph = useUiStore((s) => s.focusGlyph);
  const toggleGlyph = useUiStore((s) => s.toggleGlyph);
  const setHover = useUiStore((s) => s.setHover);
  const hovered = useUiStore((s) => s.hoverGlyphId === glyph.id);

  const setRef = useCallback(
    (el: HTMLButtonElement | null) => registerTile(glyph.id, el),
    [glyph.id],
  );

  return (
    <button
      type="button"
      ref={setRef}
      onClick={(e) => {
        if (e.shiftKey || e.metaKey) toggleGlyph(glyph.id);
        else focusGlyph(glyph.id);
      }}
      onPointerEnter={() => setHover(glyph.id)}
      onPointerLeave={() => setHover(null)}
      className={clsx(
        "flex h-full w-full flex-col items-center gap-1 rounded border p-1 text-center transition-colors",
        glyph.id_state_manual
          ? selected
            ? "border-green-600 bg-green-100 ring-2 ring-green-300"
            : hovered
              ? "border-green-500 bg-green-100"
              : "border-green-400 bg-green-50 hover:border-green-500"
          : selected
            ? "border-blue-500 bg-blue-50"
            : hovered
              ? "border-amber-400 bg-amber-50"
              : "border-slate-200 bg-white hover:border-slate-400",
      )}
    >
      <div className="flex h-16 w-full items-center justify-center overflow-hidden bg-white">
        <img
          src={glyphDataUri(glyph)}
          alt={glyph.class_name}
          className="h-16 w-full object-contain"
        />
      </div>
      <span
        className="w-full truncate text-xs text-slate-700"
        title={glyph.class_name}
      >
        {glyph.class_name}
      </span>
      <div className="flex w-full items-center justify-between px-0.5 text-[10px]">
        <span
          className={clsx(
            "rounded px-1 font-semibold",
            glyph.id_state_manual
              ? "bg-green-100 text-green-700"
              : "bg-slate-100 text-slate-500",
          )}
        >
          {glyph.id_state_manual ? "M" : "A"}
        </span>
        <span className="text-slate-500">
          {formatConfidence(glyph.confidence)}
        </span>
      </div>
    </button>
  );
}
