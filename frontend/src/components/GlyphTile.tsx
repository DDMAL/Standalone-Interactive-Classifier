import { formatConfidence, glyphDataUri } from "@/lib/format";
import type { GlyphDTO } from "@/types/api";
import { clsx } from "clsx";

interface GlyphTileProps {
  glyph: GlyphDTO;
  selected: boolean;
  onSelect: (id: string) => void;
}

export function GlyphTile({ glyph, selected, onSelect }: GlyphTileProps) {
  return (
    <button
      type="button"
      onClick={() => onSelect(glyph.id)}
      className={clsx(
        "flex h-full w-full flex-col items-center gap-1 rounded border p-1 text-center transition-colors",
        selected
          ? "border-blue-500 bg-blue-50"
          : "border-slate-200 bg-white hover:border-slate-400",
      )}
    >
      <div className="flex h-16 w-full items-center justify-center overflow-hidden bg-white">
        <img
          src={glyphDataUri(glyph)}
          alt={glyph.class_name}
          className="max-h-16 max-w-full object-contain"
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
