import { PageOverlay } from "@/components/PageOverlay";
import { ZoomPanContainer } from "@/components/ZoomPanContainer";
import type { UseZoomPan } from "@/hooks/useZoomPan";
import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import { useState } from "react";

interface PageImagePaneProps {
  glyphs: GlyphDTO[];
  zoomPan: UseZoomPan;
}

interface NaturalSize {
  w: number;
  h: number;
}

export function PageImagePane({ glyphs, zoomPan }: PageImagePaneProps) {
  const pageObjectUrl = useUiStore((s) => s.pageObjectUrl);
  const [natural, setNatural] = useState<NaturalSize>({ w: 0, h: 0 });

  return (
    <aside className="w-1/3 shrink-0 border-r border-slate-200 bg-slate-100">
      {pageObjectUrl ? (
        <ZoomPanContainer zoomPan={zoomPan}>
          <div className="relative">
            <img
              src={pageObjectUrl}
              alt="Manuscript page"
              draggable={false}
              onLoad={(e) => {
                const img = e.currentTarget;
                setNatural({ w: img.naturalWidth, h: img.naturalHeight });
              }}
              className="block max-w-full select-none"
            />
            <PageOverlay
              glyphs={glyphs}
              naturalWidth={natural.w}
              naturalHeight={natural.h}
            />
          </div>
        </ZoomPanContainer>
      ) : (
        <p className="p-4 text-sm text-slate-400">Page image unavailable.</p>
      )}
    </aside>
  );
}
