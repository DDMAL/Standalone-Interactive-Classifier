import type { UseZoomPan } from "@/hooks/useZoomPan";
import type { ReactNode } from "react";

interface ZoomPanContainerProps {
  zoomPan: UseZoomPan;
  children: ReactNode;
}

/**
 * Viewport for the page image. The outer div clips and catches wheel; the
 * inner div carries the translate+scale.
 */
export function ZoomPanContainer({ zoomPan, children }: ZoomPanContainerProps) {
  return (
    <div
      ref={zoomPan.containerRef}
      onWheel={zoomPan.onWheel}
      className="relative h-full w-full overflow-hidden touch-none"
    >
      <div
        style={{
          transform: zoomPan.transform,
          transformOrigin: "0 0",
          width: "100%",
        }}
      >
        {children}
      </div>
    </div>
  );
}
