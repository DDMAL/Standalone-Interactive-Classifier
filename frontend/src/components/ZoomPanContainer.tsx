import type { UseZoomPan } from "@/hooks/useZoomPan";
import { type ReactNode, useEffect } from "react";

interface ZoomPanContainerProps {
  zoomPan: UseZoomPan;
  children: ReactNode;
}

/**
 * Viewport for the page image. The outer div clips and catches wheel; the
 * inner div carries the translate+scale.
 */
export function ZoomPanContainer({ zoomPan, children }: ZoomPanContainerProps) {
  const { containerRef, onWheel } = zoomPan;

  // Wheel is bound here rather than via React's `onWheel` prop: React
  // delegates wheel passively at the root, so the handler's preventDefault
  // is ignored and the browser still performs its default scroll. Nothing
  // inside this iframe scrolls, so that default chains out to the host
  // document and scrolls mothra's page under us -- most visibly on a laptop,
  // where the host page is tall enough to have somewhere to go. A
  // non-passive listener makes preventDefault stick and keeps the gesture in
  // the pane.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [containerRef, onWheel]);

  return (
    <div
      ref={containerRef}
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
