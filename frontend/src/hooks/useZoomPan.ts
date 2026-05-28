import { useCallback, useRef, useState } from "react";

export interface ZoomPanState {
  scale: number;
  tx: number;
  ty: number;
}

const MIN_SCALE = 0.25;
const MAX_SCALE = 8;
const ZOOM_STEP = 1.2;

const INITIAL: ZoomPanState = { scale: 1, tx: 0, ty: 0 };

function clamp(s: number): number {
  return Math.max(MIN_SCALE, Math.min(MAX_SCALE, s));
}

export interface UseZoomPan {
  state: ZoomPanState;
  containerRef: React.RefObject<HTMLDivElement>;
  /** Use as `style={{ transform: zp.transform, transformOrigin: "0 0" }}`. */
  transform: string;
  onWheel: (e: React.WheelEvent<HTMLDivElement>) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  reset: () => void;
  pan: (dx: number, dy: number) => void;
}

/**
 * Tracks zoom (scale) and pan (tx/ty) for a CSS-transformed container.
 * Wheel + keyboard zoom anchor at the cursor or container center; the
 * anchor math keeps the focused point fixed under it.
 */
export function useZoomPan(): UseZoomPan {
  const [state, setState] = useState<ZoomPanState>(INITIAL);
  const containerRef = useRef<HTMLDivElement>(null);

  const zoomAt = useCallback(
    (factor: number, anchorClientX?: number, anchorClientY?: number) => {
      setState((cur) => {
        const next = clamp(cur.scale * factor);
        if (next === cur.scale) return cur;
        const el = containerRef.current;
        if (!el) return { ...cur, scale: next };
        const rect = el.getBoundingClientRect();
        const ax =
          anchorClientX === undefined
            ? rect.width / 2
            : anchorClientX - rect.left;
        const ay =
          anchorClientY === undefined
            ? rect.height / 2
            : anchorClientY - rect.top;
        // Local point under anchor in the child's coords:
        //   ax = tx + px*scale  →  px = (ax - tx) / scale
        // Keep visual position fixed:
        //   newTx = ax - px * newScale
        const px = (ax - cur.tx) / cur.scale;
        const py = (ay - cur.ty) / cur.scale;
        return {
          scale: next,
          tx: ax - px * next,
          ty: ay - py * next,
        };
      });
    },
    [],
  );

  const onWheel = useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      // Trackpad pinch zoom arrives as ctrlKey-wheel; plain wheel pans.
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const factor = e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
        zoomAt(factor, e.clientX, e.clientY);
        return;
      }
      e.preventDefault();
      setState((cur) => ({
        ...cur,
        tx: cur.tx - e.deltaX,
        ty: cur.ty - e.deltaY,
      }));
    },
    [zoomAt],
  );

  const zoomIn = useCallback(() => zoomAt(ZOOM_STEP), [zoomAt]);
  const zoomOut = useCallback(() => zoomAt(1 / ZOOM_STEP), [zoomAt]);
  const reset = useCallback(() => setState(INITIAL), []);
  const pan = useCallback(
    (dx: number, dy: number) =>
      setState((cur) => ({ ...cur, tx: cur.tx + dx, ty: cur.ty + dy })),
    [],
  );

  return {
    state,
    containerRef,
    transform: `translate(${state.tx}px, ${state.ty}px) scale(${state.scale})`,
    onWheel,
    zoomIn,
    zoomOut,
    reset,
    pan,
  };
}
