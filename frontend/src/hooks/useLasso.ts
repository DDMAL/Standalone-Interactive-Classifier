import {
  intersects,
  rectFromAnchor,
  rectFromGlyph,
  screenToImage,
} from "@/lib/bbox";
import type { Rect } from "@/lib/bbox";
import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import {
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useRef,
  useState,
} from "react";

interface ActiveLasso {
  pointerId: number;
  anchorX: number;
  anchorY: number;
  curX: number;
  curY: number;
  modifier: boolean;
  moved: boolean;
}

export interface UseLasso {
  /** Image-space rect to render as a marquee, or `null` when inactive. */
  rect: Rect | null;
  onPointerDown: (e: ReactPointerEvent<SVGSVGElement>) => void;
  onPointerMove: (e: ReactPointerEvent<SVGSVGElement>) => void;
  onPointerUp: (e: ReactPointerEvent<SVGSVGElement>) => void;
}

/**
 * Pointer-state-machine for marquee selection on the overlay SVG.
 *
 * - Down on background (target === currentTarget) records the anchor and
 *   captures the pointer.
 * - Move updates the rect via rAF (one repaint per frame).
 * - Up commits: `extendSelection` if shift/cmd was held, else `setSelection`.
 *   A zero-motion drag (background click) clears the selection.
 *
 * Hit testing runs only on `pointerup`, never during the move, so deep
 * pages stay smooth.
 */
export function useLasso(
  glyphs: GlyphDTO[],
  naturalW: number,
  naturalH: number,
): UseLasso {
  const [rect, setRect] = useState<Rect | null>(null);
  const activeRef = useRef<ActiveLasso | null>(null);
  const rafRef = useRef<number | null>(null);
  const setSelection = useUiStore((s) => s.setSelection);
  const extendSelection = useUiStore((s) => s.extendSelection);
  const clearSelection = useUiStore((s) => s.clearSelection);

  const flushRect = useCallback(() => {
    rafRef.current = null;
    const a = activeRef.current;
    if (!a) return;
    setRect(rectFromAnchor(a.anchorX, a.anchorY, a.curX, a.curY));
  }, []);

  const onPointerDown = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      // Bbox <rect> children should `stopPropagation()` on their own
      // pointerdown, so reaching here means the background was hit.
      if (e.target !== e.currentTarget) return;
      if (naturalW === 0 || naturalH === 0) return;
      e.preventDefault();
      const { x, y } = screenToImage(
        e.clientX,
        e.clientY,
        e.currentTarget,
        naturalW,
        naturalH,
      );
      activeRef.current = {
        pointerId: e.pointerId,
        anchorX: x,
        anchorY: y,
        curX: x,
        curY: y,
        modifier: e.shiftKey || e.metaKey,
        moved: false,
      };
      e.currentTarget.setPointerCapture(e.pointerId);
      setRect({ x, y, w: 0, h: 0 });
    },
    [naturalW, naturalH],
  );

  const onPointerMove = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      const a = activeRef.current;
      if (!a || a.pointerId !== e.pointerId) return;
      const { x, y } = screenToImage(
        e.clientX,
        e.clientY,
        e.currentTarget,
        naturalW,
        naturalH,
      );
      a.curX = x;
      a.curY = y;
      if (Math.abs(x - a.anchorX) > 1 || Math.abs(y - a.anchorY) > 1) {
        a.moved = true;
      }
      if (rafRef.current === null) {
        rafRef.current = requestAnimationFrame(flushRect);
      }
    },
    [naturalW, naturalH, flushRect],
  );

  const onPointerUp = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      const a = activeRef.current;
      if (!a || a.pointerId !== e.pointerId) return;
      activeRef.current = null;
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        /* already released */
      }
      setRect(null);

      if (!a.moved) {
        // Background click — treat as clear, unless the user was holding a
        // modifier (in which case do nothing, to match drag semantics).
        if (!a.modifier) clearSelection();
        return;
      }

      const marquee = rectFromAnchor(a.anchorX, a.anchorY, a.curX, a.curY);
      const hits: string[] = [];
      for (const g of glyphs) {
        if (g.category !== "Neumes") continue;
        if (intersects(marquee, rectFromGlyph(g))) hits.push(g.id);
      }
      if (a.modifier) extendSelection(hits);
      else setSelection(hits);
    },
    [glyphs, clearSelection, extendSelection, setSelection],
  );

  return { rect, onPointerDown, onPointerMove, onPointerUp };
}
