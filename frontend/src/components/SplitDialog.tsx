import { Button } from "@/components/ui/Button";
import { useSplit } from "@/hooks/useSplit";
import { rectFromAnchor } from "@/lib/bbox";
import type { Rect } from "@/lib/bbox";
import { glyphDataUri } from "@/lib/format";
import type { GlyphDTO } from "@/types/api";
import * as Dialog from "@radix-ui/react-dialog";
import {
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

interface SplitDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sessionId: string;
  glyph: GlyphDTO;
}

/**
 * Drawing-canvas dialog that wraps `POST /sessions/{id}/glyphs/{gid}/split`.
 *
 * The user draws one or more axis-aligned rectangles over the source
 * glyph's image; each rectangle becomes one new UNCLASSIFIED child glyph
 * on submit. Rectangles are drawn in glyph-local coordinates (the SVG
 * viewBox covers the parent's `ncols × nrows` plus a margin on each
 * side); we add the parent's `ulx`/`uly` on submit so the API receives
 * page-coordinate regions.
 *
 * The margin around the image is intentional: it lets the user start or
 * end a drag *outside* the glyph's pixel bounds, which is the natural
 * gesture for drawing a rect that hugs the image edge. Drafts that fall
 * outside the parent's bbox get clipped by `snapToPixels` on commit, so
 * the API still only sees in-bounds regions.
 *
 * Children come back UNCLASSIFIED with confidence 0, so they surface at
 * the top of the ascending-confidence queue for the user to label one at
 * a time. We clear the selection on success rather than picking the
 * children, because they likely don't all belong to the same class —
 * sending them through multi-edit would be the wrong default.
 */
export function SplitDialog({
  open,
  onOpenChange,
  sessionId,
  glyph,
}: SplitDialogProps) {
  // Committed rectangles in glyph-local coordinates (integers, ≥1 px).
  const [rects, setRects] = useState<Rect[]>([]);
  // In-progress drag rectangle; null when not drawing. Kept separate from
  // `rects` so a cancelled drag (e.g. zero-motion click) doesn't leak.
  const [draft, setDraft] = useState<Rect | null>(null);
  const draftRef = useRef<{
    pointerId: number;
    anchorX: number;
    anchorY: number;
  } | null>(null);
  const rafRef = useRef<number | null>(null);
  const split = useSplit(sessionId);
  const splitReset = split.reset;

  // Pad the drawing surface beyond the image so the user can start/end
  // drags from outside the image. Scales with the smaller dimension so
  // it stays a visible-but-not-dominant margin at any glyph size.
  const pad = useMemo(
    () => Math.max(8, Math.round(Math.min(glyph.ncols, glyph.nrows) * 0.15)),
    [glyph.ncols, glyph.nrows],
  );
  const vbW = glyph.ncols + 2 * pad;
  const vbH = glyph.nrows + 2 * pad;

  // Reset state every time the dialog opens; otherwise yesterday's
  // rectangles linger for the next victim.
  useEffect(() => {
    if (open) {
      setRects([]);
      setDraft(null);
      draftRef.current = null;
      splitReset();
    }
  }, [open, splitReset]);

  // Convert a client (screen) pointer position into viewBox coordinates.
  // Since our viewBox starts at (-pad, -pad), this naturally produces
  // negative values when the pointer is inside the margin zone, which is
  // exactly what we want for the draft rect — the snap-and-clip on
  // commit handles the conversion back into in-bounds glyph coords.
  const pointerToGlyph = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>): { x: number; y: number } => {
      const svg = e.currentTarget;
      const r = svg.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) return { x: 0, y: 0 };
      return {
        x: ((e.clientX - r.left) / r.width) * vbW - pad,
        y: ((e.clientY - r.top) / r.height) * vbH - pad,
      };
    },
    [pad, vbW, vbH],
  );

  const flushDraft = useCallback((curX: number, curY: number) => {
    rafRef.current = null;
    const d = draftRef.current;
    if (!d) return;
    setDraft(rectFromAnchor(d.anchorX, d.anchorY, curX, curY));
  }, []);

  const onPointerDown = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      if (e.target !== e.currentTarget) return;
      if (split.isPending) return;
      e.preventDefault();
      const { x, y } = pointerToGlyph(e);
      draftRef.current = { pointerId: e.pointerId, anchorX: x, anchorY: y };
      e.currentTarget.setPointerCapture(e.pointerId);
      setDraft({ x, y, w: 0, h: 0 });
    },
    [pointerToGlyph, split.isPending],
  );

  const onPointerMove = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      const d = draftRef.current;
      if (!d || d.pointerId !== e.pointerId) return;
      const { x, y } = pointerToGlyph(e);
      if (rafRef.current === null) {
        rafRef.current = requestAnimationFrame(() => flushDraft(x, y));
      }
    },
    [pointerToGlyph, flushDraft],
  );

  const onPointerUp = useCallback(
    (e: ReactPointerEvent<SVGSVGElement>) => {
      const d = draftRef.current;
      if (!d || d.pointerId !== e.pointerId) return;
      draftRef.current = null;
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        /* already released */
      }
      const { x, y } = pointerToGlyph(e);
      const final = rectFromAnchor(d.anchorX, d.anchorY, x, y);
      setDraft(null);
      // Snap to integer pixel grid and clip to the parent's bbox.
      // Zero-motion clicks (or fully-outside drags) are discarded — the
      // backend would reject non-positive sizes anyway.
      const snapped = snapToPixels(final, glyph.ncols, glyph.nrows);
      if (snapped) setRects((rs) => [...rs, snapped]);
    },
    [pointerToGlyph, glyph.ncols, glyph.nrows],
  );

  function handleDeleteRect(idx: number) {
    setRects((rs) => rs.filter((_, i) => i !== idx));
  }

  function handleClearAll() {
    setRects([]);
  }

  async function handleSubmit() {
    if (rects.length === 0 || split.isPending) return;
    const regions: [number, number, number, number][] = rects.map((r) => [
      r.x + glyph.ulx,
      r.y + glyph.uly,
      r.w,
      r.h,
    ]);
    try {
      await split.mutateAsync({ glyphId: glyph.id, regions });
      onOpenChange(false);
    } catch {
      // error state shown inline; dialog stays open so the user can adjust
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-slate-900/40" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[90vw] max-w-2xl -translate-x-1/2 -translate-y-1/2 rounded-lg border border-slate-200 bg-white p-5 shadow-lg focus:outline-none">
          <Dialog.Title className="text-base font-semibold text-slate-800">
            Split glyph into pieces
          </Dialog.Title>
          <Dialog.Description className="mt-2 text-sm text-slate-600">
            Drag on the image to draw rectangles. Each rectangle becomes one new
            unclassified glyph; the original is removed. Children are
            re-classified on the next round.
          </Dialog.Description>

          <SplitCanvas
            glyph={glyph}
            pad={pad}
            vbW={vbW}
            vbH={vbH}
            rects={rects}
            draft={draft}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onDeleteRect={handleDeleteRect}
          />

          <div className="mt-3 flex items-center justify-between text-xs text-slate-600">
            <span>
              {rects.length === 0
                ? "Drag on the image to draw your first rectangle."
                : `${rects.length} rectangle${rects.length === 1 ? "" : "s"} drawn.`}
            </span>
            {rects.length > 0 && (
              <Button
                variant="ghost"
                onClick={handleClearAll}
                disabled={split.isPending}
                className="px-2 py-0.5 text-xs"
              >
                Clear all
              </Button>
            )}
          </div>

          {split.isError && (
            <p className="mt-2 text-xs text-red-600">
              {(split.error as Error)?.message}
            </p>
          )}

          <div className="mt-5 flex justify-end gap-2">
            <Dialog.Close asChild>
              <Button variant="ghost" disabled={split.isPending}>
                Cancel
              </Button>
            </Dialog.Close>
            <Button
              onClick={() => void handleSubmit()}
              disabled={rects.length === 0 || split.isPending}
            >
              {split.isPending
                ? "Splitting…"
                : `Split into ${rects.length || ""} glyph${
                    rects.length === 1 ? "" : "s"
                  }`.trim()}
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

interface SplitCanvasProps {
  glyph: GlyphDTO;
  /** Margin (in image-pixel units) extending the viewBox around the image. */
  pad: number;
  /** Total viewBox width = ncols + 2*pad. */
  vbW: number;
  /** Total viewBox height = nrows + 2*pad. */
  vbH: number;
  rects: Rect[];
  draft: Rect | null;
  onPointerDown: (e: ReactPointerEvent<SVGSVGElement>) => void;
  onPointerMove: (e: ReactPointerEvent<SVGSVGElement>) => void;
  onPointerUp: (e: ReactPointerEvent<SVGSVGElement>) => void;
  onDeleteRect: (idx: number) => void;
}

/**
 * The drawing surface: a single SVG containing both the source glyph's
 * image (via `<image>`) and the rectangle overlay, with a viewBox that
 * extends `pad` pixels beyond the image on every side. That margin is
 * what makes drags-starting-outside-the-image work.
 *
 * `image-rendering: pixelated` keeps neume binarisations legible at
 * zoom-in instead of blurring them. A thin slate stroke marks the
 * image bounds so the margin zone reads as "drawable padding" rather
 * than "the image extends here."
 */
function SplitCanvas({
  glyph,
  pad,
  vbW,
  vbH,
  rects,
  draft,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onDeleteRect,
}: SplitCanvasProps) {
  // Screen pixels per viewBox unit. The labels and delete buttons are
  // authored in *screen* pixels (a constant readable size) and converted
  // to viewBox units via `k = 1 / unitPx`, so they stay the same physical
  // size on screen no matter how high-resolution the glyph is — a small
  // glyph zooms in, a large one zooms out, but the chrome stays legible.
  const svgRef = useRef<SVGSVGElement>(null);
  const [unitPx, setUnitPx] = useState(0);
  useLayoutEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const measure = () => {
      const w = svg.getBoundingClientRect().width;
      // width / vbW == height / vbH under xMidYMid meet with a matching
      // aspect ratio, so either axis yields the same scale.
      if (w > 0) setUnitPx(w / vbW);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(svg);
    return () => ro.disconnect();
  }, [vbW]);

  return (
    <div className="mt-4 flex max-h-[60vh] items-center justify-center overflow-hidden rounded border border-slate-200 bg-slate-100 p-2">
      <svg
        ref={svgRef}
        viewBox={`${-pad} ${-pad} ${vbW} ${vbH}`}
        preserveAspectRatio="xMidYMid meet"
        className="block max-h-[55vh] cursor-crosshair touch-none"
        style={{ aspectRatio: `${vbW} / ${vbH}`, maxWidth: "100%" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <title>Draw split rectangles</title>
        {/* Margin background — slightly different shade so the user
            sees where the image ends and the drawable margin begins. */}
        <rect
          x={-pad}
          y={-pad}
          width={vbW}
          height={vbH}
          className="fill-slate-50 pointer-events-none"
        />
        {/* The source glyph at its native pixel size. preserveAspectRatio
            on the parent SVG handles overall scaling; the image renders
            at its true ncols × nrows so coordinates line up exactly.
            `pointer-events: none` is critical — without it the image
            steals pointerdown events from the SVG, and `onPointerDown`'s
            `e.target !== e.currentTarget` guard then refuses to start a
            drag (you'd only be able to draw from the margin zone). */}
        <image
          href={glyphDataUri(glyph)}
          x={0}
          y={0}
          width={glyph.ncols}
          height={glyph.nrows}
          preserveAspectRatio="none"
          style={{ imageRendering: "pixelated", pointerEvents: "none" }}
        />
        {/* Outline of the image bounds so the user can see them at a
            glance — particularly important when the image is mostly
            white pixels and would otherwise blend into the margin. */}
        <rect
          x={0}
          y={0}
          width={glyph.ncols}
          height={glyph.nrows}
          fill="none"
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
          className="stroke-slate-400 pointer-events-none"
        />
        {rects.map((r, i) => {
          // ViewBox units per screen pixel — the inverse of `unitPx`. Used
          // to size the label/button chrome in constant screen pixels.
          // Until the first measurement lands, `unitPx` is 0; skip the
          // chrome that frame rather than divide by zero (it appears on
          // the next paint once the observer fires).
          const k = unitPx > 0 ? 1 / unitPx : 0;
          // Chrome sizes in screen px, converted to viewBox units.
          const font = 20 * k;
          const padU = 2 * k;
          const badge = 25 * k; // fixed circular badge size
          const chromeH = badge + 2 * padU; // row height incl. inset padding
          const labelNudge = { x: -1, y: 0 };  // screen px; tweak to taste
          const xNudge = { x: 2, y: -1 };     // × glyphs usually sit a hair high

          return (
            // biome-ignore lint/suspicious/noArrayIndexKey: rects are append-only and reorderable via delete only
            <g key={i}>
              {/* Same pointer-events-none reasoning as the image above: a
                  committed rect must not block new drags that start on
                  top of it. The delete button below keeps its own
                  `pointer-events-auto`, so removal still works. */}
              <rect
                x={r.x}
                y={r.y}
                width={r.w}
                height={r.h}
                strokeWidth={1.5}
                vectorEffect="non-scaling-stroke"
                className="fill-emerald-500/20 stroke-emerald-600 pointer-events-none"
              />
              {k > 0 && (
                // One row pinned to the top edge of the rect: number on
                // the left, delete on the right. Height is the chrome's
                // own height (not the rect's) and overflow is visible, so
                // a short or zoomed rect can never clip the controls; the
                // flex row keeps both flush to the rect's top corners
                // regardless of the rect's size.
                <foreignObject
                  x={r.x}
                  y={r.y}
                  width={r.w}
                  height={chromeH}
                  className="overflow-visible pointer-events-none"
                >
                  <div
                    className="flex w-full items-start justify-between"
                    style={{ gap: `${padU}px`, padding: `${padU}px` }}
                  >
                    <span
                      className="inline-flex shrink-0 items-center justify-center bg-emerald-600 font-semibold leading-none text-white shadow"
                      style={{
                        minWidth: `${badge}px`,
                        height: `${badge}px`,
                        padding: `0 ${padU * 2}px`,
                        borderRadius: `${5 * k}px`,
                        fontSize: `${font}px`,
                      }}
                    >
                      <span style={{ transform: `translate(${labelNudge.x * k}px, ${labelNudge.y * k}px)` }}>
                        {i + 1}
                      </span>
                    </span>
                    {/* The foreignObject lets us place a real <button> over
                        the SVG so the click target is properly hit-tested. */}
                    <button
                      type="button"
                      onPointerDown={(e) => e.stopPropagation()}
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteRect(i);
                      }}
                      className="pointer-events-auto inline-flex shrink-0 items-center justify-center bg-white/90 font-semibold leading-none text-rose-600 shadow hover:bg-rose-50"
                      style={{
                        minWidth: `${badge}px`,
                        height: `${badge}px`,
                        padding: `0 ${padU * 2}px`,
                        borderRadius: `${5 * k}px`,
                        fontSize: `${font}px`,
                      }}
                      aria-label={`Remove rectangle ${i + 1}`}
                    >
                      <span style={{ transform: `translate(${xNudge.x * k}px, ${xNudge.y * k}px)` }}>
                        ×
                      </span>
                    </button>
                  </div>
                </foreignObject>
              )}
            </g>
          );
        })}
        {draft && (
          <rect
            x={draft.x}
            y={draft.y}
            width={draft.w}
            height={draft.h}
            strokeWidth={1.5}
            vectorEffect="non-scaling-stroke"
            strokeDasharray="4 2"
            className="fill-emerald-500/10 stroke-emerald-500 pointer-events-none"
          />
        )}
      </svg>
    </div>
  );
}

/**
 * Snap a continuous-coordinate rectangle to the integer pixel grid and
 * clip it to the parent's [0, ncols) × [0, nrows) bbox. Returns null for
 * degenerate (zero-area) rectangles so the caller can discard them
 * silently — the backend would reject `ncols <= 0 || nrows <= 0` anyway.
 */
function snapToPixels(
  r: Rect,
  parentNcols: number,
  parentNrows: number,
): Rect | null {
  const x0 = Math.max(0, Math.round(r.x));
  const y0 = Math.max(0, Math.round(r.y));
  const x1 = Math.min(parentNcols, Math.round(r.x + r.w));
  const y1 = Math.min(parentNrows, Math.round(r.y + r.h));
  const w = x1 - x0;
  const h = y1 - y0;
  if (w <= 0 || h <= 0) return null;
  return { x: x0, y: y0, w, h };
}
