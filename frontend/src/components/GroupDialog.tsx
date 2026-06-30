import { ClassNameInput } from "@/components/ClassNameInput";
import { GlyphImage } from "@/components/GlyphImage";
import { Button } from "@/components/ui/Button";
import { useGroup } from "@/hooks/useGroup";
import { useModalGuard } from "@/hooks/useModalGuard";
import { PageImageProvider, usePageImageEl } from "@/hooks/usePageImage";
import { glyphDataUri } from "@/lib/format";
import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import * as Dialog from "@radix-ui/react-dialog";
import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

interface GroupDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sessionId: string;
  /** The glyphs to merge — rendered as a before/after preview. */
  glyphs: GlyphDTO[];
  /** Class name seeded into the input — usually the dominant class in the
   *  multi-selection. */
  initialClassName: string;
  classNames: string[];
}

/**
 * Dialog that wraps `POST /sessions/{id}/group`: takes a multi-selection and
 * a user-supplied class name, merges into one new manual glyph on submit.
 *
 * Shows a before/after preview — the individual source glyphs on the left and
 * a reconstruction of the merged result on the right — so the user can confirm
 * the join lines up before committing.
 */
export function GroupDialog({
  open,
  onOpenChange,
  sessionId,
  glyphs,
  initialClassName,
  classNames,
}: GroupDialogProps) {
  const [className, setClassName] = useState(initialClassName);
  const group = useGroup(sessionId);
  const groupReset = group.reset;

  const glyphIds = useMemo(() => glyphs.map((g) => g.id), [glyphs]);

  // Suppress page-level keyboard shortcuts while the dialog is open so an
  // Enter (e.g. when focus isn't in the input) doesn't also fire a
  // background apply/classify on the selection underneath.
  useModalGuard(open);

  // Reset the seed and any prior mutation error whenever the dialog opens.
  useEffect(() => {
    if (open) {
      setClassName(initialClassName);
      groupReset();
    }
  }, [open, initialClassName, groupReset]);

  async function submit(name: string) {
    const trimmed = name.trim();
    if (!trimmed || glyphIds.length < 2 || group.isPending) return;
    try {
      await group.mutateAsync({ glyph_ids: glyphIds, class_name: trimmed });
      onOpenChange(false);
    } catch {
      // error state shown inline; dialog stays open
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    void submit(className);
  }

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-slate-900/40" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[90vw] max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-slate-200 bg-white p-5 shadow-lg focus:outline-none">
          <Dialog.Title className="text-base font-semibold text-slate-800">
            Group {glyphs.length} glyphs
          </Dialog.Title>
          <Dialog.Description className="mt-2 text-sm text-slate-600">
            Merge the selected glyphs into one new manual glyph with the class
            name below. The source glyphs are removed.
          </Dialog.Description>

          {/* The preview honours the grid's binarized/original toggle, so it
              renders glyphs the same way the user is already seeing them. */}
          <PageImageProvider>
            <GroupPreview glyphs={glyphs} />
          </PageImageProvider>

          <form onSubmit={handleSubmit} className="mt-4 space-y-3">
            <div>
              <span className="mb-1 block text-xs font-medium text-slate-700">
                Class name
              </span>
              <ClassNameInput
                value={className}
                onChange={setClassName}
                options={classNames}
                onApply={(v) => void submit(v)}
              />
            </div>
            {group.isError && (
              <p className="text-xs text-red-600">
                {(group.error as Error)?.message}
              </p>
            )}
            <div className="flex justify-end gap-2 pt-2">
              <Dialog.Close asChild>
                <Button variant="ghost" disabled={group.isPending}>
                  Cancel
                </Button>
              </Dialog.Close>
              <Button
                type="submit"
                disabled={
                  group.isPending || !className.trim() || glyphIds.length < 2
                }
              >
                {group.isPending ? "Grouping…" : "Group"}
              </Button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

/**
 * Before/after strip: the source glyphs on the left, an arrow, and a
 * reconstruction of the merged glyph on the right.
 */
function GroupPreview({ glyphs }: { glyphs: GlyphDTO[] }) {
  return (
    <div className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-3">
      <div className="flex items-stretch gap-3">
        <div className="min-w-0 flex-1">
          <p className="mb-1 text-[11px] font-medium text-slate-500">
            Selected ({glyphs.length})
          </p>
          <div className="flex flex-wrap items-center gap-2">
            {glyphs.map((g) => (
              <div
                key={g.id}
                className="flex h-14 w-14 items-center justify-center overflow-hidden rounded border border-slate-200 bg-white"
                title={g.class_name}
              >
                <GlyphImage glyph={g} className="h-14 w-full object-contain" />
              </div>
            ))}
          </div>
        </div>
        <div className="flex shrink-0 items-center self-center text-2xl text-slate-300">
          →
        </div>
        <div className="shrink-0">
          <p className="mb-1 text-[11px] font-medium text-slate-500">Result</p>
          <div className="flex h-14 w-14 items-center justify-center overflow-hidden rounded border-2 border-green-400 bg-white">
            <MergedPreview
              glyphs={glyphs}
              className="h-14 w-full object-contain"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Canvas reconstruction of the grouped glyph, mirroring the backend's
 * `manual_group`: the result spans the union of every input's bounding box.
 *
 * * "original" — crop that union region out of the page image. This is
 *   pixel-identical to how {@link GlyphImage} will later render the grouped
 *   glyph, gap pixels included.
 * * "binarized" — OR the child masks onto a black canvas. The masks are
 *   white-on-black bilevel PNGs, so `globalCompositeOperation = "lighter"`
 *   takes the per-pixel max — exactly the bitwise-OR the backend performs.
 *   (The backend additionally fills the between-bbox gap from the full-page
 *   mask; that gap stays black here, a preview-only difference.)
 *
 * Falls back to the mask composite whenever the page image is unavailable.
 */
function MergedPreview({
  glyphs,
  className,
}: {
  glyphs: GlyphDTO[];
  className?: string;
}) {
  const mode = useUiStore((s) => s.glyphImageMode);
  const pageImg = usePageImageEl();
  const ref = useRef<HTMLCanvasElement | null>(null);

  const box = useMemo(() => {
    const ulx = Math.min(...glyphs.map((g) => g.ulx));
    const uly = Math.min(...glyphs.map((g) => g.uly));
    const lrx = Math.max(...glyphs.map((g) => g.ulx + g.ncols));
    const lry = Math.max(...glyphs.map((g) => g.uly + g.nrows));
    return { ulx, uly, ncols: lrx - ulx, nrows: lry - uly };
  }, [glyphs]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { ulx, uly, ncols, nrows } = box;

    if (mode === "original" && pageImg) {
      ctx.clearRect(0, 0, ncols, nrows);
      ctx.drawImage(pageImg, ulx, uly, ncols, nrows, 0, 0, ncols, nrows);
      return;
    }

    let alive = true;
    const sources = glyphs.map((g) => {
      const im = new Image();
      im.src = glyphDataUri(g);
      return { im, g };
    });
    void Promise.all(
      sources.map(({ im }) => im.decode().catch(() => undefined)),
    ).then(() => {
      if (!alive) return;
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, ncols, nrows);
      ctx.globalCompositeOperation = "lighter";
      for (const { im, g } of sources) {
        ctx.drawImage(im, g.ulx - ulx, g.uly - uly, g.ncols, g.nrows);
      }
      ctx.globalCompositeOperation = "source-over";
    });
    return () => {
      alive = false;
    };
  }, [box, mode, pageImg, glyphs]);

  return (
    <canvas
      ref={ref}
      width={box.ncols}
      height={box.nrows}
      aria-label="Grouped result preview"
      className={className}
    />
  );
}
