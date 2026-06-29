import { Button } from "@/components/ui/Button";
import { useClassify } from "@/hooks/useClassify";
import { useComplete } from "@/hooks/useComplete";
import { useRebinarize } from "@/hooks/useRebinarize";
import { useSave } from "@/hooks/useSave";
import { type GlyphImageMode, useUiStore } from "@/store/uiStore";
import type { BinarizationMethod } from "@/types/api";
import { clsx } from "clsx";
import { useEffect, useRef, useState } from "react";

interface ToolbarProps {
  sessionId: string;
  glyphCount: number;
  trainingSize: number;
  binarizationMethod: BinarizationMethod;
}

const K_CHOICES = [1, 3, 5, 7] as const;

const BIN_METHODS: { value: BinarizationMethod; label: string }[] = [
  { value: "global", label: "Global" },
  { value: "otsu", label: "Otsu" },
  { value: "sauvola", label: "Sauvola" },
];

const GLYPH_VIEWS: { value: GlyphImageMode; label: string }[] = [
  { value: "binarized", label: "Binarized" },
  { value: "original", label: "Original" },
];

export function Toolbar({
  sessionId,
  glyphCount,
  trainingSize,
  binarizationMethod,
}: ToolbarProps) {
  const save = useSave(sessionId);
  const complete = useComplete(sessionId);
  const classify = useClassify(sessionId);
  const rebinarize = useRebinarize(sessionId);
  const clearSession = useUiStore((s) => s.clearSession);
  const knnK = useUiStore((s) => s.knnK);
  const setKnnK = useUiStore((s) => s.setKnnK);
  const glyphImageMode = useUiStore((s) => s.glyphImageMode);
  const setGlyphImageMode = useUiStore((s) => s.setGlyphImageMode);

  // A k value is only meaningful when the training set has at least k
  // examples — kNN needs k neighbours to vote on. Higher k values become
  // (un)available as the training set grows or shrinks.
  const isKAvailable = (k: number) => trainingSize >= k;

  // If the selected k outgrows the training set, fall back to the lowest
  // available k value so we never ask the classifier for more neighbours
  // than it has. When the pool is empty (the starting state, no training
  // set selected), no k is available and we settle on k=1.
  useEffect(() => {
    if (trainingSize < knnK) {
      const fallback = K_CHOICES.find((k) => trainingSize >= k) ?? K_CHOICES[0];
      if (fallback !== knnK) setKnnK(fallback);
    }
  }, [trainingSize, knnK, setKnnK]);

  // Changing k re-runs the classification stage with the new neighbour
  // count. No-op when the same k is clicked, while a classify is in flight,
  // or when the training set is too small for that k, to avoid
  // redundant/concurrent/invalid rounds.
  const handleKChange = (k: number) => {
    if (k === knnK || classify.isPending || !isKAvailable(k)) return;
    setKnnK(k);
    classify.mutate(k);
  };

  // Switching the method re-binarises the page and rebuilds every glyph
  // mask. No-op when the active method is re-clicked or a switch is in
  // flight. Manual groups/splits reset; labels are kept (see the hook).
  const handleMethodChange = (method: BinarizationMethod) => {
    if (method === binarizationMethod || rebinarize.isPending) return;
    rebinarize.mutate(method);
  };

  return (
    <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-2">
      <div className="flex items-baseline gap-3">
        <span className="font-semibold text-slate-800">
          Interactive Classifier
        </span>
        <span className="text-sm text-slate-500">{glyphCount} glyphs</span>
      </div>
      <div className="flex items-center gap-2">
        <div
          className="flex items-center gap-1 rounded border border-slate-200 bg-slate-50 px-2 py-1"
          title="Page binarisation. Switching rebuilds glyph masks; manual groups/splits reset, labels are kept. Re-run classify to refresh auto labels."
        >
          <span className="text-xs font-medium text-slate-600">Binarize</span>
          <div className="flex overflow-hidden rounded border border-slate-300">
            {BIN_METHODS.map(({ value, label }) => (
              <button
                key={value}
                type="button"
                onClick={() => handleMethodChange(value)}
                disabled={rebinarize.isPending}
                className={clsx(
                  "px-2 py-0.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60",
                  value === binarizationMethod
                    ? "bg-blue-600 text-white"
                    : "bg-white text-slate-700 hover:bg-slate-100",
                )}
              >
                {label}
              </button>
            ))}
          </div>
          {rebinarize.isPending && (
            <span className="text-xs text-slate-500">Re-binarizing…</span>
          )}
        </div>
        <div
          className="flex items-center gap-1 rounded border border-slate-200 bg-slate-50 px-2 py-1"
          title="What the glyph tiles show: the binarized foreground mask, or the original page crop. Display-only; does not change the underlying glyph data."
        >
          <span className="text-xs font-medium text-slate-600">Glyphs</span>
          <div className="flex overflow-hidden rounded border border-slate-300">
            {GLYPH_VIEWS.map(({ value, label }) => (
              <button
                key={value}
                type="button"
                onClick={() => setGlyphImageMode(value)}
                className={clsx(
                  "px-2 py-0.5 text-xs font-medium transition-colors",
                  value === glyphImageMode
                    ? "bg-blue-600 text-white"
                    : "bg-white text-slate-700 hover:bg-slate-100",
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <div
          className="flex items-center gap-1 rounded border border-slate-200 bg-slate-50 px-2 py-1"
          title="Neighbour count for kNN classification"
        >
          <span className="text-xs font-medium text-slate-600">k</span>
          <div className="flex overflow-hidden rounded border border-slate-300">
            {K_CHOICES.map((k) => {
              const available = isKAvailable(k);
              return (
                <button
                  key={k}
                  type="button"
                  onClick={() => handleKChange(k)}
                  disabled={classify.isPending || !available}
                  title={
                    available
                      ? undefined
                      : `Needs at least ${k} training glyphs (have ${trainingSize})`
                  }
                  className={clsx(
                    "px-2 py-0.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60",
                    k === knnK
                      ? "bg-blue-600 text-white"
                      : "bg-white text-slate-700 hover:bg-slate-100",
                  )}
                >
                  {k}
                </button>
              );
            })}
          </div>
          <span
            className="text-xs text-slate-500"
            title="Number of glyphs in the training set used for kNN classification"
          >
            {trainingSize.toLocaleString()} training glyphs
          </span>
        </div>
        <Button variant="ghost" onClick={clearSession}>
          New session
        </Button>
        <Button
          variant="secondary"
          onClick={() => save.mutate()}
          disabled={save.isPending}
        >
          {save.isPending ? "Saving…" : "Save"}
        </Button>
        <ExportMenu
          pending={complete.isPending}
          trainingSize={trainingSize}
          onExport={(includeTraining) => complete.mutate(includeTraining)}
        />
      </div>
    </header>
  );
}

interface ExportMenuProps {
  pending: boolean;
  trainingSize: number;
  onExport: (includeTraining: boolean) => void;
}

/**
 * Split-style export control: clicking the caret reveals a choice between
 * exporting just this page and exporting this page folded into the whole
 * training set. The menu closes on outside click or Escape.
 */
function ExportMenu({ pending, trainingSize, onExport }: ExportMenuProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const choose = (includeTraining: boolean) => {
    setOpen(false);
    onExport(includeTraining);
  };

  return (
    <div ref={ref} className="relative">
      <Button onClick={() => setOpen((v) => !v)} disabled={pending}>
        {pending ? "Exporting…" : "Complete & Export ▾"}
      </Button>
      {open && (
        <div className="absolute right-0 z-10 mt-1 w-72 overflow-hidden rounded border border-slate-200 bg-white shadow-lg">
          <button
            type="button"
            onClick={() => choose(false)}
            className="block w-full px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-100"
          >
            <span className="font-medium">Export this page</span>
            <span className="block text-xs text-slate-500">
              GameraXML for the current page only
            </span>
          </button>
          <button
            type="button"
            onClick={() => choose(true)}
            disabled={trainingSize === 0}
            title={
              trainingSize === 0
                ? "No training set loaded for this session"
                : undefined
            }
            className="block w-full border-t border-slate-100 px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent"
          >
            <span className="font-medium">Export this page + training set</span>
            <span className="block text-xs text-slate-500">
              One GameraXML combining this page with all{" "}
              {trainingSize.toLocaleString()} training glyphs
            </span>
          </button>
        </div>
      )}
    </div>
  );
}
