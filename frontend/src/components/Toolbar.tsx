import type { ExportSelection } from "@/api/sessions";
import { Button } from "@/components/ui/Button";
import { AlertCircleIcon } from "@/components/ui/icons";
import { useClassify } from "@/hooks/useClassify";
import { useComplete } from "@/hooks/useComplete";
import { useRebinarize } from "@/hooks/useRebinarize";
import { useUndoApply } from "@/hooks/useUpdateGlyphs";
import { type GlyphImageMode, useUiStore } from "@/store/uiStore";
import type { BinarizationMethod } from "@/types/api";
import { clsx } from "clsx";
import { useEffect, useRef, useState } from "react";

interface ToolbarProps {
  sessionId: string;
  glyphCount: number;
  trainingSize: number;
  /** Working neumes the user has labelled by hand (export option). */
  manualNeumeCount: number;
  /** Training glyphs sourced from a built-in preset (export option). */
  presetTrainingCount: number;
  /** Training glyphs sourced from an uploaded file (export option). */
  uploadedTrainingCount: number;
  binarizationMethod: BinarizationMethod;
}

const K_CHOICES = [1, 3, 5, 7] as const;

// Below this many training glyphs, kNN has too few examples to classify
// reliably, so we surface a warning next to the training-set count.
const SMALL_TRAINING_THRESHOLD = 10;

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
  manualNeumeCount,
  presetTrainingCount,
  uploadedTrainingCount,
  binarizationMethod,
}: ToolbarProps) {
  const complete = useComplete(sessionId);
  const classify = useClassify(sessionId);
  const rebinarize = useRebinarize(sessionId);
  const undoApply = useUndoApply(sessionId);
  const clearSession = useUiStore((s) => s.clearSession);
  const knnK = useUiStore((s) => s.knnK);
  const undoStack = useUiStore((s) => s.undoStack);
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
  // mask. No-op when the active method is re-clicked or a round is already
  // in flight. Labels, manual splits and manual groups are all kept — the
  // masks are re-derived per glyph bbox, so ids survive (see the hook).
  //
  // Rebinarising carries the prior auto labels forward verbatim, but they
  // were derived from the *old* masks — so chain a classify round to refresh
  // them from the new pixels, the same way changing k does. Skipped when the
  // training pool is too small for the current k, which is also the case the
  // backend rejects. `trainingSize` counts training-set glyphs, which the
  // re-ingest leaves untouched, so the closure value is still accurate after
  // the await.
  const handleMethodChange = async (method: BinarizationMethod) => {
    if (method === binarizationMethod || rebinarize.isPending) return;
    if (classify.isPending) return;
    try {
      await rebinarize.mutateAsync(method);
      if (isKAvailable(knnK)) await classify.mutateAsync(knnK);
    } catch {
      // Both mutations surface their own error state; catching here only
      // keeps the rejection from going unhandled.
    }
  };

  return (
    <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-2">
      <div className="flex items-baseline gap-3">
        <button
          type="button"
          onClick={() => clearSession()}
          title="Return to the main page"
          className="font-semibold text-slate-800 transition-colors hover:text-blue-600"
        >
          Interactive Classifier
        </button>
        <span className="text-sm text-slate-500">{glyphCount} glyphs</span>
      </div>
      <div className="flex items-center gap-2">
        <div
          className="flex items-center gap-1 rounded border border-slate-200 bg-slate-50 px-2 py-1"
          title="Page binarisation. Switching rebuilds glyph masks and re-runs classify so auto labels match the new masks. Manual labels, splits and groups are kept."
        >
          <span className="text-xs font-medium text-slate-600">Binarize</span>
          <div className="flex overflow-hidden rounded border border-slate-300">
            {BIN_METHODS.map(({ value, label }) => (
              <button
                key={value}
                type="button"
                onClick={() => handleMethodChange(value)}
                disabled={rebinarize.isPending || classify.isPending}
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
          <button
            type="button"
            onClick={() => classify.mutate(knnK)}
            disabled={classify.isPending || !isKAvailable(knnK)}
            title={
              trainingSize === 0
                ? "No training glyphs yet — apply at least one label first"
                : !isKAvailable(knnK)
                  ? `Needs at least ${knnK} training glyphs (have ${trainingSize})`
                  : "Re-run kNN classification with the current k and training set"
            }
            className="px-2 py-0.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60 bg-blue-600 text-white hover:bg-blue-700 rounded"
          >
            {classify.isPending ? "Classifying…" : "↺ Reclassify"}
          </button>
          <span
            className="text-xs text-slate-500"
            title="Number of glyphs in the training set used for kNN classification"
          >
            {trainingSize.toLocaleString()} training glyphs
          </span>
          {trainingSize < SMALL_TRAINING_THRESHOLD && <SmallTrainingWarning />}
        </div>
        <Button
          variant="secondary"
          onClick={() => undoApply.mutate()}
          disabled={undoStack.length === 0 || undoApply.isPending}
          title={
            undoStack.length > 0
              ? `Undo: ${undoStack[undoStack.length - 1].description}`
              : "Nothing to undo"
          }
        >
          {undoApply.isPending ? "Undoing…" : "Undo"}
        </Button>
        <Button variant="ghost" onClick={() => clearSession()}>
          New session
        </Button>
        <ExportMenu
          pending={complete.isPending}
          pageCount={glyphCount}
          manualNeumeCount={manualNeumeCount}
          presetTrainingCount={presetTrainingCount}
          uploadedTrainingCount={uploadedTrainingCount}
          onExport={(selection) => complete.mutate(selection)}
        />
      </div>
    </header>
  );
}

/**
 * Circled exclamation mark surfaced when the training set is small. Hovering
 * (or focusing) it reveals a notification warning that kNN may perform poorly
 * with so few training examples.
 */
function SmallTrainingWarning() {
  return (
    <span className="group relative inline-flex">
      <button
        type="button"
        aria-label="Small training set warning"
        aria-describedby="small-training-warning-tooltip"
        className="inline-flex h-4 w-4 items-center justify-center rounded-full text-amber-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
      >
        <AlertCircleIcon width={16} height={16} />
      </button>
      <span
        id="small-training-warning-tooltip"
        role="tooltip"
        className="pointer-events-none absolute right-0 top-full z-20 mt-1.5 w-56 rounded border border-amber-200 bg-amber-50 px-2.5 py-1.5 text-xs leading-snug text-amber-800 opacity-0 shadow-md transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
      >
        The training set is small (fewer than {SMALL_TRAINING_THRESHOLD}{" "}
        glyphs), so kNN classification may perform poorly.
      </span>
    </span>
  );
}

interface ExportMenuProps {
  pending: boolean;
  pageCount: number;
  manualNeumeCount: number;
  presetTrainingCount: number;
  uploadedTrainingCount: number;
  onExport: (selection: ExportSelection) => void;
}

// The four sections the export can fold into one GameraXML. `key` matches the
// ExportSelection field; `count` gates availability — a section with nothing
// in it is disabled (and force-unchecked).
type ExportOptionKey = keyof ExportSelection;

/**
 * Checkbox export menu: the caret reveals one checkbox per exportable section
 * (whole page, manual neumes, uploaded training, preset training). The user
 * ticks any combination and hits Export; the selected sections are
 * concatenated into a single GameraXML. Closes on outside click or Escape.
 */
function ExportMenu({
  pending,
  pageCount,
  manualNeumeCount,
  presetTrainingCount,
  uploadedTrainingCount,
  onExport,
}: ExportMenuProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  // Default to the common case: the neumes labelled by hand on this page.
  const [selection, setSelection] = useState<Record<ExportOptionKey, boolean>>({
    page: false,
    manualNeumes: true,
    presetTraining: false,
    uploadedTraining: false,
  });

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

  const options: {
    key: ExportOptionKey;
    label: string;
    count: number;
    unit: string;
  }[] = [
    {
      key: "page",
      label: "Whole annotated page",
      count: pageCount,
      unit: "glyphs",
    },
    {
      key: "manualNeumes",
      label: "Manual neumes (this page)",
      count: manualNeumeCount,
      unit: "hand-labelled neumes",
    },
    {
      key: "uploadedTraining",
      label: "Uploaded training set",
      count: uploadedTrainingCount,
      unit: "uploaded glyphs",
    },
    {
      key: "presetTraining",
      label: "Preset training set",
      count: presetTrainingCount,
      unit: "preset glyphs",
    },
  ];

  const toggle = (key: ExportOptionKey) =>
    setSelection((prev) => ({ ...prev, [key]: !prev[key] }));

  // A section is only selectable if it has content; ignore ticks that survive
  // on a now-empty section so we never post an empty-section flag.
  const effective: ExportSelection = {
    page: selection.page && pageCount > 0,
    manualNeumes: selection.manualNeumes && manualNeumeCount > 0,
    presetTraining: selection.presetTraining && presetTrainingCount > 0,
    uploadedTraining: selection.uploadedTraining && uploadedTrainingCount > 0,
  };
  const anySelected = Object.values(effective).some(Boolean);

  const handleExport = () => {
    if (!anySelected) return;
    setOpen(false);
    onExport(effective);
  };

  return (
    <div ref={ref} className="relative">
      <Button onClick={() => setOpen((v) => !v)} disabled={pending}>
        {pending ? "Exporting…" : "Complete & Export ▾"}
      </Button>
      {open && (
        <div className="absolute right-0 z-10 mt-1 w-80 overflow-hidden rounded border border-slate-200 bg-white shadow-lg">
          <div className="border-b border-slate-100 px-3 py-2 text-xs font-medium text-slate-500">
            Include in export
          </div>
          <div className="p-1">
            {options.map(({ key, label, count, unit }) => {
              const disabled = count === 0;
              return (
                <label
                  key={key}
                  className={clsx(
                    "flex cursor-pointer items-start gap-2 rounded px-2 py-1.5 text-sm hover:bg-slate-100",
                    disabled &&
                      "cursor-not-allowed opacity-50 hover:bg-transparent",
                  )}
                >
                  <input
                    type="checkbox"
                    className="mt-0.5"
                    checked={effective[key]}
                    disabled={disabled}
                    onChange={() => toggle(key)}
                  />
                  <span>
                    <span className="font-medium text-slate-700">{label}</span>
                    <span className="block text-xs text-slate-500">
                      {disabled
                        ? `No ${unit}`
                        : `${count.toLocaleString()} ${unit}`}
                    </span>
                  </span>
                </label>
              );
            })}
          </div>
          <div className="border-t border-slate-100 p-2">
            <Button
              className="w-full"
              onClick={handleExport}
              disabled={!anySelected}
              title={
                anySelected
                  ? undefined
                  : "Select at least one section to export"
              }
            >
              Export selected
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
