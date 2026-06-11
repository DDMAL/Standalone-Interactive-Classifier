import { ClassNameInput } from "@/components/ClassNameInput";
import { GroupDialog } from "@/components/GroupDialog";
import { Button } from "@/components/ui/Button";
import { useUpdateGlyphs } from "@/hooks/useUpdateGlyphs";
import { isEditableTarget } from "@/lib/keymap";
import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

interface MultiEditPanelProps {
  sessionId: string;
  selectedGlyphs: GlyphDTO[];
  classNames: string[];
}

/**
 * Multi-selection editor: shown when `selectedGlyphIds.size >= 2`. Applies
 * one class name to every Neume member, groups them into a new glyph, or
 * bulk-deletes. Non-Neume members are filtered out before the apply mutation.
 */
export function MultiEditPanel({
  sessionId,
  selectedGlyphs,
  classNames,
}: MultiEditPanelProps) {
  const clearSelection = useUiStore((s) => s.clearSelection);
  const softDeleteGlyphs = useUiStore((s) => s.softDeleteGlyphs);
  const updateGlyphs = useUpdateGlyphs(sessionId);

  const { neumeIds, neumeCount, nonNeumeCount, dominant } = useMemo(() => {
    const neumes = selectedGlyphs.filter((g) => g.category === "Neumes");
    const counts = new Map<string, number>();
    for (const g of neumes) {
      const key = g.class_name.trim();
      if (!key) continue;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    let best = "";
    let bestN = 0;
    for (const [k, n] of counts) {
      if (n > bestN) {
        best = k;
        bestN = n;
      }
    }
    return {
      neumeIds: neumes.map((g) => g.id),
      neumeCount: neumes.length,
      nonNeumeCount: selectedGlyphs.length - neumes.length,
      dominant: best,
    };
  }, [selectedGlyphs]);

  const [className, setClassName] = useState(dominant);
  const [groupOpen, setGroupOpen] = useState(false);

  // Reseed the input when the selection (and therefore dominant) changes,
  // but only when the user hasn't typed something else of their own. We
  // detect "untouched" by comparing to the previous dominant — if className
  // still equals the old dominant, the user hasn't edited it.
  const prevDominant = useRef(dominant);
  useEffect(() => {
    if (className === prevDominant.current) setClassName(dominant);
    prevDominant.current = dominant;
  }, [dominant, className]);

  const totalCount = selectedGlyphs.length;
  const pending = updateGlyphs.isPending;

  const applyRef = useRef<() => Promise<void>>(() => Promise.resolve());

  async function applyToMany(override?: string) {
    const name = (override ?? className).trim();
    if (!name || neumeIds.length === 0 || pending) return;
    if (override !== undefined && override !== className) {
      setClassName(name);
    }
    await updateGlyphs.mutateAsync({
      glyphIds: neumeIds,
      patch: { class_name: name, id_state_manual: true },
    });
  }
  applyRef.current = () => applyToMany();

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    void applyToMany();
  }

  function handleDelete() {
    softDeleteGlyphs(selectedGlyphs.map((g) => g.id));
  }

  // Enter-from-anywhere mirrors SingleEditor's approach: window-level
  // listener gated by isEditableTarget so the autocomplete keeps its own
  // Enter handling.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== "Enter") return;
      if (isEditableTarget(e.target)) return;
      e.preventDefault();
      void applyRef.current();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  // Cmd/Ctrl+G opens GroupDialog; Cmd/Ctrl+E focuses the class input. Both
  // short-circuit while typing in an input.
  const inputRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (isEditableTarget(e.target)) return;
      if (!(e.metaKey || e.ctrlKey)) return;
      const key = e.key.toLowerCase();
      if (key === "g" && neumeIds.length >= 2) {
        e.preventDefault();
        setGroupOpen(true);
      } else if (key === "e") {
        e.preventDefault();
        const input = inputRef.current?.querySelector("input");
        input?.focus();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [neumeIds.length]);

  const lastResult = updateGlyphs.data;

  return (
    <aside className="w-72 shrink-0 overflow-auto border-l border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-800">Multi-edit</h2>
        <Button
          variant="ghost"
          onClick={() => clearSelection()}
          className="px-2 py-0.5"
        >
          ✕
        </Button>
      </div>

      <p className="mb-3 rounded border border-slate-200 bg-slate-50 p-2 text-xs text-slate-600">
        <span className="font-semibold">{totalCount} selected</span> ·{" "}
        {neumeCount} Neumes, {nonNeumeCount} non-Neumes
      </p>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div ref={inputRef}>
          <span className="mb-1 block text-xs font-medium text-slate-700">
            Class name
          </span>
          <ClassNameInput
            value={className}
            onChange={setClassName}
            options={classNames}
            onApply={(v) => void applyToMany(v)}
          />
        </div>
        {nonNeumeCount > 0 && (
          <p className="text-xs text-amber-700">
            Skipping {nonNeumeCount} non-Neume glyph
            {nonNeumeCount === 1 ? "" : "s"}.
          </p>
        )}
        {updateGlyphs.isError && (
          <p className="text-xs text-red-600">
            {(updateGlyphs.error as Error)?.message}
          </p>
        )}
        {lastResult && lastResult.failed.length > 0 && (
          <p className="text-xs text-amber-700">
            {lastResult.applied} of{" "}
            {lastResult.applied + lastResult.failed.length} applied.
          </p>
        )}
        <Button
          type="submit"
          disabled={pending || !className.trim() || neumeIds.length === 0}
          className="w-full"
        >
          {pending
            ? "Applying…"
            : `Apply to ${neumeCount} Neume${neumeCount === 1 ? "" : "s"}`}
        </Button>
      </form>

      <div className="mt-4 border-t border-slate-200 pt-3">
        <Button
          variant="secondary"
          onClick={() => setGroupOpen(true)}
          disabled={pending || neumeIds.length < 2}
          className="w-full"
        >
          Group as new glyph
        </Button>
        <p className="mt-2 text-xs text-slate-400">
          Merges the selected Neumes into one manual glyph.{" "}
          <kbd className="rounded border border-slate-300 bg-slate-50 px-1">
            Cmd/Ctrl+G
          </kbd>
        </p>
      </div>

      <div className="mt-4 border-t border-slate-200 pt-3">
        <Button
          variant="secondary"
          onClick={handleDelete}
          disabled={pending}
          className="w-full border-red-300 text-red-700 hover:bg-red-50"
        >
          Delete {totalCount} glyph{totalCount === 1 ? "" : "s"}
        </Button>
        <p className="mt-2 text-xs text-slate-400">
          Moves to the Deleted section; committed on export. Press{" "}
          <kbd className="rounded border border-slate-300 bg-slate-50 px-1">
            Esc
          </kbd>{" "}
          to clear.
        </p>
      </div>

      <GroupDialog
        open={groupOpen}
        onOpenChange={setGroupOpen}
        sessionId={sessionId}
        glyphIds={neumeIds}
        initialClassName={dominant || className}
        classNames={classNames}
      />
    </aside>
  );
}
