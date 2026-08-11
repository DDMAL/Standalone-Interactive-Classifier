import { ClassNameInput } from "@/components/ClassNameInput";
import { GlyphImage } from "@/components/GlyphImage";
import { Button } from "@/components/ui/Button";
import { useModalGuard } from "@/hooks/useModalGuard";
import { useUpdateGlyphsPerGlyph } from "@/hooks/useUpdateGlyphs";
import { isEditableTarget } from "@/lib/keymap";
import { type UndoEntry, useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import * as Dialog from "@radix-ui/react-dialog";
import { clsx } from "clsx";
import {
  type KeyboardEvent as ReactKeyboardEvent,
  useEffect,
  useRef,
  useState,
} from "react";

interface BatchConfirmDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sessionId: string;
  /** Only Neume glyphs — non-Neumes are filtered out by the caller. */
  glyphs: GlyphDTO[];
  classNames: string[];
}

type GroupedSnapshot = [string, GlyphDTO[]][];

function normalizeClassName(className: string): string {
  return className.trim();
}

function isUnclassifiedClassName(className: string): boolean {
  return normalizeClassName(className).toUpperCase() === "UNCLASSIFIED";
}

function buildGroups(
  glyphs: GlyphDTO[],
  assignments: Map<string, string>,
): GroupedSnapshot {
  const map = new Map<string, GlyphDTO[]>();
  for (const glyph of glyphs) {
    const cls = normalizeClassName(
      assignments.get(glyph.id) ?? glyph.class_name,
    );
    const groupedClassName = cls || "UNCLASSIFIED";
    if (!map.has(groupedClassName)) map.set(groupedClassName, []);
    map.get(groupedClassName)?.push(glyph);
  }
  return [...map.entries()].sort(([a], [b]) => {
    if (a === "UNCLASSIFIED") return 1;
    if (b === "UNCLASSIFIED") return -1;
    return a.localeCompare(b);
  });
}

/**
 * Verification mosaic dialog for "Apply each in own class". Shows every
 * selected Neume with its current class name (editable). On confirm, each
 * glyph is committed as manual with its individual class, then classify runs.
 */
export function BatchConfirmDialog({
  open,
  onOpenChange,
  sessionId,
  glyphs,
  classNames,
}: BatchConfirmDialogProps) {
  const initAssignments = () =>
    new Map(glyphs.map((g) => [g.id, g.class_name]));
  const [assignments, setAssignments] =
    useState<Map<string, string>>(initAssignments);
  const [groupedSnapshot, setGroupedSnapshot] = useState<GroupedSnapshot>(() =>
    buildGroups(glyphs, initAssignments()),
  );
  const updatePerGlyph = useUpdateGlyphsPerGlyph(sessionId);
  const resetMutation = updatePerGlyph.reset;

  useModalGuard(open);

  // Snapshot both assignments and groups when the dialog opens so we always
  // see fresh kNN predictions and the group layout stays stable during edits.
  // biome-ignore lint/correctness/useExhaustiveDependencies: snapshot glyphs at open time only; stale deps intentional
  useEffect(() => {
    if (!open) return;
    const init = new Map(glyphs.map((g) => [g.id, g.class_name]));
    setAssignments(init);
    setGroupedSnapshot(buildGroups(glyphs, init));
    resetMutation();
  }, [open, resetMutation]);

  function setAssignment(glyphId: string, className: string) {
    setAssignments((prev) => {
      const next = new Map(prev);
      next.set(glyphId, className);
      return next;
    });
  }

  const pushUndo = useUiStore((s) => s.pushUndo);

  async function handleConfirm() {
    const items = glyphs
      .map((g) => ({
        id: g.id,
        class_name: normalizeClassName(assignments.get(g.id) ?? g.class_name),
      }))
      .filter(
        ({ class_name }) => class_name && !isUnclassifiedClassName(class_name),
      );
    if (items.length === 0) {
      onOpenChange(false);
      return;
    }
    const itemIds = new Set(items.map(({ id }) => id));
    const undoEntry: UndoEntry = {
      description: `Confirm ${items.length} neume${items.length === 1 ? "" : "s"} in own classes`,
      snapshots: glyphs
        .filter((g) => itemIds.has(g.id))
        .map((g) => ({
          id: g.id,
          class_name: g.class_name,
          id_state_manual: g.id_state_manual,
        })),
    };
    try {
      const result = await updatePerGlyph.mutateAsync({ assignments: items });
      if (result.failed.length === 0) {
        pushUndo(undoEntry);
        onOpenChange(false);
      }
    } catch {
      // error shown inline; dialog stays open
    }
  }

  const confirmButtonRef = useRef<HTMLButtonElement>(null);

  function handleKeyDown(e: ReactKeyboardEvent<HTMLDivElement>) {
    if (e.key !== "Enter") return;
    if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey) return;
    if (isEditableTarget(e.target)) return;
    if (pending) return;
    e.preventDefault();
    void handleConfirm();
  }

  const pending = updatePerGlyph.isPending;
  const glyphImageMode = useUiStore((s) => s.glyphImageMode);
  const setGlyphImageMode = useUiStore((s) => s.setGlyphImageMode);

  // Count unclassified from live assignments so the warning stays accurate.
  const unclassifiedCount = glyphs.filter((g) => {
    const cls = normalizeClassName(assignments.get(g.id) ?? g.class_name);
    return !cls || isUnclassifiedClassName(cls);
  }).length;
  const lastResult = updatePerGlyph.data;

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-slate-900/40" />
        <Dialog.Content
          onKeyDown={handleKeyDown}
          className="fixed left-1/2 top-1/2 z-50 w-[92vw] max-w-3xl -translate-x-1/2 -translate-y-1/2 rounded-lg border border-slate-200 bg-white p-5 shadow-lg focus:outline-none"
        >
          <div className="flex items-start justify-between gap-4">
            <div>
              <Dialog.Title className="text-base font-semibold text-slate-800">
                Confirm {glyphs.length} neume{glyphs.length === 1 ? "" : "s"} in
                their own classes
              </Dialog.Title>
              <Dialog.Description className="mt-1 text-sm text-slate-600">
                Neumes are grouped by class. Edit any label to reassign before
                confirming.
              </Dialog.Description>
            </div>
            <div className="flex items-center gap-1 rounded border border-slate-200 bg-slate-50 px-2 py-1">
              <span className="text-xs font-medium text-slate-600">Glyphs</span>
              <div className="flex overflow-hidden rounded border border-slate-300">
                {(
                  [
                    { value: "binarized", label: "Binarized" },
                    { value: "original", label: "Original" },
                  ] as const
                ).map(({ value, label }) => (
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
          </div>

          <div className="mt-4 max-h-[55vh] space-y-4 overflow-y-auto rounded border border-slate-100 p-3">
            {groupedSnapshot.map(([cls, groupGlyphs]) => {
              const isUnclassifiedGroup = cls === "UNCLASSIFIED";
              return (
                <div key={cls}>
                  <div className="mb-2 flex items-center gap-2 border-b border-slate-100 pb-1">
                    <span
                      className={`text-sm font-semibold ${isUnclassifiedGroup ? "text-amber-700" : "text-slate-700"}`}
                    >
                      {cls}
                    </span>
                    <span className="text-xs text-slate-400">
                      ({groupGlyphs.length})
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-3">
                    {groupGlyphs.map((glyph) => {
                      const glyphCls =
                        assignments.get(glyph.id) ?? glyph.class_name;
                      const isUnclassified =
                        !glyphCls.trim() || glyphCls === "UNCLASSIFIED";
                      return (
                        <div
                          key={glyph.id}
                          className="flex w-32 flex-col gap-1.5"
                        >
                          <div
                            className={`flex h-16 w-full items-center justify-center overflow-hidden rounded border bg-slate-50 ${
                              isUnclassified
                                ? "border-amber-300"
                                : "border-slate-200"
                            }`}
                          >
                            <GlyphImage
                              glyph={glyph}
                              className="h-16 w-full object-contain"
                            />
                          </div>
                          <ClassNameInput
                            value={glyphCls}
                            onChange={(v) => setAssignment(glyph.id, v)}
                            options={classNames}
                            onApply={(v) => {
                              setAssignment(glyph.id, v);
                              confirmButtonRef.current?.focus();
                            }}
                          />
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          {unclassifiedCount > 0 && (
            <p className="mt-3 text-xs text-amber-700">
              {unclassifiedCount} neume{unclassifiedCount === 1 ? "" : "s"}{" "}
              still UNCLASSIFIED and will be skipped.
            </p>
          )}

          {updatePerGlyph.isError && (
            <p className="mt-2 text-xs text-red-600">
              {(updatePerGlyph.error as Error)?.message}
            </p>
          )}
          {lastResult && lastResult.failed.length > 0 && (
            <p className="mt-2 text-xs text-amber-700">
              {lastResult.applied} of{" "}
              {lastResult.applied + lastResult.failed.length} applied.
            </p>
          )}

          <div className="mt-4 flex justify-end gap-2 border-t border-slate-200 pt-3">
            <Dialog.Close asChild>
              <Button variant="ghost" disabled={pending}>
                Cancel
              </Button>
            </Dialog.Close>
            <Button
              ref={confirmButtonRef}
              onClick={() => void handleConfirm()}
              disabled={pending}
            >
              {pending
                ? "Confirming…"
                : `Confirm ${glyphs.length} neume${glyphs.length === 1 ? "" : "s"}`}
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
