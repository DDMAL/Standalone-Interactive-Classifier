import { ClassNameInput } from "@/components/ClassNameInput";
import { Button } from "@/components/ui/Button";
import { useGroup } from "@/hooks/useGroup";
import * as Dialog from "@radix-ui/react-dialog";
import { type FormEvent, useEffect, useState } from "react";

interface GroupDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sessionId: string;
  glyphIds: string[];
  /** Class name seeded into the input — usually the dominant class in the
   *  multi-selection. */
  initialClassName: string;
  classNames: string[];
}

/**
 * Dialog that wraps `POST /sessions/{id}/group`: takes a multi-selection and
 * a user-supplied class name, merges into one new manual glyph on submit.
 */
export function GroupDialog({
  open,
  onOpenChange,
  sessionId,
  glyphIds,
  initialClassName,
  classNames,
}: GroupDialogProps) {
  const [className, setClassName] = useState(initialClassName);
  const group = useGroup(sessionId);
  const groupReset = group.reset;

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
            Group {glyphIds.length} glyphs
          </Dialog.Title>
          <Dialog.Description className="mt-2 text-sm text-slate-600">
            Merge the selected glyphs into one new manual glyph with the class
            name below. The source glyphs are removed.
          </Dialog.Description>
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
