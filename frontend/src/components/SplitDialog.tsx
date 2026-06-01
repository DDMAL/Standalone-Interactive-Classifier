import { Button } from "@/components/ui/Button";
import { glyphDataUri } from "@/lib/format";
import type { GlyphDTO } from "@/types/api";
import * as Dialog from "@radix-ui/react-dialog";

interface SplitDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  glyph: GlyphDTO;
}

/**
 * Placeholder for the future "split one bbox into many" flow sketched in
 * PLAN.md. The endpoint and backend cropping aren't built yet — this
 * dialog reserves the entry point in the UI and previews the source glyph
 * so users can see what the eventual flow will operate on.
 */
export function SplitDialog({ open, onOpenChange, glyph }: SplitDialogProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-slate-900/40" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[90vw] max-w-lg -translate-x-1/2 -translate-y-1/2 rounded-lg border border-slate-200 bg-white p-5 shadow-lg focus:outline-none">
          <Dialog.Title className="text-base font-semibold text-slate-800">
            Split glyph
          </Dialog.Title>
          <Dialog.Description className="mt-2 text-sm text-slate-600">
            Cut this glyph into multiple replacement glyphs by drawing sub-rects
            over its image. Not yet implemented — the backend split endpoint
            ships in a later phase.
          </Dialog.Description>

          <div className="mt-4 flex items-center justify-center rounded border border-dashed border-slate-300 bg-slate-50 p-4">
            <img
              src={glyphDataUri(glyph)}
              alt={glyph.class_name}
              className="max-h-40 object-contain"
            />
          </div>

          <dl className="mt-3 space-y-1 text-xs text-slate-600">
            <div className="flex justify-between">
              <dt>Class</dt>
              <dd>{glyph.class_name || "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt>Size</dt>
              <dd>
                {glyph.ncols}×{glyph.nrows}
              </dd>
            </div>
          </dl>

          <div className="mt-5 flex justify-end gap-2">
            <Dialog.Close asChild>
              <Button variant="ghost">Close</Button>
            </Dialog.Close>
            <Button disabled>Split (coming soon)</Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
