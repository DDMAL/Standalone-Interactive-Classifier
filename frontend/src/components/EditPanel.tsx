import { ClassNameInput } from "@/components/ClassNameInput";
import { Button } from "@/components/ui/Button";
import { useClassify } from "@/hooks/useClassify";
import { sessionKey } from "@/hooks/useSession";
import { useUpdateGlyph } from "@/hooks/useUpdateGlyph";
import { formatConfidence, glyphDataUri } from "@/lib/format";
import { useUiStore } from "@/store/uiStore";
import { CATEGORY_ORDER, type GlyphCategory, type GlyphDTO } from "@/types/api";
import { useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";

interface EditPanelProps {
  sessionId: string;
  glyph: GlyphDTO;
  classNames: string[];
}

// Mounted with key={glyph.id} by SessionView, so local state resets per glyph.
export function EditPanel({ sessionId, glyph, classNames }: EditPanelProps) {
  const [className, setClassName] = useState(glyph.class_name);
  const updateGlyph = useUpdateGlyph(sessionId);
  const classify = useClassify(sessionId);
  const queryClient = useQueryClient();
  const selectGlyph = useUiStore((s) => s.selectGlyph);

  const pending = updateGlyph.isPending || classify.isPending;
  const isNeume = glyph.category === "Neumes";

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const name = className.trim();
    if (!name) return;
    await updateGlyph.mutateAsync({
      glyphId: glyph.id,
      patch: { class_name: name, id_state_manual: true },
    });
    await classify.mutateAsync(1);
    queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
  }

  // Move the glyph to another MOTHRA category. The backend resets its
  // label on the move, so we just refetch the session and keep it selected.
  async function moveToCategory(target: GlyphCategory) {
    await updateGlyph.mutateAsync({
      glyphId: glyph.id,
      patch: { category: target },
    });
    queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
  }

  return (
    <aside className="w-72 shrink-0 overflow-auto border-l border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-800">Edit glyph</h2>
        <Button
          variant="ghost"
          onClick={() => selectGlyph(null)}
          className="px-2 py-0.5"
        >
          ✕
        </Button>
      </div>

      <div className="mb-3 flex items-center justify-center rounded border border-slate-200 bg-slate-50 p-3">
        <img
          src={glyphDataUri(glyph)}
          alt={glyph.class_name}
          className="max-h-32 object-contain"
        />
      </div>

      <dl className="mb-4 space-y-1 text-xs text-slate-600">
        <div className="flex justify-between">
          <dt>Class</dt>
          <dd>{glyph.category}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Confidence</dt>
          <dd>{formatConfidence(glyph.confidence)}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Source</dt>
          <dd>{glyph.id_state_manual ? "Manual" : "Auto"}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Position</dt>
          <dd>
            ({glyph.ulx}, {glyph.uly})
          </dd>
        </div>
        <div className="flex justify-between">
          <dt>Size</dt>
          <dd>
            {glyph.ncols}×{glyph.nrows}
          </dd>
        </div>
      </dl>

      {isNeume ? (
        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <span className="mb-1 block text-xs font-medium text-slate-700">
              Class name
            </span>
            <ClassNameInput
              value={className}
              onChange={setClassName}
              options={classNames}
            />
          </div>
          {(updateGlyph.isError || classify.isError) && (
            <p className="text-xs text-red-600">
              {((updateGlyph.error ?? classify.error) as Error)?.message}
            </p>
          )}
          <Button
            type="submit"
            disabled={pending || !className.trim()}
            className="w-full"
          >
            {pending ? "Applying…" : "Apply & reclassify"}
          </Button>
        </form>
      ) : (
        <p className="rounded border border-slate-200 bg-slate-50 p-2 text-xs text-slate-500">
          Glyphs in “{glyph.category}” are not classified. Move it to “Neumes”
          to assign a class name.
        </p>
      )}

      <div className="mt-4 border-t border-slate-200 pt-3">
        <span className="mb-2 block text-xs font-medium text-slate-700">
          Move to class
        </span>
        <div className="flex flex-wrap gap-2">
          {CATEGORY_ORDER.filter((c) => c !== glyph.category).map((target) => (
            <Button
              key={target}
              variant="secondary"
              disabled={pending}
              onClick={() => moveToCategory(target)}
              className="flex-1 whitespace-nowrap px-2 py-1 text-xs"
            >
              → {target}
            </Button>
          ))}
        </div>
      </div>
    </aside>
  );
}
