import { ClassNameInput } from "@/components/ClassNameInput";
import { Button } from "@/components/ui/Button";
import { useClassify } from "@/hooks/useClassify";
import { sessionKey } from "@/hooks/useSession";
import { useUpdateGlyph } from "@/hooks/useUpdateGlyph";
import { formatConfidence, glyphDataUri } from "@/lib/format";
import { isEditableTarget } from "@/lib/keymap";
import { useUiStore } from "@/store/uiStore";
import { CATEGORY_ORDER, type GlyphCategory, type GlyphDTO } from "@/types/api";
import { useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useEffect, useRef, useState } from "react";

interface EditPanelProps {
  sessionId: string;
  primaryGlyph: GlyphDTO | null;
  selectionSize: number;
  classNames: string[];
}

// Branches on selection size: 1 → Phase A editor, ≥2 → multi-selection
// placeholder. Mounted with a key derived from selection mode + primary id,
// so local state resets on every transition.
export function EditPanel({
  sessionId,
  primaryGlyph,
  selectionSize,
  classNames,
}: EditPanelProps) {
  if (selectionSize >= 2) {
    return <MultiSelectionPanel count={selectionSize} />;
  }
  if (selectionSize === 1 && primaryGlyph) {
    return (
      <SingleEditor
        sessionId={sessionId}
        glyph={primaryGlyph}
        classNames={classNames}
      />
    );
  }
  return null;
}

function MultiSelectionPanel({ count }: { count: number }) {
  const clearSelection = useUiStore((s) => s.clearSelection);
  const selectedGlyphIds = useUiStore((s) => s.selectedGlyphIds);
  const softDeleteGlyphs = useUiStore((s) => s.softDeleteGlyphs);

  function handleDelete() {
    softDeleteGlyphs([...selectedGlyphIds]);
  }

  return (
    <aside className="w-72 shrink-0 overflow-auto border-l border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-800">
          Multi-selection
        </h2>
        <Button
          variant="ghost"
          onClick={() => clearSelection()}
          className="px-2 py-0.5"
        >
          ✕
        </Button>
      </div>
      <p className="rounded border border-slate-200 bg-slate-50 p-3 text-sm text-slate-600">
        <span className="font-semibold">{count} selected</span> — multi-edit in
        Phase C.
      </p>
      <Button
        variant="secondary"
        onClick={handleDelete}
        className="mt-3 w-full border-red-300 text-red-700 hover:bg-red-50"
      >
        Delete {count} glyphs
      </Button>
      <p className="mt-3 text-xs text-slate-400">
        Deleted glyphs move to the “Deleted” section below Staves and can be put
        back until you export. Press{" "}
        <kbd className="rounded border border-slate-300 bg-slate-50 px-1">
          Esc
        </kbd>{" "}
        to clear the selection.
      </p>
    </aside>
  );
}

interface SingleEditorProps {
  sessionId: string;
  glyph: GlyphDTO;
  classNames: string[];
}

function SingleEditor({ sessionId, glyph, classNames }: SingleEditorProps) {
  const [className, setClassName] = useState(glyph.class_name);
  const updateGlyph = useUpdateGlyph(sessionId);
  const classify = useClassify(sessionId);
  const queryClient = useQueryClient();
  const clearSelection = useUiStore((s) => s.clearSelection);
  const softDeleteGlyphs = useUiStore((s) => s.softDeleteGlyphs);

  const pending = updateGlyph.isPending || classify.isPending;
  const isNeume = glyph.category === "Neumes";

  // applyRef keeps the latest handleApply reachable from the window keydown
  // listener without re-binding the listener on every render.
  const applyRef = useRef<() => Promise<void>>(() => Promise.resolve());

  async function applyClassName(override?: string) {
    const name = (override ?? className).trim();
    if (!name) return;
    if (!isNeume) return;
    if (pending) return;
    if (override !== undefined && override !== className) {
      setClassName(name);
    }
    await updateGlyph.mutateAsync({
      glyphId: glyph.id,
      patch: { class_name: name, id_state_manual: true },
    });
    await classify.mutateAsync(1);
    queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
  }
  applyRef.current = () => applyClassName();

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    void applyClassName();
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

  function handleDelete() {
    softDeleteGlyphs([glyph.id]);
  }

  // Enter applies the current class name even when focus is on a bbox or
  // tile (i.e. anywhere outside an input/textarea). The autocomplete's own
  // Enter handling is preserved because isEditableTarget short-circuits.
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

  return (
    <aside className="w-72 shrink-0 overflow-auto border-l border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-800">Edit glyph</h2>
        <Button
          variant="ghost"
          onClick={() => clearSelection()}
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
              onApply={(v) => void applyClassName(v)}
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

      <div className="mt-4 border-t border-slate-200 pt-3">
        <Button
          variant="secondary"
          onClick={handleDelete}
          disabled={pending}
          className="w-full border-red-300 text-red-700 hover:bg-red-50"
        >
          Delete glyph
        </Button>
        <p className="mt-2 text-xs text-slate-400">
          Moves to the Deleted section; can be put back until export.
        </p>
      </div>
    </aside>
  );
}
