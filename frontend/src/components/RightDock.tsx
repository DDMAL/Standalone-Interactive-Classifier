import { EditPanel } from "@/components/EditPanel";
import { TrainingDataPanel } from "@/components/TrainingDataPanel";
import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO, SessionDTO } from "@/types/api";

interface RightDockProps {
  sessionId: string;
  session: SessionDTO;
  primaryGlyph: GlyphDTO | null;
  selectionSize: number;
  selectedGlyphs: GlyphDTO[];
}

/**
 * The right-hand slot, shared by the EditPanel and the training-data browser.
 * The two are mutually exclusive by design (see {@link useUiStore}):
 *
 * * A selection shows the EditPanel; expanding the training panel clears the
 *   selection, so the EditPanel yields the slot.
 * * Selecting any glyph collapses the training panel, so the EditPanel returns.
 *
 * The collapsed handle stays docked at the far edge in every non-expanded
 * state — including while the EditPanel is open — so the training panel can be
 * expanded at any time (which then deselects).
 */
export function RightDock({
  sessionId,
  session,
  primaryGlyph,
  selectionSize,
  selectedGlyphs,
}: RightDockProps) {
  const expanded = useUiStore((s) => s.trainingPanelExpanded);
  const expandTrainingPanel = useUiStore((s) => s.expandTrainingPanel);
  const collapseTrainingPanel = useUiStore((s) => s.collapseTrainingPanel);

  if (expanded) {
    return (
      <TrainingDataPanel
        sessionId={sessionId}
        glyphs={session.training_glyphs}
        onCollapse={collapseTrainingPanel}
      />
    );
  }

  return (
    <>
      {selectionSize >= 1 && (
        <EditPanel
          key={selectionSize === 1 ? (primaryGlyph?.id ?? "primary") : "multi"}
          sessionId={sessionId}
          primaryGlyph={primaryGlyph}
          selectionSize={selectionSize}
          selectedGlyphs={selectedGlyphs}
          classNames={session.class_names}
        />
      )}
      <TrainingHandle
        count={session.training_glyphs.length}
        onExpand={expandTrainingPanel}
      />
    </>
  );
}

function TrainingHandle({
  count,
  onExpand,
}: {
  count: number;
  onExpand: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onExpand}
      title="Show training data"
      className="flex w-9 shrink-0 flex-col items-center gap-2 border-l border-slate-200 bg-white py-3 text-slate-500 transition-colors hover:bg-slate-50 hover:text-blue-600"
    >
      <span aria-hidden className="text-xs">
        ◀
      </span>
      <span className="text-xs font-medium [writing-mode:vertical-rl]">
        Training data ({count.toLocaleString()})
      </span>
    </button>
  );
}
