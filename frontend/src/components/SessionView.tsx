import { ClassTreePanel } from "@/components/ClassTreePanel";
import { EditPanel } from "@/components/EditPanel";
import { GlyphGrid } from "@/components/GlyphGrid";
import { PageImagePane } from "@/components/PageImagePane";
import { Toolbar } from "@/components/Toolbar";
import { useSelectionSync } from "@/hooks/useSelectionSync";
import { useSession } from "@/hooks/useSession";
import { useZoomPan } from "@/hooks/useZoomPan";
import { byConfidenceAsc } from "@/lib/format";
import { actionForKey, isEditableTarget } from "@/lib/keymap";
import { useUiStore } from "@/store/uiStore";
import { useEffect, useMemo } from "react";

export function SessionView({ sessionId }: { sessionId: string }) {
  const { data: session, isLoading, isError, error } = useSession(sessionId);
  const selectedGlyphIds = useUiStore((s) => s.selectedGlyphIds);
  const primaryGlyphId = useUiStore((s) => s.primaryGlyphId);
  const clearSelection = useUiStore((s) => s.clearSelection);
  const zoomPan = useZoomPan();

  useSelectionSync();

  const sortedGlyphs = useMemo(
    () => (session ? [...session.glyphs].sort(byConfidenceAsc) : []),
    [session],
  );

  const primaryGlyph = useMemo(
    () => session?.glyphs.find((g) => g.id === primaryGlyphId) ?? null,
    [session, primaryGlyphId],
  );

  const selectedGlyphs = useMemo(() => {
    if (!session || selectedGlyphIds.size === 0) return [];
    return session.glyphs.filter((g) => selectedGlyphIds.has(g.id));
  }, [session, selectedGlyphIds]);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (isEditableTarget(e.target)) return;
      const action = actionForKey(e);
      if (!action) return;
      switch (action.type) {
        case "zoomIn":
          zoomPan.zoomIn();
          break;
        case "zoomOut":
          zoomPan.zoomOut();
          break;
        case "zoomReset":
          zoomPan.reset();
          break;
        case "clearSelection":
          clearSelection();
          break;
        case "pan":
          zoomPan.pan(action.dx, action.dy);
          break;
      }
      e.preventDefault();
    }
  }, [zoomPan.zoomIn, zoomPan.zoomOut, zoomPan.reset, zoomPan.pan, clearSelection]);

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-slate-500">
        Loading…
      </div>
    );
  }
  if (isError || !session) {
    return (
      <div className="flex h-full items-center justify-center text-red-600">
        {(error as Error)?.message ?? "Failed to load session"}
      </div>
    );
  }

  const selectionSize = selectedGlyphIds.size;
  const showEditPanel = selectionSize >= 1;

  return (
    <div className="flex h-full flex-col">
      <Toolbar sessionId={sessionId} glyphCount={session.glyphs.length} />
      <div className="flex min-h-0 flex-1">
        <ClassTreePanel sessionId={sessionId} session={session} />
        <PageImagePane glyphs={session.glyphs} zoomPan={zoomPan} />
        <GlyphGrid glyphs={sortedGlyphs} />
        {showEditPanel && (
          <EditPanel
            key={selectionSize === 1 ? (primaryGlyphId ?? "primary") : "multi"}
            sessionId={sessionId}
            primaryGlyph={primaryGlyph}
            selectionSize={selectionSize}
            selectedGlyphs={selectedGlyphs}
            classNames={session.class_names}
          />
        )}
      </div>
    </div>
  );
}
