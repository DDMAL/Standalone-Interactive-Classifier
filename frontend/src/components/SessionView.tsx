import { ClassTreePanel } from "@/components/ClassTreePanel";
import { Button } from "@/components/ui/Button";
import { GlyphGrid } from "@/components/GlyphGrid";
import { PageImagePane } from "@/components/PageImagePane";
import { RightDock } from "@/components/RightDock";
import { Toolbar } from "@/components/Toolbar";
import { PageImageProvider } from "@/hooks/usePageImage";
import { useSelectionSync } from "@/hooks/useSelectionSync";
import { useSession } from "@/hooks/useSession";
import { useUndoApply } from "@/hooks/useUpdateGlyphs";
import { useZoomPan } from "@/hooks/useZoomPan";
import { byConfidenceAsc, trainingPoolSize } from "@/lib/format";
import { actionForKey, isEditableTarget, isTypeToFocusKey } from "@/lib/keymap";
import { isModalOpen, useUiStore } from "@/store/uiStore";
import { ApiError } from "@/types/api";
import { useEffect, useMemo, useRef } from "react";

// Whether an embedding host deep-linked us straight into a session id
// (?session=<id>), as mothra's IC stage does. Read once at module scope,
// mirroring App.tsx — the query string does not change under us.
const isDeepLinked = new URLSearchParams(window.location.search).has("session");

export function SessionView({ sessionId }: { sessionId: string }) {
  const { data: session, isLoading, isError, error } = useSession(sessionId);
  const selectedGlyphIds = useUiStore((s) => s.selectedGlyphIds);
  const primaryGlyphId = useUiStore((s) => s.primaryGlyphId);
  const clearSelection = useUiStore((s) => s.clearSelection);
  const clearSession = useUiStore((s) => s.clearSession);
  const setBboxesHidden = useUiStore((s) => s.setBboxesHidden);
  const zoomPan = useZoomPan();
  const undoApply = useUndoApply(sessionId);
  const undoApplyRef = useRef(undoApply.mutate);
  undoApplyRef.current = undoApply.mutate;

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

  // Hand-labelled neumes on this page — one of the export options.
  const manualNeumeCount = useMemo(
    () =>
      session
        ? session.glyphs.filter(
            (g) => g.category === "Neumes" && g.id_state_manual,
          ).length
        : 0,
    [session],
  );

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      // A modal dialog (Split/Group) owns the keyboard while open; Radix
      // handles its own Esc-to-close, so page zoom/pan/clear must stand down.
      if (isModalOpen()) return;
      if (isEditableTarget(e.target)) return;
      // While a glyph is selected, bare alphanumeric keys are claimed by the
      // edit panel (type-to-focus), so don't let e.g. "0" reset the zoom.
      if (
        isTypeToFocusKey(e) &&
        useUiStore.getState().selectedGlyphIds.size > 0
      )
        return;
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
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    zoomPan.zoomIn,
    zoomPan.zoomOut,
    zoomPan.reset,
    zoomPan.pan,
    clearSelection,
  ]);

  // Cmd/Ctrl+Z triggers undo for the last apply operation.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (!(e.metaKey || e.ctrlKey)) return;
      if (e.key !== "z" && e.key !== "Z") return;
      if (e.shiftKey) return;
      if (isModalOpen()) return;
      if (isEditableTarget(e.target)) return;
      if (useUiStore.getState().undoStack.length === 0) return;
      e.preventDefault();
      undoApplyRef.current();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  // Hold "h" to temporarily hide all bboxes; release to bring them back.
  // Separate from actionForKey because it's a press-and-hold gesture (keydown
  // + keyup), not a one-shot action. A blur safety net avoids leaving boxes
  // hidden if the window loses focus mid-hold.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== "h" || e.repeat) return;
      if (isModalOpen()) return;
      if (isEditableTarget(e.target)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      setBboxesHidden(true);
    }
    function onKeyUp(e: KeyboardEvent) {
      if (e.key !== "h") return;
      setBboxesHidden(false);
    }
    function onBlur() {
      setBboxesHidden(false);
    }
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      setBboxesHidden(false);
    };
  }, [setBboxesHidden]);

  // Whether the server has forgotten this session entirely (see the branch
  // below). `code` is the API's own error code; the status check covers a
  // 404 that arrives without one.
  const isSessionGone =
    error instanceof ApiError &&
    (error.status === 404 || error.code === "not_found");

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-slate-500">
        Loading…
      </div>
    );
  }
  // A 404 here does not mean "bad request" — it means the server no longer
  // has this session, so every other action (classify, export, relabel) will
  // fail identically and there is nothing to retry. Causes: the API process
  // restarted while running the in-memory session store, which drops the
  // whole registry (GET /healthz reports which store is live), or a newer
  // session superseded this one for the same page. The raw backend string
  // ("Unknown session id: '...'") reads like a malformed request, so say what
  // actually happened instead.
  if (isSessionGone) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
        <p className="text-base font-semibold text-slate-800">
          This session is no longer on the server
        </p>
        <p className="max-w-md text-sm text-slate-600">
          The classifier service restarted, or this page was reopened in
          another session. Classifying done in this session was not saved.
        </p>
        {/* Deep-linked (the host opened us on a specific session id) means
            there is no upload screen to fall back to — clearing the session
            would leave a blank frame — so the way out is on the host's side.
            Standalone, dropping the session lands on the upload / resume
            screen, which is exactly where the user needs to be. */}
        {isDeepLinked ? (
          <p className="max-w-md text-sm text-slate-500">
            Close the classifier and reopen this page to start a new session.
          </p>
        ) : (
          <Button onClick={clearSession}>Start over</Button>
        )}
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

  return (
    <PageImageProvider>
      <div className="flex h-full flex-col">
        <Toolbar
          sessionId={sessionId}
          glyphCount={session.glyphs.length}
          trainingSize={trainingPoolSize(session)}
          manualNeumeCount={manualNeumeCount}
          presetTrainingCount={session.preset_training_count}
          uploadedTrainingCount={session.uploaded_training_count}
          binarizationMethod={session.binarization_method}
        />
        <div className="flex min-h-0 flex-1">
          <ClassTreePanel sessionId={sessionId} session={session} />
          <PageImagePane glyphs={session.glyphs} zoomPan={zoomPan} />
          <GlyphGrid glyphs={sortedGlyphs} />
          <RightDock
            sessionId={sessionId}
            session={session}
            primaryGlyph={primaryGlyph}
            selectionSize={selectionSize}
            selectedGlyphs={selectedGlyphs}
          />
        </div>
      </div>
    </PageImageProvider>
  );
}
