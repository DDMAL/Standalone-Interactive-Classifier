import { type UpdateGlyphArgs, classify, updateGlyph } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import { useUiStore } from "@/store/uiStore";
import { useMutation, useQueryClient } from "@tanstack/react-query";

/**
 * Pops the top entry from the undo stack and restores each glyph's
 * class_name / id_state_manual to its pre-apply snapshot. Does NOT
 * trigger classify — the user can run Reclassify manually afterwards.
 */
export function useUndoApply(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      const { undoStack, popUndo } = useUiStore.getState();
      const entry = undoStack[undoStack.length - 1];
      if (!entry || entry.snapshots.length === 0) return;

      const results = await Promise.allSettled(
        entry.snapshots.map(({ id, class_name, id_state_manual }) =>
          updateGlyph(sessionId, id, { class_name, id_state_manual }),
        ),
      );

      queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });

      const failed = results.filter((r) => r.status === "rejected");
      if (failed.length > 0) {
        throw new Error(
          `Undo failed for ${failed.length} glyph${failed.length === 1 ? "" : "s"}`,
        );
      }

      popUndo();
    },
  });
}

interface BulkUpdateArgs {
  glyphIds: string[];
  patch: UpdateGlyphArgs;
  /** Pass false when the caller is moving glyphs across categories; the
   *  backend resets the label there, so reclassify would clobber. */
  reclassify?: boolean;
}

interface BulkUpdateResult {
  applied: number;
  failed: { glyphId: string; error: unknown }[];
}

interface PerGlyphUpdateArgs {
  assignments: { id: string; class_name: string }[];
}

/**
 * Fan-out updateGlyph with a different class_name per glyph, then classify.
 * Used by BatchConfirmDialog to confirm each neume into its own class.
 */
export function useUpdateGlyphsPerGlyph(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation<BulkUpdateResult, Error, PerGlyphUpdateArgs>({
    mutationFn: async ({ assignments }) => {
      const results = await Promise.allSettled(
        assignments.map(({ id, class_name }) =>
          updateGlyph(sessionId, id, { class_name, id_state_manual: true }),
        ),
      );
      const failed: { glyphId: string; error: unknown }[] = [];
      let applied = 0;
      results.forEach((r, i) => {
        if (r.status === "fulfilled") applied += 1;
        else failed.push({ glyphId: assignments[i].id, error: r.reason });
      });
      let classifyError: unknown;
      if (applied > 0) {
        try {
          await classify(
            sessionId,
            useUiStore.getState().knnK,
            useUiStore.getState().classifierBackend,
          );
        } catch (error) {
          // The label updates above already persisted server-side --
          // invalidate below regardless of whether reclassify succeeded,
          // so the UI doesn't go stale on a classify failure. Surface the
          // error afterwards rather than swallowing it.
          classifyError = error;
        }
      }
      queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
      if (classifyError) throw classifyError;
      return { applied, failed };
    },
  });
}

/**
 * Fan-out updateGlyph + single classify + invalidate. Used by MultiEditPanel
 * to apply one class name to N glyphs. Returns per-id error info so the
 * caller can show an "K of N applied" status.
 */
export function useUpdateGlyphs(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation<BulkUpdateResult, Error, BulkUpdateArgs>({
    mutationFn: async ({ glyphIds, patch, reclassify = true }) => {
      const results = await Promise.allSettled(
        glyphIds.map((id) => updateGlyph(sessionId, id, patch)),
      );
      const failed: { glyphId: string; error: unknown }[] = [];
      let applied = 0;
      results.forEach((r, i) => {
        if (r.status === "fulfilled") applied += 1;
        else failed.push({ glyphId: glyphIds[i], error: r.reason });
      });
      let classifyError: unknown;
      if (reclassify && applied > 0) {
        try {
          await classify(
            sessionId,
            useUiStore.getState().knnK,
            useUiStore.getState().classifierBackend,
          );
        } catch (error) {
          // See useUpdateGlyphsPerGlyph above -- persisted labels shouldn't
          // go stale in the UI just because reclassify failed afterwards.
          classifyError = error;
        }
      }
      queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
      if (classifyError) throw classifyError;
      return { applied, failed };
    },
  });
}
