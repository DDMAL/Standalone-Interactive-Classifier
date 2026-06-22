import { completeSession, deleteGlyph } from "@/api/sessions";
import { downloadBlob } from "@/lib/download";
import { useUiStore } from "@/store/uiStore";
import { useMutation } from "@tanstack/react-query";

/**
 * Commits any soft-deleted glyphs to the backend, then completes the
 * session and downloads the GameraXML response. Deletes run in parallel;
 * the put-back affordance is gone once this kicks off.
 *
 * Pass `includeTraining` to fold the whole training set into the export
 * alongside this page; omit it (or pass `false`) for a page-only export.
 */
export function useComplete(sessionId: string) {
  return useMutation({
    mutationFn: async (includeTraining: boolean) => {
      const ids = [...useUiStore.getState().deletedGlyphIds];
      if (ids.length > 0) {
        await Promise.all(ids.map((id) => deleteGlyph(sessionId, id)));
        useUiStore.getState().clearDeleted();
      }
      return completeSession(sessionId, includeTraining);
    },
    onSuccess: ({ blob, filename }) => downloadBlob(blob, filename),
  });
}
