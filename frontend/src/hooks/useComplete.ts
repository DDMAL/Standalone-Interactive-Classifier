import {
  type ExportSelection,
  completeSession,
  deleteGlyph,
} from "@/api/sessions";
import { downloadBlob } from "@/lib/download";
import { useUiStore } from "@/store/uiStore";
import { useMutation } from "@tanstack/react-query";

/**
 * Commits any soft-deleted glyphs to the backend, then completes the
 * session and downloads the GameraXML response. Deletes run in parallel;
 * the put-back affordance is gone once this kicks off.
 *
 * Pass an {@link ExportSelection} to choose which sections (page, manual
 * neumes, preset/uploaded training) get folded into the exported XML.
 */
export function useComplete(sessionId: string) {
  return useMutation({
    mutationFn: async (selection: ExportSelection) => {
      const ids = [...useUiStore.getState().deletedGlyphIds];
      if (ids.length > 0) {
        await Promise.all(ids.map((id) => deleteGlyph(sessionId, id)));
        useUiStore.getState().clearDeleted();
      }
      return completeSession(sessionId, selection);
    },
    onSuccess: ({ blob, filename }) => downloadBlob(blob, filename),
  });
}
