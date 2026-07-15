import {
  type ExportSelection,
  completeSession,
  deleteGlyph,
} from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import { downloadBlob } from "@/lib/download";
import { useUiStore } from "@/store/uiStore";
import type { SessionDTO } from "@/types/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";

/**
 * Commits any soft-deleted glyphs to the backend, then completes the
 * session and downloads the GameraXML response. Deletes run in parallel;
 * the put-back affordance is gone once this kicks off.
 *
 * Pass an {@link ExportSelection} to choose which sections (page, manual
 * neumes, preset/uploaded training) get folded into the exported XML.
 *
 * Completing moves the session into the terminal EXPORT state, which is
 * read-only — no further classifying is possible. So once the download
 * fires we clear the session and drop back to the main page rather than
 * leaving the user on an edit view that can only 409.
 */
export function useComplete(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (selection: ExportSelection) => {
      const ids = [...useUiStore.getState().deletedGlyphIds];
      if (ids.length > 0) {
        await Promise.all(ids.map((id) => deleteGlyph(sessionId, id)));
        // Prune the committed deletions from the cached session *before*
        // clearing the UI-side filter. `completeSession` returns the XML
        // blob, not a fresh SessionDTO, so nothing else drops these glyphs
        // from the query cache; if we cleared `deletedGlyphIds` while they
        // still sat in the cache, the grid would resurrect them into their
        // category sections (the "deleted glyphs come back on export" bug).
        const deleted = new Set(ids);
        queryClient.setQueryData<SessionDTO>(sessionKey(sessionId), (old) =>
          old
            ? { ...old, glyphs: old.glyphs.filter((g) => !deleted.has(g.id)) }
            : old,
        );
        useUiStore.getState().clearDeleted();
      }
      return completeSession(sessionId, selection);
    },
    onSuccess: ({ blob, filename }) => {
      downloadBlob(blob, filename);
      useUiStore.getState().clearSession();
    },
  });
}
