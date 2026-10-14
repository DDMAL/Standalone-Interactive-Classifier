import { type ExportSelection, completeSession } from "@/api/sessions";
import { commitSoftDeletes } from "@/hooks/useFlushDeletions";
import { downloadBlob } from "@/lib/download";
import { useUiStore } from "@/store/uiStore";
import { useMutation, useQueryClient } from "@tanstack/react-query";

/**
 * Commits any soft-deleted glyphs to the backend, then exports the
 * GameraXML for the session and downloads it. Deletes run in parallel;
 * the put-back affordance is gone once this kicks off.
 *
 * Pass an {@link ExportSelection} to choose which sections (page, manual
 * neumes, preset/uploaded training) get folded into the exported XML.
 *
 * What happens after the download depends on where we run:
 * - Embedded in a host (mothra), the export does not finalise the session
 *   (`completeSession` sends `finalize=false`), so we stay on the edit view
 *   and the user can keep correcting and export again.
 * - Standalone, exporting moves the session into the terminal, read-only
 *   EXPORT state, so we clear the session and drop back to the main page
 *   rather than leaving the user on an edit view that can only 409.
 *
 * The commit itself lives in {@link commitSoftDeletes}, shared with the
 * host-driven flush (see useFlushDeletions): an embedding host exports
 * server-side and never reaches this button, so it needs the same commit on
 * its own trigger.
 */
export function useComplete(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (selection: ExportSelection) => {
      await commitSoftDeletes(queryClient, sessionId);
      return completeSession(sessionId, selection);
    },
    onSuccess: ({ blob, filename }) => {
      downloadBlob(blob, filename);
      if (window.parent === window) useUiStore.getState().clearSession();
    },
  });
}
