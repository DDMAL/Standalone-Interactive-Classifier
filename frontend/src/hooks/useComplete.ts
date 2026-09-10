import { type ExportSelection, completeSession } from "@/api/sessions";
import { commitSoftDeletes } from "@/hooks/useFlushDeletions";
import { downloadBlob } from "@/lib/download";
import { useMutation, useQueryClient } from "@tanstack/react-query";

/**
 * Commits any soft-deleted glyphs to the backend, then exports the
 * GameraXML for the session and downloads it. Deletes run in parallel;
 * the put-back affordance is gone once this kicks off.
 *
 * Pass an {@link ExportSelection} to choose which sections (page, manual
 * neumes, preset/uploaded training) get folded into the exported XML.
 *
 * Exporting no longer finalises the session — the backend leaves it
 * editable — so we stay on the edit view after the download. The user can
 * keep correcting and export again as many times as they like.
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
    },
  });
}
