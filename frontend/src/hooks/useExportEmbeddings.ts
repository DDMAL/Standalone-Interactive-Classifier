import { type ExportSelection, exportSessionEmbeddings } from "@/api/sessions";
import { downloadBlob } from "@/lib/download";
import { useMutation } from "@tanstack/react-query";

/**
 * Downloads a companion SSL embeddings (.npz) file for the ssl_fusion
 * backend, using the same section selection as {@link useComplete}. Unlike
 * completing the session, this is read-only -- it doesn't move the session
 * to the terminal EXPORT state, so it can be triggered independently of
 * (before or after) the GameraXML export.
 */
export function useExportEmbeddings(sessionId: string) {
  return useMutation({
    mutationFn: (selection: ExportSelection) =>
      exportSessionEmbeddings(sessionId, selection),
    onSuccess: ({ blob, filename }) => downloadBlob(blob, filename),
  });
}
