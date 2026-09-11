import {
  type CreateSessionArgs,
  type CreateSessionFromStagingArgs,
  classify,
  completeSession,
  createSession,
  createSessionFromStaging,
} from "@/api/sessions";
import { downloadBlob } from "@/lib/download";
import { useUiStore } from "@/store/uiStore";
import { useMutation } from "@tanstack/react-query";

/**
 * A one-shot shortcut for the whole classify-then-export loop: create the
 * session and run a single classification round over the page — all without
 * entering the interactive session view.
 *
 * Standalone, it then downloads the result as GameraXML. Embedded in a host
 * app (mothra), it instead hands the session off to the host — which drives
 * completion via its own server-to-server /complete bridge and queues the
 * page — so "trust the classifier" pages land in the host's encode queue
 * exactly like manually-classified ones, with no stray browser download.
 *
 * Only meaningful when a training pool exists (presets and/or uploads); with
 * an empty pool the classify round has nothing to learn from and fails, so the
 * caller greys out the trigger in that case.
 */
export type AutoExportInput =
  | { kind: "upload"; args: CreateSessionArgs }
  | { kind: "staging"; args: CreateSessionFromStagingArgs };

export function useAutoExport() {
  return useMutation({
    mutationFn: async (input: AutoExportInput) => {
      const dto =
        input.kind === "staging"
          ? await createSessionFromStaging(input.args)
          : await createSession(input.args);
      // Session creation already runs a classify round when training data is
      // present, but do one explicit round here so auto-export owns the whole
      // classify-then-export sequence regardless of that ingest-time detail.
      const { classifierBackend, knnK } = useUiStore.getState();
      await classify(dto.id, knnK, classifierBackend);

      // Embedded: don't finalise/download here. Announce the session so the
      // host's session state (and its "queue page" button) stays in sync, then
      // signal auto-export so the host runs its own queue path — the same one
      // the manual button uses, keeping GameraXML server-side per design.
      if (window.parent !== window) {
        window.parent.postMessage(
          { type: "ic:session-created", sessionId: dto.id },
          "*",
        );
        window.parent.postMessage(
          { type: "ic:auto-export", sessionId: dto.id },
          "*",
        );
        return dto.id;
      }

      const { blob, filename } = await completeSession(dto.id, { page: true });
      downloadBlob(blob, filename);
      return filename;
    },
  });
}
