import {
  type CreateSessionArgs,
  type CreateSessionFromStagingArgs,
  classify,
  completeSession,
  createSession,
  createSessionFromStaging,
} from "@/api/sessions";
import { downloadBlob } from "@/lib/download";
import { useMutation } from "@tanstack/react-query";

/**
 * A one-shot shortcut for the whole classify-then-export loop: create the
 * session, run a single classification round over the page, and download the
 * result as GameraXML — all without entering the interactive session view.
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
      await classify(dto.id);
      const { blob, filename } = await completeSession(dto.id, { page: true });
      downloadBlob(blob, filename);
      return filename;
    },
  });
}
