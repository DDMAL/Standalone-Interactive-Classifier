import {
  type CreateSessionArgs,
  type CreateSessionFromStagingArgs,
  classify,
  createSession,
  createSessionFromStaging,
} from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import { useUiStore } from "@/store/uiStore";
import { useMutation, useQueryClient } from "@tanstack/react-query";

// When IC runs embedded in a host app (mothra), tell the host which session
// the user just created so it can drive completion. No-op when not embedded.
function notifyParentSessionCreated(sessionId: string) {
  if (window.parent !== window) {
    window.parent.postMessage({ type: "ic:session-created", sessionId }, "*");
  }
}

// Session creation runs its own classify round server-side using the
// default ("knn") backend whenever training data was provided. If the
// user picked "ssl_fusion" on the upload screen, immediately re-run
// classification with that backend so the first thing they see matches
// their choice, instead of only applying on the next manual reclassify.
function reclassifyIfSslFusionChosen(
  sessionId: string,
  hasTrainingData: boolean,
  queryClient: ReturnType<typeof useQueryClient>,
) {
  const backend = useUiStore.getState().classifierBackend;
  if (backend === "ssl_fusion" && hasTrainingData) {
    classify(sessionId, useUiStore.getState().knnK, backend).then(() =>
      queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) }),
    );
  }
}

export function useCreateSession() {
  const queryClient = useQueryClient();
  const setSession = useUiStore((s) => s.setSession);

  return useMutation({
    mutationFn: (args: CreateSessionArgs) => createSession(args),
    onSuccess: (dto, args) => {
      queryClient.setQueryData(sessionKey(dto.id), dto);
      setSession(dto.id, URL.createObjectURL(args.pageImage));
      notifyParentSessionCreated(dto.id);
      const hasTrainingData =
        (args.trainingPresets?.length ?? 0) > 0 ||
        (args.trainingFiles?.length ?? 0) > 0;
      reclassifyIfSslFusionChosen(dto.id, hasTrainingData, queryClient);
    },
  });
}

export function useCreateSessionFromStaging() {
  const queryClient = useQueryClient();
  const setSession = useUiStore((s) => s.setSession);

  return useMutation({
    mutationFn: (args: CreateSessionFromStagingArgs) =>
      createSessionFromStaging(args),
    onSuccess: (dto, args) => {
      queryClient.setQueryData(sessionKey(dto.id), dto);
      // The page wasn't uploaded here, so point the preview at the server.
      setSession(dto.id, `/sessions/${dto.id}/page`);
      notifyParentSessionCreated(dto.id);
      const hasTrainingData =
        (args.trainingPresets?.length ?? 0) > 0 ||
        (args.trainingFiles?.length ?? 0) > 0;
      reclassifyIfSslFusionChosen(dto.id, hasTrainingData, queryClient);
    },
  });
}
