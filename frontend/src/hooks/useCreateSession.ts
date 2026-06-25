import {
  type CreateSessionArgs,
  type CreateSessionFromStagingArgs,
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

export function useCreateSession() {
  const queryClient = useQueryClient();
  const setSession = useUiStore((s) => s.setSession);

  return useMutation({
    mutationFn: (args: CreateSessionArgs) => createSession(args),
    onSuccess: (dto, args) => {
      queryClient.setQueryData(sessionKey(dto.id), dto);
      setSession(dto.id, URL.createObjectURL(args.pageImage));
      notifyParentSessionCreated(dto.id);
    },
  });
}

export function useCreateSessionFromStaging() {
  const queryClient = useQueryClient();
  const setSession = useUiStore((s) => s.setSession);

  return useMutation({
    mutationFn: (args: CreateSessionFromStagingArgs) =>
      createSessionFromStaging(args),
    onSuccess: (dto) => {
      queryClient.setQueryData(sessionKey(dto.id), dto);
      // The page wasn't uploaded here, so point the preview at the server.
      setSession(dto.id, `/sessions/${dto.id}/page`);
      notifyParentSessionCreated(dto.id);
    },
  });
}
