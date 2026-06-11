import { saveSession } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export function useSave(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => saveSession(sessionId),
    onSuccess: (dto) => queryClient.setQueryData(sessionKey(sessionId), dto),
  });
}
