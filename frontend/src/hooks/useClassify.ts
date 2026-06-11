import { classify } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export function useClassify(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (k: number) => classify(sessionId, k),
    onSuccess: (dto) => queryClient.setQueryData(sessionKey(sessionId), dto),
  });
}
