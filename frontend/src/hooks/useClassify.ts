import { classify } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import type { ClassifierBackend } from "@/store/uiStore";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export function useClassify(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      k,
      backend = "knn",
    }: { k: number; backend?: ClassifierBackend }) =>
      classify(sessionId, k, backend),
    onSuccess: (dto) => queryClient.setQueryData(sessionKey(sessionId), dto),
  });
}
