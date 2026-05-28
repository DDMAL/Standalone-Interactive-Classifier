import { deleteClass } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import type { SessionDTO } from "@/types/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export function useDeleteClass(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation<SessionDTO, Error, string>({
    mutationFn: (name) => deleteClass(sessionId, name),
    onSuccess: (dto) => queryClient.setQueryData(sessionKey(sessionId), dto),
  });
}
