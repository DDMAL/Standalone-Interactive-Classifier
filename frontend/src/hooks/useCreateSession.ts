import { type CreateSessionArgs, createSession } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import { useUiStore } from "@/store/uiStore";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export function useCreateSession() {
  const queryClient = useQueryClient();
  const setSession = useUiStore((s) => s.setSession);

  return useMutation({
    mutationFn: (args: CreateSessionArgs) => createSession(args),
    onSuccess: (dto, args) => {
      queryClient.setQueryData(sessionKey(dto.id), dto);
      setSession(dto.id, URL.createObjectURL(args.pageImage));
    },
  });
}
