import { renameClass } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import type { SessionDTO } from "@/types/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";

interface RenameClassArgs {
  name: string;
  newName: string;
}

export function useRenameClass(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation<SessionDTO, Error, RenameClassArgs>({
    mutationFn: ({ name, newName }) => renameClass(sessionId, name, newName),
    onSuccess: (dto) => queryClient.setQueryData(sessionKey(sessionId), dto),
  });
}
