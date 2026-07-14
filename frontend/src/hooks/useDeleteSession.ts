import { clearSessions, deleteSession } from "@/api/sessions";
import { sessionListKey } from "@/hooks/useSessionList";
import { useMutation, useQueryClient } from "@tanstack/react-query";

/**
 * Discard a stored session (DELETE /sessions/{id}) and refresh the resume list
 * so the removed session drops out. Backs the delete action in
 * {@link SessionResumeList}. Deletion is permanent — the session and its glyphs
 * are freed on the backend.
 */
export function useDeleteSession() {
  const queryClient = useQueryClient();
  return useMutation<void, Error, string>({
    mutationFn: (id) => deleteSession(id),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: sessionListKey }),
  });
}

/**
 * Discard *every* stored session (DELETE /sessions) and refresh the resume
 * list, which then empties. Backs the "clear all" action in
 * {@link SessionResumeList}. Permanent and irreversible — the client confirms
 * before calling this.
 */
export function useClearSessions() {
  const queryClient = useQueryClient();
  return useMutation<{ deleted: number }, Error, void>({
    mutationFn: () => clearSessions(),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: sessionListKey }),
  });
}
