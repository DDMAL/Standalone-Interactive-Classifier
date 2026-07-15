import { deleteTrainingGlyph } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import type { SessionDTO } from "@/types/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";

/**
 * Delete one glyph from the session's training pool. The endpoint returns the
 * updated session, so we drop it straight into the cache — the training panel
 * and the toolbar's training count re-render without a refetch. Does not
 * reclassify; the user re-runs classify from the toolbar if they want the
 * working set re-scored against the smaller pool.
 */
export function useDeleteTrainingGlyph(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation<SessionDTO, Error, string>({
    mutationFn: (glyphId) => deleteTrainingGlyph(sessionId, glyphId),
    onSuccess: (dto) => queryClient.setQueryData(sessionKey(sessionId), dto),
  });
}
