import { rebinarize } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import type { BinarizationMethod } from "@/types/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";

/**
 * Switch the page's binarisation method. Every glyph mask is re-derived from
 * the re-binarised page at that glyph's own bbox, so nothing the user built
 * is lost: manual labels, manual splits and manual groups all survive with
 * their ids intact. The fresh session DTO replaces the cached one so the grid
 * + page overlay re-render under the new masks.
 */
export function useRebinarize(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (method: BinarizationMethod) => rebinarize(sessionId, method),
    onSuccess: (dto) => queryClient.setQueryData(sessionKey(sessionId), dto),
  });
}
