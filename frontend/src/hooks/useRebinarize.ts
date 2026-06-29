import { rebinarize } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import type { BinarizationMethod } from "@/types/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";

/**
 * Switch the page's binarisation method. Rebuilds every glyph mask from
 * the retained page + bboxes; manual labels are kept, manual groups/splits
 * reset. The fresh session DTO replaces the cached one so the grid + page
 * overlay re-render under the new masks.
 */
export function useRebinarize(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (method: BinarizationMethod) => rebinarize(sessionId, method),
    onSuccess: (dto) => queryClient.setQueryData(sessionKey(sessionId), dto),
  });
}
