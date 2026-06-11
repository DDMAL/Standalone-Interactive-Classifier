import { type SplitArgs, splitGlyph } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";

interface UseSplitVars {
  glyphId: string;
  regions: SplitArgs["regions"];
}

/**
 * Slice one glyph into N children along user-drawn rectangles. On success
 * the session is invalidated and the selection is cleared — children are
 * UNCLASSIFIED with confidence 0, so they naturally surface at the top of
 * the ascending-confidence queue for the user to re-label one at a time.
 */
export function useSplit(sessionId: string) {
  const queryClient = useQueryClient();
  const clearSelection = useUiStore((s) => s.clearSelection);
  return useMutation<GlyphDTO[], Error, UseSplitVars>({
    mutationFn: ({ glyphId, regions }) =>
      splitGlyph(sessionId, glyphId, { regions }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
      clearSelection();
    },
  });
}
