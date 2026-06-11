import { type ManualGroupArgs, manualGroup } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";

/**
 * Merge a multi-selection into a single new manual glyph. On success the
 * session is invalidated and the new glyph becomes the only selection.
 */
export function useGroup(sessionId: string) {
  const queryClient = useQueryClient();
  const selectGlyph = useUiStore((s) => s.selectGlyph);
  return useMutation<GlyphDTO, Error, ManualGroupArgs>({
    mutationFn: (body) => manualGroup(sessionId, body),
    onSuccess: (newGlyph) => {
      queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
      selectGlyph(newGlyph.id);
    },
  });
}
