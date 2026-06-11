import { type UpdateGlyphArgs, updateGlyph } from "@/api/sessions";
import { useMutation } from "@tanstack/react-query";

export function useUpdateGlyph(sessionId: string) {
  return useMutation({
    mutationFn: ({
      glyphId,
      patch,
    }: { glyphId: string; patch: UpdateGlyphArgs }) =>
      updateGlyph(sessionId, glyphId, patch),
  });
}
