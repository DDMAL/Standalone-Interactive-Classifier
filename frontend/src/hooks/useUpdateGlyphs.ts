import { type UpdateGlyphArgs, classify, updateGlyph } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import { useMutation, useQueryClient } from "@tanstack/react-query";

interface BulkUpdateArgs {
  glyphIds: string[];
  patch: UpdateGlyphArgs;
  /** Pass false when the caller is moving glyphs across categories; the
   *  backend resets the label there, so reclassify would clobber. */
  reclassify?: boolean;
}

interface BulkUpdateResult {
  applied: number;
  failed: { glyphId: string; error: unknown }[];
}

/**
 * Fan-out updateGlyph + single classify + invalidate. Used by MultiEditPanel
 * to apply one class name to N glyphs. Returns per-id error info so the
 * caller can show an "K of N applied" status.
 */
export function useUpdateGlyphs(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation<BulkUpdateResult, Error, BulkUpdateArgs>({
    mutationFn: async ({ glyphIds, patch, reclassify = true }) => {
      const results = await Promise.allSettled(
        glyphIds.map((id) => updateGlyph(sessionId, id, patch)),
      );
      const failed: { glyphId: string; error: unknown }[] = [];
      let applied = 0;
      results.forEach((r, i) => {
        if (r.status === "fulfilled") applied += 1;
        else failed.push({ glyphId: glyphIds[i], error: r.reason });
      });
      if (reclassify && applied > 0) {
        await classify(sessionId, 1);
      }
      queryClient.invalidateQueries({ queryKey: sessionKey(sessionId) });
      return { applied, failed };
    },
  });
}
