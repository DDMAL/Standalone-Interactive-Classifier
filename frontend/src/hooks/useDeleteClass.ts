import { deleteClass, updateGlyph } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import type { SessionDTO } from "@/types/api";
import { useMutation, useQueryClient } from "@tanstack/react-query";

export interface DeleteClassArgs {
  /** The class path to drop (also drops dotted-namespace subclasses). */
  name: string;
  /**
   * IDs of present glyphs still carrying this class (or its dotted
   * subclasses). They're reset to UNCLASSIFIED *before* the class is dropped,
   * otherwise `class_names` would re-derive the name from the working set and
   * the delete would appear to do nothing. Empty for a plain delete.
   */
  unclassifyGlyphIds?: string[];
}

export function useDeleteClass(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation<SessionDTO, Error, DeleteClassArgs>({
    mutationFn: async ({ name, unclassifyGlyphIds = [] }) => {
      await Promise.all(
        unclassifyGlyphIds.map((id) =>
          updateGlyph(sessionId, id, {
            class_name: "UNCLASSIFIED",
            id_state_manual: false,
          }),
        ),
      );
      return deleteClass(sessionId, name);
    },
    onSuccess: (dto) => queryClient.setQueryData(sessionKey(sessionId), dto),
  });
}
