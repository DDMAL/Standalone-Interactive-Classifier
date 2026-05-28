import { useSession } from "@/hooks/useSession";
import { useUiStore } from "@/store/uiStore";
import { useCallback } from "react";

/**
 * Returns a callback that selects every working-set glyph whose class_name
 * matches `path`. When `includeSubtree` is true, matches also include any
 * class that starts with `path + "."` — i.e. the whole dotted-namespace.
 */
export function useClassSelection() {
  const sessionId = useUiStore((s) => s.sessionId);
  const setSelection = useUiStore((s) => s.setSelection);
  const deletedGlyphIds = useUiStore((s) => s.deletedGlyphIds);
  const { data: session } = useSession(sessionId);

  return useCallback(
    (path: string, includeSubtree: boolean) => {
      if (!session) return;
      const prefix = `${path}.`;
      const ids = session.glyphs
        .filter((g) => {
          if (deletedGlyphIds.has(g.id)) return false;
          return (
            g.class_name === path ||
            (includeSubtree && g.class_name.startsWith(prefix))
          );
        })
        .map((g) => g.id);
      setSelection(ids);
    },
    [session, deletedGlyphIds, setSelection],
  );
}
