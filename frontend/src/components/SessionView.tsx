import { EditPanel } from "@/components/EditPanel";
import { GlyphGrid } from "@/components/GlyphGrid";
import { PageImagePane } from "@/components/PageImagePane";
import { Toolbar } from "@/components/Toolbar";
import { useSession } from "@/hooks/useSession";
import { byConfidenceAsc } from "@/lib/format";
import { useUiStore } from "@/store/uiStore";
import { useMemo } from "react";

export function SessionView({ sessionId }: { sessionId: string }) {
  const { data: session, isLoading, isError, error } = useSession(sessionId);
  const selectedGlyphId = useUiStore((s) => s.selectedGlyphId);

  const sortedGlyphs = useMemo(
    () => (session ? [...session.glyphs].sort(byConfidenceAsc) : []),
    [session],
  );

  const selectedGlyph = useMemo(
    () => session?.glyphs.find((g) => g.id === selectedGlyphId) ?? null,
    [session, selectedGlyphId],
  );

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center text-slate-500">
        Loading…
      </div>
    );
  }
  if (isError || !session) {
    return (
      <div className="flex h-full items-center justify-center text-red-600">
        {(error as Error)?.message ?? "Failed to load session"}
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <Toolbar sessionId={sessionId} glyphCount={session.glyphs.length} />
      <div className="flex min-h-0 flex-1">
        <PageImagePane />
        <GlyphGrid glyphs={sortedGlyphs} />
        {selectedGlyph && (
          <EditPanel
            key={selectedGlyph.id}
            sessionId={sessionId}
            glyph={selectedGlyph}
            classNames={session.class_names}
          />
        )}
      </div>
    </div>
  );
}
