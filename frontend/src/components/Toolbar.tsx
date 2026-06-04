import { Button } from "@/components/ui/Button";
import { useComplete } from "@/hooks/useComplete";
import { useSave } from "@/hooks/useSave";
import { useUiStore } from "@/store/uiStore";

interface ToolbarProps {
  sessionId: string;
  glyphCount: number;
}

export function Toolbar({ sessionId, glyphCount }: ToolbarProps) {
  const save = useSave(sessionId);
  const complete = useComplete(sessionId);
  const clearSession = useUiStore((s) => s.clearSession);

  return (
    <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-2">
      <div className="flex items-baseline gap-3">
        <span className="font-semibold text-slate-800">
          Interactive Classifier
        </span>
        <span className="text-sm text-slate-500">{glyphCount} glyphs</span>
      </div>
      <div className="flex items-center gap-2">
        <Button variant="ghost" onClick={clearSession}>
          New session
        </Button>
        <Button
          variant="secondary"
          onClick={() => save.mutate()}
          disabled={save.isPending}
        >
          {save.isPending ? "Saving…" : "Save"}
        </Button>
        <Button onClick={() => complete.mutate()} disabled={complete.isPending}>
          {complete.isPending ? "Exporting…" : "Complete & Export"}
        </Button>
      </div>
    </header>
  );
}
