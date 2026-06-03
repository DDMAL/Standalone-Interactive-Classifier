import { Button } from "@/components/ui/Button";
import { useClassify } from "@/hooks/useClassify";
import { useComplete } from "@/hooks/useComplete";
import { useSave } from "@/hooks/useSave";
import { useUiStore } from "@/store/uiStore";
import { clsx } from "clsx";

interface ToolbarProps {
  sessionId: string;
  glyphCount: number;
}

const K_CHOICES = [1, 3, 5, 7] as const;

export function Toolbar({ sessionId, glyphCount }: ToolbarProps) {
  const save = useSave(sessionId);
  const complete = useComplete(sessionId);
  const classify = useClassify(sessionId);
  const clearSession = useUiStore((s) => s.clearSession);
  const knnK = useUiStore((s) => s.knnK);
  const setKnnK = useUiStore((s) => s.setKnnK);

  // Changing k re-runs the classification stage with the new neighbour
  // count. No-op when the same k is clicked, or while a classify is in
  // flight, to avoid redundant/concurrent rounds.
  const handleKChange = (k: number) => {
    if (k === knnK || classify.isPending) return;
    setKnnK(k);
    classify.mutate(k);
  };

  return (
    <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-2">
      <div className="flex items-baseline gap-3">
        <span className="font-semibold text-slate-800">
          Interactive Classifier
        </span>
        <span className="text-sm text-slate-500">{glyphCount} glyphs</span>
      </div>
      <div className="flex items-center gap-2">
        <div
          className="flex items-center gap-1 rounded border border-slate-200 bg-slate-50 px-2 py-1"
          title="Neighbour count for kNN classification"
        >
          <span className="text-xs font-medium text-slate-600">k</span>
          <div className="flex overflow-hidden rounded border border-slate-300">
            {K_CHOICES.map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => handleKChange(k)}
                disabled={classify.isPending}
                className={clsx(
                  "px-2 py-0.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60",
                  k === knnK
                    ? "bg-blue-600 text-white"
                    : "bg-white text-slate-700 hover:bg-slate-100",
                )}
              >
                {k}
              </button>
            ))}
          </div>
        </div>
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
