import { useSessionList } from "@/hooks/useSessionList";
import { useUiStore } from "@/store/uiStore";
import type { ClassifierState } from "@/types/api";

// Human labels for the lifecycle state shown as a badge on each row.
const STATE_LABEL: Record<ClassifierState, string> = {
  import: "Importing",
  classifying: "In progress",
  export: "Completed",
};

function formatWhen(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleString();
}

/**
 * "Resume a saved session" list for standalone IC. Enumerates stored sessions
 * (GET /sessions) and, on click, opens one via the same {@link setSession} path
 * the mothra deep-link uses — the page image is served by the API since this
 * frontend didn't upload it. Renders nothing while loading, on error, or when
 * there are no saved sessions, so first-time users just see the upload form.
 */
export function SessionResumeList() {
  const setSession = useUiStore((s) => s.setSession);
  const { data, isLoading, isError } = useSessionList();

  if (isLoading || isError) return null;
  const sessions = data ?? [];
  if (sessions.length === 0) return null;

  return (
    <div className="w-[28rem] rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold text-slate-800">
        Resume a saved session
      </h2>
      <ul className="max-h-64 space-y-1 overflow-y-auto">
        {sessions.map((s) => {
          const when = formatWhen(s.updated_at);
          return (
            <li key={s.id}>
              <button
                type="button"
                onClick={() => setSession(s.id, `/sessions/${s.id}/page`)}
                className="flex w-full items-center justify-between gap-3 rounded border border-slate-200 px-3 py-2 text-left hover:bg-slate-50"
              >
                <span className="min-w-0">
                  <span className="block truncate text-sm font-medium text-slate-700">
                    {s.source_name || s.id}
                  </span>
                  <span className="block text-xs text-slate-400">
                    {s.n_glyphs} glyph{s.n_glyphs === 1 ? "" : "s"}
                    {when && ` · ${when}`}
                  </span>
                </span>
                <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
                  {STATE_LABEL[s.state]}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
