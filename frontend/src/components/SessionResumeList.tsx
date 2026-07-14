import { useState } from "react";

import { ChevronLeftIcon, ChevronRightIcon } from "@/components/ui/icons";
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
 * Expandable "resume a saved session" sidebar for standalone IC, docked to the
 * left of the create-session window. Enumerates stored sessions (GET /sessions)
 * and, on click, opens one via the same {@link setSession} path the mothra
 * deep-link uses — the page image is served by the API since this frontend
 * didn't upload it. Collapses to a slim rail to give the form room. Renders
 * nothing while loading, on error, or when there are no saved sessions, so
 * first-time users just see the upload form.
 */
export function SessionResumeList() {
  const setSession = useUiStore((s) => s.setSession);
  const { data, isLoading, isError } = useSessionList();
  // Collapsed by default so the create form is front-and-centre; the user
  // expands the rail when they want to resume a saved session.
  const [expanded, setExpanded] = useState(false);

  if (isLoading || isError) return null;
  const sessions = data ?? [];
  if (sessions.length === 0) return null;

  if (!expanded) {
    return (
      <aside className="flex h-full w-11 shrink-0 flex-col items-center gap-3 border-r border-slate-200 bg-white py-3">
        <button
          type="button"
          onClick={() => setExpanded(true)}
          title="Show saved sessions"
          aria-label="Show saved sessions"
          aria-expanded={false}
          className="rounded p-1.5 text-slate-500 hover:bg-slate-100 hover:text-slate-700"
        >
          <ChevronRightIcon />
        </button>
        <span
          className="select-none text-xs font-medium text-slate-500"
          style={{ writingMode: "vertical-rl" }}
        >
          Saved sessions ({sessions.length})
        </span>
      </aside>
    );
  }

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="flex items-center justify-between gap-2 border-b border-slate-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-800">
          Resume a saved session
        </h2>
        <button
          type="button"
          onClick={() => setExpanded(false)}
          title="Collapse"
          aria-label="Collapse saved sessions"
          aria-expanded={true}
          className="rounded p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-700"
        >
          <ChevronLeftIcon />
        </button>
      </div>
      <ul className="flex-1 space-y-1 overflow-y-auto p-3">
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
    </aside>
  );
}
