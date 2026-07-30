import clsx from "clsx";
import { useState } from "react";

import { ConfirmDialog } from "@/components/ConfirmDialog";
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  TrashIcon,
} from "@/components/ui/icons";
import { useClearSessions, useDeleteSession } from "@/hooks/useDeleteSession";
import { useSessionList } from "@/hooks/useSessionList";
import { useUiStore } from "@/store/uiStore";
import type { SessionSummary } from "@/types/api";

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
 *
 * Sessions are grouped into two nested lists — "In progress" and "Completed" —
 * so a finished page can't be mistaken for a resumable one. Persisted sessions
 * are always `classifying` or `export` (creation ingests before saving), so
 * anything not completed belongs in the in-progress bucket.
 */
export function SessionResumeList() {
  const setSession = useUiStore((s) => s.setSession);
  const { data, isLoading, isError } = useSessionList();
  const deleteSessionMut = useDeleteSession();
  const clearSessionsMut = useClearSessions();
  // Collapsed by default so the create form is front-and-centre; the user
  // expands the rail when they want to resume a saved session.
  const [expanded, setExpanded] = useState(false);
  // The session awaiting delete confirmation, or null when the prompt is shut.
  const [deleteTarget, setDeleteTarget] = useState<SessionSummary | null>(null);
  // Whether the "clear all" confirmation prompt is open.
  const [clearOpen, setClearOpen] = useState(false);

  if (isLoading || isError) return null;
  const sessions = data ?? [];
  if (sessions.length === 0) return null;

  const inProgress = sessions.filter((s) => s.state !== "export");
  const completed = sessions.filter((s) => s.state === "export");

  function handleConfirmDelete() {
    if (!deleteTarget) return;
    deleteSessionMut.mutate(deleteTarget.id, {
      onSettled: () => setDeleteTarget(null),
    });
  }

  function handleConfirmClear() {
    clearSessionsMut.mutate(undefined, {
      onSettled: () => setClearOpen(false),
    });
  }

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
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setClearOpen(true)}
            title="Delete all saved sessions"
            aria-label="Delete all saved sessions"
            className="rounded px-1.5 py-0.5 text-xs font-medium text-slate-500 hover:bg-red-50 hover:text-red-600"
          >
            Clear all
          </button>
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
      </div>
      <div className="flex-1 space-y-5 overflow-y-auto p-3">
        <SessionGroup
          title="In progress"
          sessions={inProgress}
          onOpen={(id) => setSession(id, `/sessions/${id}/page`)}
          onDelete={(s) => setDeleteTarget(s)}
        />
        <SessionGroup
          title="Completed"
          sessions={completed}
          onDelete={(s) => setDeleteTarget(s)}
        />
      </div>

      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(o) => {
          if (!o) setDeleteTarget(null);
        }}
        title={
          deleteTarget
            ? `Delete session '${deleteTarget.source_name || deleteTarget.id}'?`
            : "Delete session?"
        }
        description="This permanently removes the saved session and all of its glyphs. This cannot be undone."
        confirmLabel="Delete"
        destructive
        pending={deleteSessionMut.isPending}
        onConfirm={handleConfirmDelete}
      />

      <ConfirmDialog
        open={clearOpen}
        onOpenChange={setClearOpen}
        title={`Delete all ${sessions.length} saved session${sessions.length === 1 ? "" : "s"}?`}
        description="This permanently removes every saved session and all of their glyphs. This cannot be undone."
        confirmLabel="Delete all"
        destructive
        pending={clearSessionsMut.isPending}
        onConfirm={handleConfirmClear}
      />
    </aside>
  );
}

interface SessionGroupProps {
  title: string;
  sessions: SessionSummary[];
  /** Omitted for read-only groups (e.g. completed sessions), which render
   * their open action disabled. */
  onOpen?: (id: string) => void;
  /** Request a delete-confirmation prompt for the given session. */
  onDelete: (session: SessionSummary) => void;
}

/**
 * One titled, nested list of session rows. Renders nothing when its bucket is
 * empty so a section header never sits above zero rows. The open action is
 * clickable only when {@link onOpen} is supplied — a completed session is
 * terminal/read-only, so opening it would just 409 on the first mutating
 * action. The trash button is always available so any saved session, finished
 * or not, can be discarded.
 */
function SessionGroup({
  title,
  sessions,
  onOpen,
  onDelete,
}: SessionGroupProps) {
  if (sessions.length === 0) return null;
  const disabled = onOpen === undefined;
  return (
    <section>
      <h3 className="px-1 pb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">
        {title} ({sessions.length})
      </h3>
      <ul className="space-y-1">
        {sessions.map((s) => {
          const when = formatWhen(s.updated_at);
          const label = s.source_name || s.id;
          return (
            <li key={s.id}>
              <div className="flex items-stretch overflow-hidden rounded border border-slate-200">
                <button
                  type="button"
                  disabled={disabled}
                  onClick={disabled ? undefined : () => onOpen(s.id)}
                  title={
                    disabled
                      ? "This session is completed and can't be resumed"
                      : undefined
                  }
                  aria-disabled={disabled}
                  className={clsx(
                    "flex min-w-0 flex-1 flex-col gap-0.5 px-3 py-2 text-left",
                    disabled
                      ? "cursor-not-allowed opacity-60"
                      : "hover:bg-slate-50",
                  )}
                >
                  <span className="truncate text-sm font-medium text-slate-700">
                    {label}
                  </span>
                  <span className="text-xs text-slate-400">
                    {s.n_glyphs} glyph{s.n_glyphs === 1 ? "" : "s"}
                    {when && ` · ${when}`}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => onDelete(s)}
                  title="Delete session"
                  aria-label={`Delete session ${label}`}
                  className="flex shrink-0 items-center border-l border-slate-200 px-2.5 text-slate-400 hover:bg-red-50 hover:text-red-600"
                >
                  <TrashIcon />
                </button>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
