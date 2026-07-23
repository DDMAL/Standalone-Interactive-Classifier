import { listSessions } from "@/api/sessions";
import type { SessionSummary } from "@/types/api";
import { useQuery } from "@tanstack/react-query";

export const sessionListKey = ["sessions"] as const;

/**
 * The stored sessions available to resume (GET /sessions). Refetched whenever
 * the upload screen mounts so a session completed/deleted elsewhere doesn't
 * linger in the list. Against the in-memory store this is empty after a
 * restart; against the persistent store it spans restarts.
 *
 * Pass ``projectId`` to scope the list to one mothra project (the per-project
 * "saved sessions" view). The key stays prefixed by {@link sessionListKey} so
 * the delete/clear mutations' `invalidateQueries({ queryKey: sessionListKey })`
 * still matches — React Query invalidates by prefix.
 */
export function useSessionList(projectId?: number) {
  return useQuery<SessionSummary[]>({
    queryKey: [...sessionListKey, projectId ?? null],
    queryFn: () => listSessions(projectId),
    staleTime: 0,
  });
}
