import { listSessions } from "@/api/sessions";
import type { SessionSummary } from "@/types/api";
import { useQuery } from "@tanstack/react-query";

export const sessionListKey = ["sessions"] as const;

/**
 * The stored sessions available to resume (GET /sessions). Refetched whenever
 * the upload screen mounts so a session completed/deleted elsewhere doesn't
 * linger in the list. Against the in-memory store this is empty after a
 * restart; against the persistent store it spans restarts.
 */
export function useSessionList() {
  return useQuery<SessionSummary[]>({
    queryKey: sessionListKey,
    queryFn: listSessions,
    staleTime: 0,
  });
}
