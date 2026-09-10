import { deleteGlyph } from "@/api/sessions";
import { sessionKey } from "@/hooks/useSession";
import { useUiStore } from "@/store/uiStore";
import type { SessionDTO } from "@/types/api";
import { type QueryClient, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

/**
 * Commit the UI's soft-deleted glyphs to the backend for `sessionId`, then
 * clear the put-back affordance. Returns how many were committed.
 *
 * Deleting a glyph only ever marks it in {@link useUiStore} (hidden from the
 * grid/overlay, recoverable from the Deleted section) — nothing reaches the
 * API until this runs. Shared by the two things that end that grace period:
 * {@link useComplete}'s export, and the host-driven flush below.
 *
 * The cached session is pruned *before* `clearDeleted`, or the grid would
 * resurrect the glyphs into their category sections for the moment between
 * the two (the "deleted glyphs come back on export" bug).
 */
export async function commitSoftDeletes(
  queryClient: QueryClient,
  sessionId: string,
): Promise<number> {
  const ids = [...useUiStore.getState().deletedGlyphIds];
  if (ids.length === 0) return 0;
  await Promise.all(ids.map((id) => deleteGlyph(sessionId, id)));
  const deleted = new Set(ids);
  queryClient.setQueryData<SessionDTO>(sessionKey(sessionId), (old) =>
    old
      ? { ...old, glyphs: old.glyphs.filter((g) => !deleted.has(g.id)) }
      : old,
  );
  useUiStore.getState().clearDeleted();
  return ids.length;
}

/**
 * Let an embedding host (mothra) commit this session's soft deletes on
 * demand, by posting `{type: "ic:flush-deletions", sessionId, requestId}`.
 * Replies with `ic:deletions-flushed` (carrying `count`) or
 * `ic:deletions-flush-failed` (carrying `error`), echoing `requestId` so a
 * host with several requests in flight can tell them apart.
 *
 * Why the host needs this: soft deletes live only in this frame's memory
 * until an export commits them, and mothra never presses IC's own Export
 * button — it exports server-to-server (`POST /sessions/{id}/complete`)
 * from its encode step, which reads the *backend's* working set. Without a
 * flush, every glyph the user deleted comes back in that GameraXML and ends
 * up as a neume in the MEI. Worse, mothra reloads this iframe when its
 * filmstrip changes page, so the ids are simply gone by then — the host has
 * to ask while the frame is still alive.
 *
 * Nothing about the standalone app changes: with no parent there is no one
 * to send the message, and the listener isn't even installed. Embedded, the
 * user-visible behaviour is unchanged too — this performs exactly the
 * deletes the Export button would have, just triggered by the host.
 */
export function useDeletionFlushBridge() {
  const queryClient = useQueryClient();
  useEffect(() => {
    if (window.parent === window) return;

    function onMessage(e: MessageEvent) {
      // Only the embedding host may drive this; a message from any other
      // frame is ignored outright.
      if (e.source !== window.parent) return;
      const data = e.data;
      if (data?.type !== "ic:flush-deletions") return;
      const requestId =
        typeof data.requestId === "string" ? data.requestId : null;
      const reply = (payload: Record<string, unknown>) =>
        window.parent.postMessage({ ...payload, requestId }, "*");

      const current = useUiStore.getState().sessionId;
      const requested =
        typeof data.sessionId === "string" ? data.sessionId : null;
      // A mismatch means the host thinks we're on a different page than we
      // are; committing this frame's deletions against the session it named
      // would delete the wrong glyphs. Report instead, and let the host
      // decide (it surfaces this rather than silently queueing the page).
      if (requested && current && requested !== current) {
        reply({
          type: "ic:deletions-flush-failed",
          sessionId: requested,
          error: `classifier is on session ${current}, not ${requested}`,
        });
        return;
      }
      // No session open (the create-session screen, or the auto-classify
      // path that never enters one) — nothing can have been soft-deleted,
      // so this is a well-defined no-op rather than an error.
      const target = current ?? requested;
      if (!target) {
        reply({ type: "ic:deletions-flushed", sessionId: null, count: 0 });
        return;
      }
      commitSoftDeletes(queryClient, target)
        .then((count) =>
          reply({ type: "ic:deletions-flushed", sessionId: target, count }),
        )
        .catch((err: unknown) =>
          reply({
            type: "ic:deletions-flush-failed",
            sessionId: target,
            error: String((err as Error)?.message ?? err),
          }),
        );
    }

    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [queryClient]);
}
