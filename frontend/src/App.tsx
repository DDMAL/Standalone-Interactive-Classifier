import { useEffect } from "react";

import { SessionView } from "@/components/SessionView";
import { UploadView } from "@/components/UploadView";
import { useUiStore } from "@/store/uiStore";

// Deep-link params an embedding host (mothra) may pass:
//  ?session=<id>  — open straight into an existing session (page served by
//                   the API at /sessions/<id>/page, since this frontend never
//                   uploaded it).
//  ?staged=<id>   — open the create-session screen with the page + bboxes
//                   already staged; the user only adds training data +
//                   vocabulary, then starts the session.
//  ?manage=1&project_id=<id> — open the saved-sessions management page scoped
//                   to that mothra project (mothra iframes this in its "manage
//                   IC sessions" modal). Clicking an in-progress session hands
//                   it back to the host via "ic:resume-session" when embedded
//                   (see the manage branch below); standalone it resumes in
//                   place via setSession.
const params = new URLSearchParams(window.location.search);
const deepLinkSessionId = params.get("session");
const stagedId = params.get("staged") ?? undefined;

export default function App() {
  const sessionId = useUiStore((s) => s.sessionId);
  const setSession = useUiStore((s) => s.setSession);

  useEffect(() => {
    if (deepLinkSessionId && !useUiStore.getState().sessionId) {
      setSession(deepLinkSessionId, `/sessions/${deepLinkSessionId}/page`);
      // Resuming a saved session skips the create-session screen (and thus
      // useCreateSession's notify), so tell the embedding host directly —
      // otherwise the host would wait forever for "ic:session-created"
      // before enabling its encode/complete action.
      if (window.parent !== window) {
        window.parent.postMessage(
          { type: "ic:session-created", sessionId: deepLinkSessionId },
          "*",
        );
      }
    }
  }, [setSession]);

  if (sessionId) return <SessionView sessionId={sessionId} />;
  // Don't flash the upload form while the deep-linked session is being loaded.
  if (deepLinkSessionId) return null;
  // Saved-sessions management page — full-width, always-open resume list
  // scoped to the mothra project.
  //
  // Embedded (mothra's "manage IC sessions" modal): clicking an in-progress
  // session must NOT swap this list for a SessionView inside the modal — the
  // host has a whole IC stage for that, with the page filmstrip, clef controls
  // and encode queue a bare in-modal SessionView can't reach. So hand the
  // click back and let mothra navigate; it re-opens the very same session,
  // since sessions are unique per (project_id, image_id).
  // Standalone (not iframed): no host to hand off to, so resume in place.
  if (manage)
    return (
      <div className="flex h-full w-full justify-center bg-slate-50 p-4">
        <SessionResumeList
          projectId={
            manageProjectId != null ? Number(manageProjectId) : undefined
          }
          standalonePage
          onResume={
            window.parent !== window
              ? (s) =>
                  window.parent.postMessage(
                    {
                      type: "ic:resume-session",
                      sessionId: s.id,
                      imageId: s.image_id,
                      sourceName: s.source_name,
                    },
                    "*",
                  )
              : undefined
          }
        />
      </div>
    );
  return <UploadView stagedId={stagedId} />;
}
