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
const params = new URLSearchParams(window.location.search);
const deepLinkSessionId = params.get("session");
const stagedId = params.get("staged") ?? undefined;

export default function App() {
  const sessionId = useUiStore((s) => s.sessionId);
  const setSession = useUiStore((s) => s.setSession);

  useEffect(() => {
    if (deepLinkSessionId && !useUiStore.getState().sessionId) {
      setSession(deepLinkSessionId, `/sessions/${deepLinkSessionId}/page`);
    }
  }, [setSession]);

  if (sessionId) return <SessionView sessionId={sessionId} />;
  // Don't flash the upload form while the deep-linked session is being loaded.
  if (deepLinkSessionId) return null;
  return <UploadView stagedId={stagedId} />;
}
