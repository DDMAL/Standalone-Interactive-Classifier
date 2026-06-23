import { useEffect } from "react";

import { SessionView } from "@/components/SessionView";
import { UploadView } from "@/components/UploadView";
import { useUiStore } from "@/store/uiStore";

// When an embedding host (e.g. mothra) creates a session via the HTTP API and
// loads the SPA at `?session=<id>`, deep-link straight into that session
// instead of showing the upload form. The page image is served by the API at
// `/sessions/<id>/page`, since this frontend never performed the upload and so
// has no local object URL for it.
const deepLinkSessionId = new URLSearchParams(window.location.search).get(
  "session",
);

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
  return <UploadView />;
}
