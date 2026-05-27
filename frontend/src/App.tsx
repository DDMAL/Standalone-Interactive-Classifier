import { SessionView } from "@/components/SessionView";
import { UploadView } from "@/components/UploadView";
import { useUiStore } from "@/store/uiStore";

export default function App() {
  const sessionId = useUiStore((s) => s.sessionId);
  return sessionId ? <SessionView sessionId={sessionId} /> : <UploadView />;
}
