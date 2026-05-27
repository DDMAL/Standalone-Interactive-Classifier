import { completeSession } from "@/api/sessions";
import { downloadBlob } from "@/lib/download";
import { useMutation } from "@tanstack/react-query";

export function useComplete(sessionId: string) {
  return useMutation({
    mutationFn: () => completeSession(sessionId),
    onSuccess: ({ blob, filename }) => downloadBlob(blob, filename),
  });
}
