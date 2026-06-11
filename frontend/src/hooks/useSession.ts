import { getSession } from "@/api/sessions";
import type { SessionDTO } from "@/types/api";
import { useQuery } from "@tanstack/react-query";

export const sessionKey = (id: string) => ["session", id] as const;

export function useSession(id: string | null) {
  return useQuery<SessionDTO>({
    queryKey: sessionKey(id ?? ""),
    queryFn: () => getSession(id as string),
    enabled: !!id,
    staleTime: 0,
  });
}
