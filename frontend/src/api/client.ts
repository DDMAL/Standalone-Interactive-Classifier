import { ApiError, type ErrorCode } from "@/types/api";

// In dev, Vite proxies "/sessions" to the API (see vite.config.ts), so an
// empty base means same-origin requests. VITE_API_BASE is a fallback for
// non-proxied builds.
const BASE = import.meta.env.DEV ? "" : (import.meta.env.VITE_API_BASE ?? "");

async function toApiError(res: Response): Promise<ApiError> {
  let detail = res.statusText;
  let code: ErrorCode | "unknown" = "unknown";
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") detail = body.detail;
    if (typeof body?.code === "string") code = body.code;
  } catch {
    // non-JSON error body; keep statusText
  }
  return new ApiError(detail, code, res.status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) throw await toApiError(res);
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const http = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: "POST",
      headers:
        body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  postForm: <T>(path: string, form: FormData) =>
    request<T>(path, { method: "POST", body: form }),
  delete: (path: string) => request<void>(path, { method: "DELETE" }),
};

// Raw blob fetch for the XML export, which is not JSON.
export async function postForBlob(
  path: string,
): Promise<{ blob: Blob; filename: string }> {
  const res = await fetch(`${BASE}${path}`, { method: "POST" });
  if (!res.ok) throw await toApiError(res);
  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition") ?? "";
  const match = disposition.match(/filename="?([^"]+)"?/);
  return { blob, filename: match?.[1] ?? "export.xml" };
}
