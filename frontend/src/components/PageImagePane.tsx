import { useUiStore } from "@/store/uiStore";

export function PageImagePane() {
  const pageObjectUrl = useUiStore((s) => s.pageObjectUrl);

  return (
    <aside className="w-1/3 shrink-0 overflow-auto border-r border-slate-200 bg-slate-100 p-2">
      {pageObjectUrl ? (
        <img src={pageObjectUrl} alt="Manuscript page" className="max-w-full" />
      ) : (
        <p className="p-4 text-sm text-slate-400">Page image unavailable.</p>
      )}
    </aside>
  );
}
