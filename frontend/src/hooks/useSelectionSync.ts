import { getTileEl } from "@/lib/tileRefs";
import { useUiStore } from "@/store/uiStore";
import { useEffect } from "react";

/**
 * Scrolls the grid tile corresponding to `primaryGlyphId` into view when it
 * changes. Mounted once at the SessionView level.
 */
export function useSelectionSync(): void {
  const primary = useUiStore((s) => s.primaryGlyphId);
  useEffect(() => {
    if (!primary) return;
    // Defer one tick so a freshly mounted tile (e.g. category expand) is
    // registered before we look it up.
    const handle = requestAnimationFrame(() => {
      const el = getTileEl(primary);
      el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
    return () => cancelAnimationFrame(handle);
  }, [primary]);
}
