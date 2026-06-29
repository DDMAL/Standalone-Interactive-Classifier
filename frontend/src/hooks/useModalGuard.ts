import { useUiStore } from "@/store/uiStore";
import { useEffect } from "react";

/**
 * Registers an open modal dialog with the global UI store so that
 * window-level keyboard shortcuts (Enter-to-classify, type-to-focus,
 * zoom/pan, Esc-to-clear, Cmd/Ctrl+G, …) stand down while the dialog has
 * focus. Without this, pressing Enter inside e.g. the split dialog — which
 * has no text input to catch the key — would also fire the background
 * "Apply & reclassify" handler, silently classifying the very glyph the
 * user was trying to split.
 *
 * Increments on open and decrements on close/unmount, so the store holds a
 * count of currently-open dialogs (see `modalOpenCount`).
 */
export function useModalGuard(open: boolean): void {
  const openModal = useUiStore((s) => s.openModal);
  const closeModal = useUiStore((s) => s.closeModal);
  useEffect(() => {
    if (!open) return;
    openModal();
    return () => closeModal();
  }, [open, openModal, closeModal]);
}
