import type { WheelEvent } from "react";

/**
 * Wheel handler for a dialog's greyed-out backdrop (the fixed-inset overlay,
 * not the dialog box itself). The overlay has no scrollable content of its
 * own, and when IC is running embedded in mothra's iframe a wheel event over
 * it never bubbles out to the parent document by itself — iframed content is
 * its own scroll boundary. Forward the delta so mothra's page scrolls
 * underneath instead of the scroll silently doing nothing. Standalone (IC
 * opened directly, not embedded) has no parent to forward to, so this is a
 * no-op there.
 */
export function forwardOverlayScroll(e: WheelEvent): void {
  if (window.parent === window) return;
  window.parent.postMessage(
    { type: "ic:overlay-scroll", deltaX: e.deltaX, deltaY: e.deltaY },
    "*",
  );
}
