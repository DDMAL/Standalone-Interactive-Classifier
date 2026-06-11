export type KeyAction =
  | { type: "zoomIn" }
  | { type: "zoomOut" }
  | { type: "zoomReset" }
  | { type: "clearSelection" }
  | { type: "pan"; dx: number; dy: number };

const PAN_STEP = 40;

export function actionForKey(e: KeyboardEvent): KeyAction | null {
  switch (e.key) {
    case "+":
    case "=":
      return { type: "zoomIn" };
    case "-":
    case "_":
      return { type: "zoomOut" };
    case "0":
      return { type: "zoomReset" };
    case "Escape":
      return { type: "clearSelection" };
    case "ArrowLeft":
      return { type: "pan", dx: PAN_STEP, dy: 0 };
    case "ArrowRight":
      return { type: "pan", dx: -PAN_STEP, dy: 0 };
    case "ArrowUp":
      return { type: "pan", dx: 0, dy: PAN_STEP };
    case "ArrowDown":
      return { type: "pan", dx: 0, dy: -PAN_STEP };
    default:
      return null;
  }
}

export function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (target.isContentEditable) return true;
  return false;
}
