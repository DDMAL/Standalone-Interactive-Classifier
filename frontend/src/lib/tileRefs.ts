// Module-scoped registry that lets PageOverlay-driven selection scroll the
// corresponding grid tile into view without threading refs through the tree.
const tileEls = new Map<string, HTMLElement>();

export function registerTile(id: string, el: HTMLElement | null): void {
  if (el === null) tileEls.delete(id);
  else tileEls.set(id, el);
}

export function getTileEl(id: string): HTMLElement | undefined {
  return tileEls.get(id);
}
