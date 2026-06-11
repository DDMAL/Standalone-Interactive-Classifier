import type { GlyphDTO } from "@/types/api";

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export function rectFromGlyph(g: GlyphDTO): Rect {
  return { x: g.ulx, y: g.uly, w: g.ncols, h: g.nrows };
}

export function rectFromAnchor(
  ax: number,
  ay: number,
  bx: number,
  by: number,
): Rect {
  return {
    x: Math.min(ax, bx),
    y: Math.min(ay, by),
    w: Math.abs(ax - bx),
    h: Math.abs(ay - by),
  };
}

export function intersects(a: Rect, b: Rect): boolean {
  return (
    a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y
  );
}

/**
 * Convert a screen-space pointer position into image-space (viewBox)
 * coordinates, using the overlay SVG's current bounding rect. Works
 * regardless of any CSS transform applied to an ancestor, because the
 * bounding rect reflects the post-transform layout.
 */
export function screenToImage(
  clientX: number,
  clientY: number,
  svg: SVGSVGElement,
  naturalW: number,
  naturalH: number,
): { x: number; y: number } {
  const r = svg.getBoundingClientRect();
  if (r.width === 0 || r.height === 0) return { x: 0, y: 0 };
  return {
    x: ((clientX - r.left) / r.width) * naturalW,
    y: ((clientY - r.top) / r.height) * naturalH,
  };
}
