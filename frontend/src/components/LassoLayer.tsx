import type { Rect } from "@/lib/bbox";

interface LassoLayerProps {
  rect: Rect | null;
}

export function LassoLayer({ rect }: LassoLayerProps) {
  if (!rect) return null;
  return (
    <rect
      x={rect.x}
      y={rect.y}
      width={rect.w}
      height={rect.h}
      strokeWidth={1}
      vectorEffect="non-scaling-stroke"
      className="fill-mothra-cyan/10 stroke-mothra-cyan pointer-events-none"
      strokeDasharray="4 2"
    />
  );
}
