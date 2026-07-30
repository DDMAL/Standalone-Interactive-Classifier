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
      className="fill-blue-500/10 stroke-blue-500 pointer-events-none"
      strokeDasharray="4 2"
    />
  );
}
