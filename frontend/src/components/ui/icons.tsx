import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

const base = {
  width: 14,
  height: 14,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

export function SelectIcon(props: IconProps) {
  return (
    <svg {...base} {...props} aria-hidden>
      <title>Select</title>
      <path d="M4 4h7" />
      <path d="M4 4v7" />
      <path d="M20 20h-7" />
      <path d="M20 20v-7" />
    </svg>
  );
}

export function RenameIcon(props: IconProps) {
  return (
    <svg {...base} {...props} aria-hidden>
      <title>Rename</title>
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z" />
    </svg>
  );
}

export function TrashIcon(props: IconProps) {
  return (
    <svg {...base} {...props} aria-hidden>
      <title>Delete</title>
      <path d="M3 6h18" />
      <path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    </svg>
  );
}

export function ChevronLeftIcon(props: IconProps) {
  return (
    <svg {...base} {...props} aria-hidden>
      <title>Collapse</title>
      <polyline points="15 18 9 12 15 6" />
    </svg>
  );
}

export function ChevronRightIcon(props: IconProps) {
  return (
    <svg {...base} {...props} aria-hidden>
      <title>Expand</title>
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}
