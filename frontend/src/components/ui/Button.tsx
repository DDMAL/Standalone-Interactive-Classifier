import { clsx } from "clsx";
import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
}

const styles: Record<Variant, string> = {
  primary: "bg-blue-600 text-white hover:bg-blue-700 disabled:bg-blue-300",
  secondary:
    "bg-slate-200 text-slate-800 hover:bg-slate-300 disabled:opacity-50",
  ghost: "bg-transparent text-slate-700 hover:bg-slate-100 disabled:opacity-50",
};

export function Button({
  variant = "primary",
  className,
  ...props
}: ButtonProps) {
  return (
    <button
      type="button"
      className={clsx(
        "rounded px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed",
        styles[variant],
        className,
      )}
      {...props}
    />
  );
}
