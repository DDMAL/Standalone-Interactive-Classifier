import { clsx } from "clsx";
import { type ButtonHTMLAttributes, forwardRef } from "react";

type Variant = "primary" | "secondary" | "ghost";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
}

const styles: Record<Variant, string> = {
  primary:
    "bg-mothra-cyan text-white hover:bg-mothra-cyan-dark disabled:bg-mothra-cyan/50",
  secondary:
    "bg-mothra-cyan-faint text-mothra-teal hover:bg-mothra-cyan-muted disabled:opacity-50",
  ghost:
    "bg-transparent text-mothra-teal hover:bg-mothra-cyan-faint disabled:opacity-50",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  function Button({ variant = "primary", className, ...props }, ref) {
    return (
      <button
        ref={ref}
        type="button"
        className={clsx(
          "rounded px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed",
          styles[variant],
          className,
        )}
        {...props}
      />
    );
  },
);
