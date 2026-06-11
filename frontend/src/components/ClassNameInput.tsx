import * as Popover from "@radix-ui/react-popover";
import { clsx } from "clsx";
import {
  type KeyboardEvent as ReactKeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

interface ClassNameInputProps {
  value: string;
  onChange: (value: string) => void;
  options: string[];
  /** Invoked on Enter inside the input. If a suggestion is highlighted, its
   *  text is passed in. preventDefault is handled here so the parent form
   *  doesn't double-submit. */
  onApply?: (value: string) => void;
}

export function ClassNameInput({
  value,
  onChange,
  options,
  onApply,
}: ClassNameInputProps) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const listRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    const q = value.trim().toLowerCase();
    const matches = q
      ? options.filter((o) => o.toLowerCase().includes(q))
      : options;
    return matches.slice(0, 50);
  }, [value, options]);

  // New filtered list → drop the highlight; the previous index would point
  // somewhere irrelevant.
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset is driven by filtered identity, not value.
  useEffect(() => {
    setActiveIndex(-1);
  }, [filtered]);

  // Keep the highlighted item visible as the user arrows through the list.
  useEffect(() => {
    if (activeIndex < 0) return;
    const el = listRef.current?.children[activeIndex] as
      | HTMLElement
      | undefined;
    el?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  function handleKeyDown(e: ReactKeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (filtered.length === 0) return;
      setOpen(true);
      setActiveIndex((i) => (i + 1) % filtered.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (filtered.length === 0) return;
      setOpen(true);
      setActiveIndex((i) => (i <= 0 ? filtered.length - 1 : i - 1));
    } else if (e.key === "Escape") {
      if (open) {
        e.preventDefault();
        setOpen(false);
      }
    } else if (e.key === "Enter") {
      let toApply = value;
      if (activeIndex >= 0 && filtered[activeIndex]) {
        toApply = filtered[activeIndex];
        onChange(toApply);
        setOpen(false);
      }
      if (onApply) {
        e.preventDefault();
        onApply(toApply);
      }
    }
  }

  return (
    <Popover.Root open={open && filtered.length > 0} onOpenChange={setOpen}>
      <Popover.Anchor asChild>
        <input
          type="text"
          value={value}
          onChange={(e) => {
            onChange(e.target.value);
            setOpen(true);
          }}
          onKeyDown={handleKeyDown}
          onFocus={() => setOpen(true)}
          placeholder="Class name"
          className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm"
        />
      </Popover.Anchor>
      <Popover.Portal>
        <Popover.Content
          align="start"
          sideOffset={4}
          onOpenAutoFocus={(e) => e.preventDefault()}
          className="z-50 max-h-56 w-[var(--radix-popover-trigger-width)] overflow-auto rounded border border-slate-200 bg-white py-1 shadow-md"
          style={{ width: "var(--radix-popover-anchor-width)" }}
        >
          <div ref={listRef}>
            {filtered.map((option, i) => {
              const isActive = i === activeIndex;
              return (
                <button
                  key={option}
                  type="button"
                  // preventDefault on mousedown keeps focus in the input
                  // so the user can keep typing or hit Enter afterwards.
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => {
                    onChange(option);
                    setOpen(false);
                  }}
                  onMouseEnter={() => setActiveIndex(i)}
                  className={clsx(
                    "block w-full truncate px-3 py-1 text-left text-sm",
                    isActive ? "bg-blue-100" : "hover:bg-slate-100",
                  )}
                >
                  {option}
                </button>
              );
            })}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
