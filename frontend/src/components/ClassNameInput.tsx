import * as Popover from "@radix-ui/react-popover";
import { useMemo, useState } from "react";

interface ClassNameInputProps {
  value: string;
  onChange: (value: string) => void;
  options: string[];
}

export function ClassNameInput({
  value,
  onChange,
  options,
}: ClassNameInputProps) {
  const [open, setOpen] = useState(false);

  const filtered = useMemo(() => {
    const q = value.trim().toLowerCase();
    const matches = q
      ? options.filter((o) => o.toLowerCase().includes(q))
      : options;
    return matches.slice(0, 50);
  }, [value, options]);

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
          {filtered.map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => {
                onChange(option);
                setOpen(false);
              }}
              className="block w-full truncate px-3 py-1 text-left text-sm hover:bg-slate-100"
            >
              {option}
            </button>
          ))}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
