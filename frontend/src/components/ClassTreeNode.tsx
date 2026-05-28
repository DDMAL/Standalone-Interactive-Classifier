import { RenameIcon, SelectIcon, TrashIcon } from "@/components/ui/icons";
import type { ClassNode } from "@/lib/classTree";
import { clsx } from "clsx";
import {
  type KeyboardEvent as ReactKeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

interface ClassTreeNodeProps {
  node: ClassNode;
  /** Working-set count for every class path. Built once at the panel root. */
  countsByClass: Map<string, number>;
  depth: number;
  onSelect: (node: ClassNode) => void;
  onRename: (node: ClassNode, newSegment: string) => void;
  onDelete: (node: ClassNode) => void;
  /** Path of the node whose rename input is currently open (one at a time). */
  renamingPath: string | null;
  setRenamingPath: (path: string | null) => void;
}

const INDENT_PX = 12;

export function ClassTreeNode({
  node,
  countsByClass,
  depth,
  onSelect,
  onRename,
  onDelete,
  renamingPath,
  setRenamingPath,
}: ClassTreeNodeProps) {
  const [expanded, setExpanded] = useState(depth < 1);
  const isRenaming = renamingPath === node.path;
  const hasChildren = node.children.length > 0;

  // Count for this node is its own glyphs plus all descendants — that's what
  // the "Select" action would gather and matches user intuition.
  const count = useMemo(() => {
    let total = 0;
    const prefix = `${node.path}.`;
    for (const [path, n] of countsByClass) {
      if (path === node.path || path.startsWith(prefix)) total += n;
    }
    return total;
  }, [countsByClass, node.path]);

  const orphan = node.isLeafClass === false && !hasChildren;

  return (
    <li>
      <div
        className="group flex items-center gap-1 rounded px-1 py-0.5 hover:bg-slate-100"
        style={{ paddingLeft: depth * INDENT_PX + 4 }}
      >
        <button
          type="button"
          onClick={() => hasChildren && setExpanded((v) => !v)}
          aria-label={hasChildren ? (expanded ? "Collapse" : "Expand") : "Leaf"}
          className={clsx(
            "inline-flex h-4 w-4 shrink-0 items-center justify-center text-[10px] text-slate-400",
            !hasChildren && "invisible",
          )}
        >
          <span
            aria-hidden
            className={clsx("transition-transform", expanded && "rotate-90")}
          >
            ▶
          </span>
        </button>

        {isRenaming ? (
          <RenameInput
            initial={node.segment}
            onCancel={() => setRenamingPath(null)}
            onCommit={(next) => {
              setRenamingPath(null);
              if (next.trim() && next !== node.segment) {
                onRename(node, next.trim());
              }
            }}
          />
        ) : (
          <button
            type="button"
            onClick={() => onSelect(node)}
            title={`Select all glyphs in ${node.path}`}
            className={clsx(
              "min-w-0 flex-1 truncate text-left text-sm",
              orphan
                ? "italic text-slate-400"
                : node.isLeafClass
                  ? "text-slate-800"
                  : "text-slate-700",
            )}
          >
            {node.segment}
            {count > 0 && (
              <span className="ml-1 inline-block rounded-full bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">
                {count}
              </span>
            )}
          </button>
        )}

        {!isRenaming && (
          <div className="ml-auto hidden gap-0.5 group-hover:flex group-focus-within:flex">
            <ActionButton
              label="Select"
              onClick={() => onSelect(node)}
              icon={<SelectIcon />}
            />
            <ActionButton
              label="Rename"
              onClick={() => setRenamingPath(node.path)}
              icon={<RenameIcon />}
            />
            <ActionButton
              label="Delete"
              onClick={() => onDelete(node)}
              icon={<TrashIcon />}
              danger
            />
          </div>
        )}
      </div>

      {hasChildren && expanded && (
        <ul>
          {node.children.map((child) => (
            <ClassTreeNode
              key={child.path}
              node={child}
              countsByClass={countsByClass}
              depth={depth + 1}
              onSelect={onSelect}
              onRename={onRename}
              onDelete={onDelete}
              renamingPath={renamingPath}
              setRenamingPath={setRenamingPath}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

function ActionButton({
  label,
  onClick,
  icon,
  danger,
}: {
  label: string;
  onClick: () => void;
  icon: React.ReactNode;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className={clsx(
        "inline-flex h-5 w-5 items-center justify-center rounded text-slate-500",
        danger ? "hover:bg-red-100 hover:text-red-700" : "hover:bg-slate-200",
      )}
    >
      {icon}
    </button>
  );
}

function RenameInput({
  initial,
  onCancel,
  onCommit,
}: {
  initial: string;
  onCancel: () => void;
  onCommit: (next: string) => void;
}) {
  const [value, setValue] = useState(initial);
  const ref = useRef<HTMLInputElement>(null);

  useEffect(() => {
    ref.current?.focus();
    ref.current?.select();
  }, []);

  function handleKeyDown(e: ReactKeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      e.preventDefault();
      onCommit(value);
    } else if (e.key === "Escape") {
      e.preventDefault();
      onCancel();
    }
  }

  return (
    <input
      ref={ref}
      type="text"
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onKeyDown={handleKeyDown}
      onBlur={() => onCommit(value)}
      className="min-w-0 flex-1 rounded border border-blue-300 bg-white px-1 py-0.5 text-sm"
    />
  );
}
