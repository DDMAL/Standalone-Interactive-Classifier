import { ClassTreeNode } from "@/components/ClassTreeNode";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { ChevronLeftIcon, ChevronRightIcon } from "@/components/ui/icons";
import { useClassSelection } from "@/hooks/useClassSelection";
import { useDeleteClass } from "@/hooks/useDeleteClass";
import { useRenameClass } from "@/hooks/useRenameClass";
import { type ClassNode, buildClassTree } from "@/lib/classTree";
import { useUiStore } from "@/store/uiStore";
import type { SessionDTO } from "@/types/api";
import { useMemo, useState } from "react";

interface ClassTreePanelProps {
  sessionId: string;
  session: SessionDTO;
}

/**
 * Left rail: parses `session.class_names` into a tree, exposes per-node
 * Select / Rename / Delete actions, and toggles between expanded (200px)
 * and a collapsed 24px strip via uiStore.
 */
export function ClassTreePanel({ sessionId, session }: ClassTreePanelProps) {
  const collapsed = useUiStore((s) => s.classTreeCollapsed);
  const setCollapsed = useUiStore((s) => s.setClassTreeCollapsed);
  const deletedGlyphIds = useUiStore((s) => s.deletedGlyphIds);

  const renameClass = useRenameClass(sessionId);
  const deleteClassMut = useDeleteClass(sessionId);
  const selectByClass = useClassSelection();

  const [renamingPath, setRenamingPath] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ClassNode | null>(null);

  const tree = useMemo(
    () => buildClassTree(session.class_names),
    [session.class_names],
  );

  // Working-set counts keyed by exact class_name; subtree counts are computed
  // in the node itself by walking this map's keys.
  const countsByClass = useMemo(() => {
    const counts = new Map<string, number>();
    for (const g of session.glyphs) {
      if (deletedGlyphIds.has(g.id)) continue;
      if (!g.class_name) continue;
      counts.set(g.class_name, (counts.get(g.class_name) ?? 0) + 1);
    }
    return counts;
  }, [session.glyphs, deletedGlyphIds]);

  function handleSelect(node: ClassNode) {
    selectByClass(node.path, !node.isLeafClass || node.children.length > 0);
  }

  function handleRename(node: ClassNode, newSegment: string) {
    const segments = node.path.split(".");
    segments[segments.length - 1] = newSegment;
    const newPath = segments.join(".");
    if (newPath === node.path) return;
    renameClass.mutate({ name: node.path, newName: newPath });
  }

  function handleConfirmDelete() {
    if (!deleteTarget) return;
    deleteClassMut.mutate(deleteTarget.path, {
      onSettled: () => setDeleteTarget(null),
    });
  }

  if (collapsed) {
    return (
      <aside className="w-6 shrink-0 border-r border-slate-200 bg-white">
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          aria-label="Expand class tree"
          title="Expand class tree"
          className="flex h-full w-full items-start justify-center pt-2 text-slate-500 hover:bg-slate-50"
        >
          <ChevronRightIcon />
        </button>
      </aside>
    );
  }

  const descendantCount = deleteTarget ? countDescendants(deleteTarget) - 1 : 0;

  return (
    <aside className="flex w-52 shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-200 px-2 py-1.5">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Classes
        </span>
        <button
          type="button"
          onClick={() => setCollapsed(true)}
          aria-label="Collapse class tree"
          title="Collapse"
          className="inline-flex h-5 w-5 items-center justify-center rounded text-slate-500 hover:bg-slate-100"
        >
          <ChevronLeftIcon />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-1">
        {tree.length === 0 ? (
          <p className="p-2 text-xs text-slate-400">
            No classes yet — apply a class to a glyph to start populating this
            tree.
          </p>
        ) : (
          <ul>
            {tree.map((node) => (
              <ClassTreeNode
                key={node.path}
                node={node}
                countsByClass={countsByClass}
                depth={0}
                onSelect={handleSelect}
                onRename={handleRename}
                onDelete={(n) => setDeleteTarget(n)}
                renamingPath={renamingPath}
                setRenamingPath={setRenamingPath}
              />
            ))}
          </ul>
        )}
      </div>

      {(renameClass.isError || deleteClassMut.isError) && (
        <p className="border-t border-red-200 bg-red-50 p-2 text-xs text-red-700">
          {((renameClass.error ?? deleteClassMut.error) as Error)?.message}
        </p>
      )}

      <ConfirmDialog
        open={!!deleteTarget}
        onOpenChange={(o) => {
          if (!o) setDeleteTarget(null);
        }}
        title={
          deleteTarget
            ? descendantCount > 0
              ? `Delete class '${deleteTarget.path}' and ${descendantCount} descendant class${descendantCount === 1 ? "" : "es"}?`
              : `Delete class '${deleteTarget.path}'?`
            : "Delete class?"
        }
        description={
          <span>
            Glyphs already labelled with this class will keep their label but
            the autocomplete will no longer offer it.
          </span>
        }
        confirmLabel="Delete"
        destructive
        pending={deleteClassMut.isPending}
        onConfirm={handleConfirmDelete}
      />
    </aside>
  );
}

function countDescendants(node: ClassNode): number {
  let n = 1;
  for (const c of node.children) n += countDescendants(c);
  return n;
}
