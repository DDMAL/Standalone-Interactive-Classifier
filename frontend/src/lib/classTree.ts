export interface ClassNode {
  /** The leaf segment for this node — e.g. "punctum" in "neume.punctum". */
  segment: string;
  /** The full dotted path of this node — e.g. "neume.punctum". */
  path: string;
  children: ClassNode[];
  /** True iff `path` itself appears in the input list (vs. only as a prefix). */
  isLeafClass: boolean;
}

/**
 * Parse a flat list of class names into a tree on "."-separated namespaces.
 * Each input name becomes a node with `isLeafClass: true`; intermediate
 * prefixes that aren't themselves in the list become `isLeafClass: false`
 * structural nodes so the tree connects.
 */
export function buildClassTree(names: string[]): ClassNode[] {
  const roots: ClassNode[] = [];
  // Lookup keyed by full path so we don't rebuild branches for shared prefixes.
  const byPath = new Map<string, ClassNode>();

  // Sort up-front so the output is deterministic at every level.
  const sorted = [...names].sort();

  for (const name of sorted) {
    if (!name) continue;
    const parts = name.split(".");
    let parent: ClassNode | null = null;
    let pathSoFar = "";
    for (let i = 0; i < parts.length; i++) {
      const segment = parts[i];
      pathSoFar = pathSoFar ? `${pathSoFar}.${segment}` : segment;
      let node = byPath.get(pathSoFar);
      if (!node) {
        node = { segment, path: pathSoFar, children: [], isLeafClass: false };
        byPath.set(pathSoFar, node);
        if (parent) parent.children.push(node);
        else roots.push(node);
      }
      if (i === parts.length - 1) node.isLeafClass = true;
      parent = node;
    }
  }

  // Children inserted in input order; sort each level alphabetically.
  const sortRec = (nodes: ClassNode[]) => {
    nodes.sort((a, b) => a.segment.localeCompare(b.segment));
    for (const n of nodes) sortRec(n.children);
  };
  sortRec(roots);
  return roots;
}
