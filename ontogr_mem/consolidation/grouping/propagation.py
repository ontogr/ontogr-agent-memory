"""Label propagation over a weighted undirected graph.

Nodes are visited in sorted order. A node moves to the label with the highest total
edge weight around it, and keeps its own label when that label is among the best.
Ties go to the smallest label. The run stops at a fixed point or after the cap.

Each edge counts ``weight * (1 + shared neighbours)``. With unique starting labels,
plain propagation lets the smallest label flood across a bridge edge and merge two
dense groups. Edges inside a group share neighbours, so they outweigh the bridge.
"""

from collections import defaultdict


def label_propagation(
    neighbours: dict[str, dict[str, int]],
    *,
    max_iterations: int = 20,
    min_size: int = 2,
) -> list[list[str]]:
    """Return the communities of at least ``min_size`` nodes.

    Args:
        neighbours: Each node mapped to its neighbours and the edge weight to each.
            Edges are expected in both directions. Self-loops and neighbours that
            are not keys of this mapping are ignored.
        max_iterations: Upper bound on sweeps over all nodes. Zero leaves every
            node in its own group.
        min_size: Smallest group that is returned.

    Returns:
        Each community as a sorted list of node ids. The communities are ordered
        by their first id. The result depends only on the graph, not on the
        insertion order of the mappings.
    """
    nodes = sorted(neighbours)
    label = {n: n for n in nodes}
    adj = {
        n: {
            m: w
            * (1 + len((neighbours[n].keys() & neighbours.get(m, {}).keys()) - {n, m}))
            for m, w in neighbours[n].items()
            if m != n and m in neighbours
        }
        for n in nodes
    }
    for _ in range(max_iterations):
        changed = False
        for n in nodes:
            weight: dict[str, int] = defaultdict(int)
            for m, w in adj[n].items():
                weight[label[m]] += w
            if not weight:
                continue
            best = max(weight.values())
            winners = sorted(lab for lab, w in weight.items() if w == best)
            if label[n] in winners:
                continue
            label[n] = winners[0]
            changed = True
        if not changed:
            break
    groups: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        groups[label[n]].append(n)
    out = [sorted(g) for g in groups.values() if len(g) >= min_size]
    return sorted(out, key=lambda g: g[0])
