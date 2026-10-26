"""Tie detection between assertions of different values.

Two assertions of different values tie when their starts fall in one period at
the coarser precision. The relation is not transitive, so connected closure is
used: every member of a component is tied to every other member.
"""

from collections.abc import Sequence

from ontogr_mem.data_models.observation import Observation
from ontogr_mem.temporal.timeutil import same_span


def tie_components(assertions: Sequence[Observation]) -> dict[str, int]:
    """Map observations to the tie component they belong to.

    Args:
        assertions: The slot's dated assertions.

    Returns:
        The component id of every observation that takes part in a tie.
    """
    parent = list(range(len(assertions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, a in enumerate(assertions):
        for j in range(i + 1, len(assertions)):
            b = assertions[j]
            if a.value_id == b.value_id:
                continue
            if a.valid_from is None or b.valid_from is None:
                continue
            if same_span(
                a.valid_from,
                a.valid_from_precision,
                b.valid_from,
                b.valid_from_precision,
            ):
                parent[find(i)] = find(j)

    members: dict[int, list[int]] = {}
    for i in range(len(assertions)):
        members.setdefault(find(i), []).append(i)
    out: dict[str, int] = {}
    for root, indexes in members.items():
        if len({assertions[i].value_id for i in indexes}) > 1:
            for i in indexes:
                out[assertions[i].obs_id] = root
    return out
