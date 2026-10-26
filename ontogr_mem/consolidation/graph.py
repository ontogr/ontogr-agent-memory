"""The entity graph the stages read: active entities joined by current facts.

Nodes are the active entities. Edges are the current facts, believed and valid now,
that are not archived and sit between two nodes. An endpoint that was merged away is
followed to its survivor. Parallel facts add up.
"""

from dataclasses import dataclass

from ontogr_mem.consolidation.models import ArchiveRecord, ArchiveState
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.data_models.entities import Entity
from ontogr_mem.temporal.query import FactView


@dataclass(frozen=True, slots=True)
class EntityGraph:
    """Active entities, undirected edge weights and the facts behind them.

    Attributes:
        entities: Active entities by id.
        neighbours: For each entity, its neighbours and the number of facts
            between the pair.
        facts: The facts that became edges.
    """

    entities: dict[str, Entity]
    neighbours: dict[str, dict[str, int]]
    facts: tuple[FactView, ...]

    def degree(self, entity_id: str) -> int:
        """Return the number of distinct neighbours of ``entity_id``."""
        return len(self.neighbours.get(entity_id, {}))


async def archived_records(ctx: StageContext) -> list[ArchiveRecord]:
    """Return the records of the items that are archived now."""
    return await ctx.store.find(ArchiveRecord, "state", ArchiveState.ARCHIVED)


async def current_facts(ctx: StageContext) -> tuple[FactView, ...]:
    """Return the current facts that are not archived."""
    if ctx.temporal is None:
        return ()
    hidden = {
        r.target_id for r in await archived_records(ctx) if r.target_kind == "fact"
    }
    views = await ctx.temporal.current_facts()
    return tuple(v for v in views if v.fact_id not in hidden)


async def build_entity_graph(ctx: StageContext) -> EntityGraph:
    """Read or reuse the graph of the tenant for this run pass."""
    if ctx.entity_graph is not None:
        return ctx.entity_graph
    if ctx.resolution is None:
        graph = EntityGraph({}, {}, ())
        ctx.entity_graph = graph
        return graph
    entities = {e.entity_id: e for e in await ctx.resolution.active_entities()}
    facts = await current_facts(ctx)
    ends = sorted({i for f in facts for i in (f.subject_id, f.object_id)})
    survivor_of: dict[str, str] = {i: i for i in ends if i in entities}
    rows: dict[str, Entity] = {}
    unresolved = set(ends) - survivor_of.keys()
    pending = set(unresolved)
    looked_up: set[str] = set()
    while pending:
        batch = pending - looked_up
        if not batch:
            break
        looked_up.update(batch)
        rows.update(
            (entity.entity_id, entity)
            for entity in await ctx.resolution.get_entities(sorted(batch))
        )
        pending = set()
        for entity_id in unresolved:
            current = entity_id
            visited: set[str] = set()
            while current not in entities and current not in visited:
                visited.add(current)
                row = rows.get(current)
                if row is None:
                    if current not in looked_up:
                        pending.add(current)
                    break
                if row.merged_into is None:
                    break
                current = row.merged_into
            if current in entities:
                survivor_of[entity_id] = current
    neighbours: dict[str, dict[str, int]] = {i: {} for i in entities}
    kept: list[FactView] = []
    for fact in facts:
        a = survivor_of.get(fact.subject_id)
        b = survivor_of.get(fact.object_id)
        if a is None or b is None or a == b:
            continue
        kept.append(fact)
        neighbours[a][b] = neighbours[a].get(b, 0) + 1
        neighbours[b][a] = neighbours[b].get(a, 0) + 1
    graph = EntityGraph(entities, neighbours, tuple(kept))
    ctx.entity_graph = graph
    return graph
