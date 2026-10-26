"""Bounded breadth-first reads over a tenant's visible fact graph."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.retrieval.errors import RetrievalError
from ontogr_mem.retrieval.keyword import KeywordKind, KeywordSearch
from ontogr_mem.temporal.query import FactView, TemporalQuery
from ontogr_mem.temporal.reader import FactReader
from ontogr_mem.temporal.temporal_store import TemporalStore


MAX_DEPTH = 4
MAX_NODES = 500
KEYWORD_SEED_LIMIT = 5
DEFAULT_SEED_COUNT = 3


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One entity in the bounded graph view.

    Attributes:
        entity_id: Id of the entity.
        name: Canonical entity name, or its id when no row exists.
        type_label: Entity type, or an empty string when no row exists.
        degree: Number of visible fact endpoints touching the entity.
    """

    entity_id: str
    name: str
    type_label: str
    degree: int


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """One visible fact connecting two returned nodes.

    Attributes:
        fact_id: Stable id of the fact.
        version: Version returned by the temporal query.
        subject_id: Subject entity id.
        object_id: Object entity id.
        relation_type_id: Relation type id.
        statement: Annotated fact statement.
        status: Lifecycle state of this version.
    """

    fact_id: str
    version: int
    subject_id: str
    object_id: str
    relation_type_id: int
    statement: str
    status: FactStatus


@dataclass(frozen=True, slots=True)
class Neighbourhood:
    """The nodes and edges reached from a set of seeds.

    Attributes:
        nodes: Returned entity nodes, ordered by id.
        edges: Induced fact edges, ordered by fact id and version.
        seeds: Valid starting ids used by the walk.
        truncated: Whether an unseen node was omitted at the node limit.
    """

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    seeds: tuple[str, ...]
    truncated: bool


async def neighbourhood(
    store: TemporalStore,
    resolver: ResolverStore,
    keywords: KeywordSearch,
    reader: FactReader,
    *,
    now: datetime,
    seeds: Sequence[str] = (),
    query: str = "",
    depth: int = 2,
    max_nodes: int = 100,
    include_history: bool = False,
) -> Neighbourhood:
    """Walk the visible graph from explicit, keyword or default seeds.

    Args:
        store: Tenant-bound temporal store.
        resolver: Resolver store for entity names and types.
        keywords: Keyword search used when no explicit seed is visible.
        reader: Reader that annotates only edges in the result.
        now: Present time used by the temporal query.
        seeds: Optional starting entity ids.
        query: Keyword query used to find starting entities.
        depth: Maximum number of hops, from zero through ``MAX_DEPTH``.
        max_nodes: Maximum returned nodes, from one through ``MAX_NODES``.
        include_history: Include closed, corrected and withdrawn versions.

    Returns:
        A sorted node and edge view with the actual seeds and truncation state.

    Raises:
        RetrievalError: A depth or node limit is outside its supported range.
    """
    if not 0 <= depth <= MAX_DEPTH:
        raise RetrievalError(f"depth must be between 0 and {MAX_DEPTH}")
    if not 1 <= max_nodes <= MAX_NODES:
        raise RetrievalError(f"max_nodes must be between 1 and {MAX_NODES}")

    temporal_query = TemporalQuery(now=now, include_history=include_history)
    views = await store.query(temporal_query)
    adjacency, degree = _graph_index(views)
    valid_seeds = await _select_seeds(seeds, query, keywords, adjacency, degree)
    seen, truncated = _walk(adjacency, valid_seeds, depth, max_nodes)
    edges = await _induced_edges(views, seen, reader, temporal_query)
    entities = await resolver.get_entities(sorted(seen), follow_merged=True)
    entities_by_id = {entity.entity_id: entity for entity in entities}
    nodes = []
    for entity_id in sorted(seen):
        entity = entities_by_id.get(entity_id)
        nodes.append(
            GraphNode(
                entity_id=entity_id,
                name=entity.canonical_name if entity else entity_id,
                type_label=entity.type_label if entity else "",
                degree=degree[entity_id],
            )
        )
    return Neighbourhood(tuple(nodes), edges, tuple(valid_seeds), truncated)


def _graph_index(
    views: Sequence[FactView],
) -> tuple[dict[str, list[str]], Counter[str]]:
    """Index both endpoints, counting every visible fact version."""
    adjacency: dict[str, list[str]] = defaultdict(list)
    degree: Counter[str] = Counter()
    for view in views:
        adjacency[view.subject_id].append(view.object_id)
        adjacency[view.object_id].append(view.subject_id)
        degree[view.subject_id] += 1
        degree[view.object_id] += 1
    return adjacency, degree


async def _select_seeds(
    seeds: Sequence[str],
    query: str,
    keywords: KeywordSearch,
    adjacency: dict[str, list[str]],
    degree: Counter[str],
) -> list[str]:
    """Prefer visible explicit seeds, then keyword hits, then highest degrees."""
    valid_seeds = sorted({entity_id for entity_id in seeds if entity_id in adjacency})
    if not valid_seeds:
        hits = await keywords.search(
            KeywordKind.ENTITY, query, limit=KEYWORD_SEED_LIMIT
        )
        valid_seeds = sorted({hit.id for hit in hits if hit.id in adjacency})
    if not valid_seeds:
        valid_seeds = sorted(
            degree, key=lambda entity_id: (-degree[entity_id], entity_id)
        )[:DEFAULT_SEED_COUNT]
    return valid_seeds


def _walk(
    adjacency: dict[str, list[str]],
    seeds: list[str],
    depth: int,
    max_nodes: int,
) -> tuple[set[str], bool]:
    """Walk breadth-first with stable ordering and a hard node bound."""
    hops: dict[str, int] = {}
    truncated = False
    frontier = seeds
    for hop in range(depth + 1):
        next_frontier: set[str] = set()
        for entity_id in frontier:
            if entity_id in hops:
                continue
            if len(hops) == max_nodes:
                truncated = True
                break
            hops[entity_id] = hop
            if hop < depth:
                next_frontier.update(
                    neighbour
                    for neighbour in adjacency[entity_id]
                    if neighbour not in hops
                )
        if truncated:
            break
        frontier = sorted(next_frontier)
    return set(hops), truncated


async def _induced_edges(
    views: Sequence[FactView],
    seen: set[str],
    reader: FactReader,
    temporal_query: TemporalQuery,
) -> tuple[GraphEdge, ...]:
    """Annotate only facts whose two endpoints survived the bounded walk."""
    edge_views = [
        view for view in views if view.subject_id in seen and view.object_id in seen
    ]
    annotations = await reader.by_ids(
        sorted({view.fact_id for view in edge_views}), temporal_query
    )
    annotated_by_key = {(view.fact_id, view.version): view for view in annotations}
    return tuple(
        GraphEdge(
            fact_id=view.fact_id,
            version=view.version,
            subject_id=view.subject_id,
            object_id=view.object_id,
            relation_type_id=view.relation_type_id,
            statement=annotated_by_key[(view.fact_id, view.version)].statement,
            status=view.status,
        )
        for view in sorted(edge_views, key=lambda item: (item.fact_id, item.version))
    )
