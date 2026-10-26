"""Entity cards and tenant-wide read summaries."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from ontogr_mem.data_models.entities import EntityStatus
from ontogr_mem.data_models.episode import Episode
from ontogr_mem.errors import AmbiguousNameError
from ontogr_mem.ingestion.store import EpisodeStore
from ontogr_mem.resolution.keys import strict_key
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.temporal.query import FactView, TemporalQuery
from ontogr_mem.temporal.reader import FactReader
from ontogr_mem.temporal.temporal_store import TemporalStore


MAX_ENTITY_FACTS = 200
EPISODE_PAGE_SIZE = 100


@dataclass(frozen=True, slots=True)
class EntityCard:
    """An entity and its latest visible fact versions.

    Attributes:
        entity_id: Canonical entity id.
        name: Canonical entity name.
        type_label: Entity type label.
        summary: Stored entity summary.
        status: Entity lifecycle state.
        merged_into: Survivor id for a merged entity, or ``None``.
        aliases: Alias text values in sorted order.
        facts: Latest visible version of each fact touching this entity.
        facts_truncated: Whether ``max_facts`` omitted any matching facts.
    """

    entity_id: str
    name: str
    type_label: str
    summary: str
    status: EntityStatus
    merged_into: str | None
    aliases: tuple[str, ...]
    facts: tuple[FactView, ...]
    facts_truncated: bool


@dataclass(frozen=True, slots=True)
class TenantStats:
    """Counts of a tenant's stored fact versions, entities and episodes.

    Attributes:
        fact_versions: Number of stored fact versions in every status.
        facts_by_status: Fact-version counts keyed by status value.
        entities: Number of stored entities in every status.
        entities_by_status: Entity counts keyed by status value.
        episodes: Number of stored episodes.
    """

    fact_versions: int
    facts_by_status: dict[str, int]
    entities: int
    entities_by_status: dict[str, int]
    episodes: int


async def entity_card(
    store: TemporalStore,
    resolver: ResolverStore,
    reader: FactReader,
    *,
    now: datetime,
    entity_id: str | None = None,
    name: str | None = None,
    include_history: bool = False,
    max_facts: int = 25,
) -> EntityCard | None:
    """Resolve an entity and return its aliases and latest fact views.

    Args:
        store: Tenant-bound temporal store.
        resolver: Tenant-bound entity and alias store.
        reader: Reader bound to the temporal store.
        now: Present time used by the temporal query.
        entity_id: Entity id to look up, or ``None``.
        name: Name resolved by its strict key, or ``None``.
        include_history: Include older stored versions before selecting each
            fact's latest version.
        max_facts: Maximum facts to return, from one through ``MAX_ENTITY_FACTS``.

    Returns:
        The card, or ``None`` when no entity matches.

    Raises:
        ValueError: The lookup keys or fact limit are invalid.
        AmbiguousNameError: The name resolves to distinct entities.
    """
    if (entity_id is None) == (name is None):
        raise ValueError("name exactly one of entity_id and name")
    if not 1 <= max_facts <= MAX_ENTITY_FACTS:
        raise ValueError(f"max_facts must be between 1 and {MAX_ENTITY_FACTS}")

    if entity_id is not None:
        entities = await resolver.get_entities([entity_id], follow_merged=False)
    else:
        entities = await resolver.entities_by_strict_key(strict_key(name or ""))
    unique_entities = {entity.entity_id: entity for entity in entities}
    if not unique_entities:
        return None
    if len(unique_entities) > 1:
        raise AmbiguousNameError(tuple(unique_entities))

    entity = next(iter(unique_entities.values()))
    aliases = await resolver.aliases_of(entity.entity_id)
    views = await reader.touching(
        [entity.entity_id],
        TemporalQuery(now=now, include_history=include_history),
    )
    latest: dict[str, FactView] = {}
    for view in views:
        current = latest.get(view.fact_id)
        if current is None or view.version > current.version:
            latest[view.fact_id] = view
    ordered = sorted(
        latest.values(),
        key=lambda view: (view.valid_from, view.fact_id),
        reverse=True,
    )
    return EntityCard(
        entity_id=entity.entity_id,
        name=entity.canonical_name,
        type_label=entity.type_label,
        summary=entity.summary,
        status=entity.status,
        merged_into=entity.merged_into,
        aliases=tuple(sorted(alias.text for alias in aliases)),
        facts=tuple(ordered[:max_facts]),
        facts_truncated=len(ordered) > max_facts,
    )


async def tenant_stats(
    store: TemporalStore,
    resolver: ResolverStore,
    episodes: EpisodeStore,
    *,
    now: datetime,
) -> TenantStats:
    """Count all stored fact versions, entity statuses and tenant episodes.

    Args:
        store: Tenant-bound temporal store.
        resolver: Tenant-bound entity store.
        episodes: Tenant-bound, paginated episode store.
        now: Present time used by the temporal query.

    Returns:
        Counts for every stored fact version, entity and episode.
    """
    facts = await store.query(TemporalQuery(now=now, include_history=True))
    entities = await resolver.all_entities()
    episode_rows = await _all_episodes(episodes)
    fact_counts = Counter(view.status.value for view in facts)
    entity_counts = Counter(entity.status.value for entity in entities)
    return TenantStats(
        fact_versions=len(facts),
        facts_by_status=dict(sorted(fact_counts.items())),
        entities=len(entities),
        entities_by_status=dict(sorted(entity_counts.items())),
        episodes=len(episode_rows),
    )


async def _all_episodes(episodes: EpisodeStore) -> list[Episode]:
    """Read every episode through the store's newest-first keyset pages."""
    rows: list[Episode] = []
    before = None
    while True:
        page = await episodes.list_episodes(limit=EPISODE_PAGE_SIZE, before=before)
        rows.extend(page)
        if len(page) < EPISODE_PAGE_SIZE:
            return rows
        last = page[-1]
        before = (last.ingested_at, last.id)
