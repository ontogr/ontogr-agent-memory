"""Shared in-memory worlds for P9i read integration tests."""

from dataclasses import replace
from datetime import UTC, datetime

from ontogr_mem.data_models import Alias, Entity
from ontogr_mem.graphstore.memory import (
    MemoryEpisodeStore,
    MemoryResolverStore,
    MemoryTemporalStore,
)
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.engine import TemporalEngine
from tests.support.clock import FixedClock
from tests.support.episodes import make_episode
from tests.unit.temporal.helpers import LocalRefResolver, obs


NOW = datetime(2026, 10, 22, tzinfo=UTC)


def entity(entity_id: str, *, name: str | None = None) -> Entity:
    """Build an active entity with a fixed clock."""
    canonical_name = name or entity_id
    key = canonical_name.casefold()
    return Entity(
        entity_id=entity_id,
        type_label="Person",
        canonical_name=canonical_name,
        strict_key=key,
        loose_key=key,
        created_at=NOW,
        updated_at=NOW,
    )


def alias(entity_id: str, text: str, alias_id: str) -> Alias:
    """Build an alias row for an entity."""
    key = text.casefold()
    return Alias(
        alias_id=alias_id,
        entity_id=entity_id,
        text=text,
        strict_key=key,
        loose_key=key,
    )


async def make_world(
    entity_ids: tuple[str, ...],
) -> tuple[
    MemoryEpisodeStore,
    MemoryResolverStore,
    MemoryTemporalStore,
    FixedClock,
    TemporalEngine,
]:
    """Build in-memory episode, resolver and temporal stores with an engine."""
    episodes = MemoryEpisodeStore()
    resolver = MemoryResolverStore()
    for entity_id in entity_ids:
        await resolver.save_entity(entity(entity_id))
    store = MemoryTemporalStore(episodes, resolver)
    clock = FixedClock(NOW)
    engine = TemporalEngine(
        store,
        DictRelationPolicy({1: RelationRules(Cardinality.ONE)}),
        LocalRefResolver(),
        clock.now,
    )
    return episodes, resolver, store, clock, engine


async def add_fact(
    episodes: MemoryEpisodeStore,
    engine: TemporalEngine,
    *,
    observation_id: str,
    episode_id: str,
    subject: str,
    object_id: str,
    slot: str,
    valid_from: datetime,
    statement: str,
) -> None:
    """Ingest one observation with caller-visible text and source."""
    episode = make_episode(episode_id, content=statement)
    await episodes.write_episodes([episode])
    observation = replace(
        obs(
            observation_id,
            object_id,
            valid_from,
            subject=subject,
            slot=slot,
            episode=episode_id,
        ),
        statement=statement,
        quote=statement,
        source_system="test",
    )
    await engine.ingest_observations([observation])
