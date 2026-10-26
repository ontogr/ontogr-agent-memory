"""A tenant world on the in-memory twins, for the consolidation tests."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ontogr_mem.consolidation.config import ConsolidationSettings
from ontogr_mem.data_models.entities import Entity
from ontogr_mem.graphstore.memory import (
    MemoryConsStore,
    MemoryEpisodeStore,
    MemoryResolverStore,
    MemoryTemporalStore,
)
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.policy import ResolutionPolicy
from ontogr_mem.resolution.resolver import EntityResolver
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.engine import TemporalEngine
from ontogr_mem.vectorstore import EmbeddingGateway, VectorStore
from ontogr_mem.vectorstore.memory import InMemoryBackend
from tests.support.embedder import MockEmbedder


T0 = datetime(2030, 1, 1, tzinfo=UTC)


class ManualClock:
    """A clock that moves only when the test advances it.

    It is callable, as the consolidation and temporal components expect, and it
    has ``now()``, as the resolution component expects.
    """

    def __init__(self, value: datetime = T0) -> None:
        """Start at ``value``."""
        self._value = value

    def __call__(self) -> datetime:
        """Return the current time."""
        return self._value

    def now(self) -> datetime:
        """Return the current time."""
        return self._value

    def advance(self, *, seconds: float = 0, days: float = 0) -> None:
        """Move the clock forward."""
        self._value += timedelta(seconds=seconds, days=days)


@dataclass
class ConsWorld:
    """Every port a stage can reach, all on in-memory twins of one tenant.

    Attributes:
        clock: The manual clock every component reads.
        store: The consolidation store.
        resolution: The entity resolver over its own store.
        episodes: Episode store that the temporal twin reads.
        temporal_store: The temporal store twin.
        temporal: The temporal engine over the twin.
        settings: Consolidation settings of the run.
    """

    clock: ManualClock
    store: MemoryConsStore
    resolution: EntityResolver
    episodes: MemoryEpisodeStore
    temporal_store: MemoryTemporalStore
    temporal: TemporalEngine
    settings: ConsolidationSettings

    async def add_entity(self, entity_id: str, name: str) -> Entity:
        """Save an active entity named ``name`` and return the stored copy."""
        now = self.clock()
        entity = Entity(
            entity_id=entity_id,
            type_label="Thing",
            canonical_name=name,
            strict_key=name.lower(),
            loose_key=name.lower(),
            created_at=now,
            updated_at=now,
        )
        return await self.resolution.store.save_entity(entity)

    async def ensure_entities(self, entity_ids: list[str]) -> None:
        """Create any ids that are not already in the resolver store."""
        for entity_id in entity_ids:
            if not await self.resolution.store.has_entity(entity_id):
                await self.add_entity(entity_id, entity_id)


async def new_world(settings: ConsolidationSettings | None = None) -> ConsWorld:
    """Build a world whose resolver and temporal engine run on the in-memory twins."""
    clock = ManualClock()
    vectors = VectorStore(InMemoryBackend(), EmbeddingGateway(MockEmbedder()))
    await vectors.ensure_ready()
    resolver_store = MemoryResolverStore()
    resolution = EntityResolver(
        resolver_store,
        vectors,
        ResolutionPolicy(),
        ResolverSettings(),
        clock,
    )
    episodes = MemoryEpisodeStore()
    temporal_store = MemoryTemporalStore(episodes, resolver_store)
    temporal = TemporalEngine(
        temporal_store,
        DictRelationPolicy({1: RelationRules(Cardinality.ONE)}),
        resolution.store,
        clock,
    )
    return ConsWorld(
        clock=clock,
        store=MemoryConsStore(),
        resolution=resolution,
        episodes=episodes,
        temporal_store=temporal_store,
        temporal=temporal,
        settings=settings or ConsolidationSettings(),
    )
