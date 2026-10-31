"""A tenant world on a live graph backend, for the consolidation tests."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ontogr_mem.consolidation.config import ConsolidationSettings
from ontogr_mem.consolidation.store import ConsStore
from ontogr_mem.data_models.entities import Entity
from ontogr_mem.graphstore import QuerySpec, ScopedRunner
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.ingestion.store import EpisodeStore
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.policy import ResolutionPolicy
from ontogr_mem.resolution.resolver import EntityResolver
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.engine import TemporalEngine
from ontogr_mem.temporal.temporal_store import TemporalStore
from tests.support.graph_backend import GraphBackend
from tests.support.tenants import new_tenant


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
    """Every port a stage can reach, all bound to one live tenant.

    Attributes:
        clock: The manual clock every component reads.
        store: The consolidation store.
        resolution: The entity resolver over the tenant's resolver store.
        episodes: Episode store of the tenant.
        temporal_store: The temporal store of the tenant.
        temporal: The temporal engine over the temporal store.
        settings: Consolidation settings of the run.
        runner: Runs raw queries against the tenant's graph.
        tenant_keyed: Whether the graph keys nodes by tenant (Neo4j).
    """

    clock: ManualClock
    store: ConsStore
    resolution: EntityResolver
    episodes: EpisodeStore
    temporal_store: TemporalStore
    temporal: TemporalEngine
    settings: ConsolidationSettings
    runner: ScopedRunner
    tenant_keyed: bool

    async def seed_live_fact(self, subject: str, obj: str) -> None:
        """Add an active FACT edge from ``subject`` to ``obj`` that blocks deletion."""
        scope = TenantScope.from_mode(self.tenant_keyed)
        await self.runner.run(
            QuerySpec(
                f"MATCH {scope.node_pattern('s', 'Entity', '$subject')}, "
                f"{scope.node_pattern('o', 'Entity', '$object')} "
                "MERGE (s)-[f:FACT]->(o) "
                f"SET f.status = 'active'{scope.stamp_clause('f')}",
                {"subject": subject, "object": obj},
                read_only=False,
                idempotent=True,
                tag="test_seed_live_fact",
            )
        )

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


async def new_world(
    backend: GraphBackend, settings: ConsolidationSettings | None = None
) -> ConsWorld:
    """Open a fresh tenant on `backend` and build every port over it."""
    clock = ManualClock()
    tenant = new_tenant()
    stores = await backend.stores_for(tenant)
    runner = backend.store.runner(await backend.store.open(tenant))
    resolution = EntityResolver(
        stores.resolver_store,
        stores.vectors,
        ResolutionPolicy(),
        ResolverSettings(),
        clock,
    )
    temporal = TemporalEngine(
        stores.temporal_store,
        DictRelationPolicy({1: RelationRules(Cardinality.ONE)}),
        resolution.store,
        clock,
    )
    return ConsWorld(
        clock=clock,
        store=stores.cons_store,
        resolution=resolution,
        episodes=stores.episodes,
        temporal_store=stores.temporal_store,
        temporal=temporal,
        settings=settings or ConsolidationSettings(),
        runner=runner,
        tenant_keyed=backend.tenant_keyed,
    )
