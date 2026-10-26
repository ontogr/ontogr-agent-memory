"""The entity graph the stages read, built over live graph tenants."""

from datetime import timedelta
from uuid import uuid4

from ontogr_mem.consolidation.archive import archive_item
from ontogr_mem.consolidation.graph import build_entity_graph, current_facts
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.data_models.entities import EntityStatus
from ontogr_mem.data_models.facts import Precision, TimeBasis
from ontogr_mem.data_models.observation import Observation, ObservationKind
from tests.support.consolidation import ConsWorld, new_world
from tests.support.graph_backend import GraphBackend


def ctx_for(world: ConsWorld) -> StageContext:
    """Build a stage context over every port of the world."""
    return StageContext(
        run_id="r1",
        clock=world.clock,
        store=world.store,
        settings=world.settings,
        resolution=world.resolution,
        temporal=world.temporal,
    )


async def add_fact(world: ConsWorld, subject: str, obj: str) -> None:
    """Ingest one current fact from ``subject`` to ``obj`` into the temporal store."""
    now = world.clock()
    await world.ensure_entities([subject, obj])
    await world.temporal.ingest_observations(
        [
            Observation(
                obs_id=uuid4().hex,
                slot_key=f"{subject}->{obj}",
                subject_id=subject,
                object_id=obj,
                relation_type_id=1,
                kind=ObservationKind.ASSERTION,
                episode_id=f"ep-{uuid4().hex}",
                occurred_at=now - timedelta(days=1),
                ingested_at=now,
                valid_from=now - timedelta(days=1),
                valid_from_precision=Precision.DAY,
                valid_to=None,
                valid_to_precision=None,
                time_basis=TimeBasis.STATED,
                inverse=False,
            )
        ]
    )


async def test_graph_has_active_entities_and_weights_parallel_facts(
    graph_backend: GraphBackend,
) -> None:
    """Parallel facts between a pair add up to one edge weight."""
    world = await new_world(
        graph_backend,
    )
    for entity_id in "abc":
        await world.add_entity(entity_id, entity_id.upper())
    await add_fact(world, "a", "b")
    await add_fact(world, "b", "a")
    await add_fact(world, "b", "c")

    graph = await build_entity_graph(ctx_for(world))

    assert set(graph.entities) == {"a", "b", "c"}
    assert graph.neighbours["a"] == {"b": 2}
    assert graph.neighbours["b"] == {"a": 2, "c": 1}
    assert graph.degree("b") == 2


async def test_graph_ignores_facts_to_unknown_entities_and_self_loops(
    graph_backend: GraphBackend,
) -> None:
    """A fact to an entity that is not active, or to itself, is not an edge."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    await add_fact(world, "a", "ghost")
    ghost = (await world.resolution.store.get_entities(["ghost"]))[0]
    await world.resolution.store.save_entity(
        ghost.model_copy(update={"status": EntityStatus.ARCHIVED}),
        bump_revision=False,
    )
    await add_fact(world, "a", "a")

    graph = await build_entity_graph(ctx_for(world))

    assert graph.neighbours == {"a": {}}
    assert graph.facts == ()


async def test_graph_follows_a_merged_endpoint_to_its_survivor(
    graph_backend: GraphBackend,
) -> None:
    """A fact that names a merged entity is an edge to the survivor."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    await world.add_entity("b", "B")
    old = await world.add_entity("old", "Old")
    merged = old.model_copy(update={"status": EntityStatus.MERGED, "merged_into": "b"})
    await world.resolution.save_entity(merged, bump_revision=False)
    await add_fact(world, "a", "old")

    graph = await build_entity_graph(ctx_for(world))

    assert graph.neighbours["a"] == {"b": 1}


async def test_graph_resolves_all_merged_endpoints_in_batches(
    graph_backend: GraphBackend,
) -> None:
    """Distinct merged endpoints are resolved with one batched store read."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    await world.add_entity("b", "B")
    for old_id in ("old1", "old2"):
        old = await world.add_entity(old_id, old_id)
        await world.resolution.save_entity(
            old.model_copy(update={"status": EntityStatus.MERGED, "merged_into": "b"}),
            bump_revision=False,
        )
    await add_fact(world, "a", "old1")
    await add_fact(world, "a", "old2")
    resolution = CountingResolution(world.resolution)
    ctx = ctx_for(world)
    ctx.resolution = resolution

    graph = await build_entity_graph(ctx)

    assert graph.neighbours["a"] == {"b": 2}
    assert resolution.requests == [(["old1", "old2"], False)]


async def test_building_the_graph_twice_reuses_the_same_reads(
    graph_backend: GraphBackend,
) -> None:
    """A stage context reuses its graph rather than reading the graph twice."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    await world.add_entity("b", "B")
    await add_fact(world, "a", "b")
    resolution = CountingResolution(world.resolution)
    temporal = CountingTemporal(world.temporal)
    ctx = ctx_for(world)
    ctx.resolution = resolution
    ctx.temporal = temporal

    first = await build_entity_graph(ctx)
    second = await build_entity_graph(ctx)

    assert second == first
    assert resolution.active_reads == 1
    assert temporal.fact_reads == 1


class CountingResolution:
    """Count graph reads while delegating to an entity resolver."""

    def __init__(self, resolution) -> None:
        """Keep the resolver and start with no reads."""
        self._resolution = resolution
        self.active_reads = 0
        self.requests = []

    async def active_entities(self):
        """Return the active entities and count the read."""
        self.active_reads += 1
        return await self._resolution.active_entities()

    async def get_entities(self, ids, *, follow_merged=False):
        """Return requested entities and record the batched request."""
        self.requests.append((list(ids), follow_merged))
        return await self._resolution.get_entities(ids, follow_merged=follow_merged)


class CountingTemporal:
    """Count fact reads while delegating to a temporal engine."""

    def __init__(self, temporal) -> None:
        """Keep the temporal engine and start with no reads."""
        self._temporal = temporal
        self.fact_reads = 0

    async def current_facts(self):
        """Return current facts and count the read."""
        self.fact_reads += 1
        return await self._temporal.current_facts()


async def test_archived_entities_and_facts_leave_the_graph(
    graph_backend: GraphBackend,
) -> None:
    """An archived entity and an archived fact are not in the graph."""
    world = await new_world(
        graph_backend,
    )
    for entity_id in "abc":
        await world.add_entity(entity_id, entity_id.upper())
    await add_fact(world, "a", "b")
    await add_fact(world, "b", "c")
    ctx = ctx_for(world)
    await archive_item(ctx, "c", "entity", "cold")
    fact_id = next(f.fact_id for f in await current_facts(ctx) if f.subject_id == "a")
    await archive_item(ctx, fact_id, "fact", "cold")

    graph = await build_entity_graph(ctx)

    assert set(graph.entities) == {"a", "b"}
    assert graph.facts == ()
