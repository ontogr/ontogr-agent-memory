"""Archiving and restoring entities and facts, over live graph tenants."""

from datetime import timedelta
from uuid import uuid4

import pytest

from ontogr_mem.consolidation.archive import archive_item, restore_item
from ontogr_mem.consolidation.errors import ConsolidationError
from ontogr_mem.consolidation.graph import (
    archived_records,
    build_entity_graph,
    current_facts,
)
from ontogr_mem.consolidation.models import ArchiveRecord, ArchiveState
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.data_models.entities import Entity, EntityStatus
from ontogr_mem.data_models.facts import Precision, TimeBasis
from ontogr_mem.data_models.observation import Observation, ObservationKind
from tests.support.consolidation import ConsWorld, new_world
from tests.support.graph_backend import GraphBackend


def ctx_for(world: ConsWorld, *, resolution: bool = True) -> StageContext:
    """Build a stage context over the world, optionally without resolution."""
    return StageContext(
        run_id="r1",
        clock=world.clock,
        store=world.store,
        settings=world.settings,
        resolution=world.resolution if resolution else None,
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


async def stored_entity(world: ConsWorld, entity_id: str) -> Entity:
    """Return the stored entity with ``entity_id``."""
    found = await world.resolution.get_entities([entity_id])
    assert len(found) == 1
    return found[0]


async def test_archiving_an_entity_archives_it_and_writes_a_record(
    graph_backend: GraphBackend,
) -> None:
    """An archived entity is stored as archived and its record is listed."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    ctx = ctx_for(world)

    await archive_item(ctx, "a", "entity", "cold")

    assert (await stored_entity(world, "a")).status is EntityStatus.ARCHIVED
    records = await archived_records(ctx)
    assert [(r.target_id, r.target_kind, r.reason) for r in records] == [
        ("a", "entity", "cold")
    ]


async def test_archiving_an_entity_leaves_updated_at_and_revision_alone(
    graph_backend: GraphBackend,
) -> None:
    """Archiving is not a change to the entity, so recency does not move."""
    world = await new_world(
        graph_backend,
    )
    before = await world.add_entity("a", "A")
    world.clock.advance(days=100)

    await archive_item(ctx_for(world), "a", "entity", "cold")

    after = await stored_entity(world, "a")
    assert after.updated_at == before.updated_at
    assert after.revision == before.revision


async def test_archiving_a_fact_hides_it_and_keeps_both_entities_active(
    graph_backend: GraphBackend,
) -> None:
    """An archived fact leaves the graph and never changes its entities."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    await world.add_entity("b", "B")
    await add_fact(world, "a", "b")
    ctx = ctx_for(world)
    fact_id = (await current_facts(ctx))[0].fact_id

    await archive_item(ctx, fact_id, "fact", "stale")

    assert (await build_entity_graph(ctx)).facts == ()
    assert (await stored_entity(world, "a")).status is EntityStatus.ACTIVE
    assert (await stored_entity(world, "b")).status is EntityStatus.ACTIVE


async def test_archiving_an_entity_that_is_not_active_is_refused(
    graph_backend: GraphBackend,
) -> None:
    """An unknown entity, or one already archived, cannot be archived again."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    ctx = ctx_for(world)
    await archive_item(ctx, "a", "entity", "cold")

    with pytest.raises(ConsolidationError):
        await archive_item(ctx, "a", "entity", "cold")
    with pytest.raises(ConsolidationError):
        await archive_item(ctx, "missing", "entity", "cold")


async def test_archiving_an_entity_needs_the_resolution_port(
    graph_backend: GraphBackend,
) -> None:
    """Archiving an entity without a resolution port is refused."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")

    with pytest.raises(ConsolidationError):
        await archive_item(ctx_for(world, resolution=False), "a", "entity", "cold")


async def test_restoring_an_entity_makes_it_active_with_a_new_revision(
    graph_backend: GraphBackend,
) -> None:
    """A restored entity is active again, at a newer revision, and not listed."""
    world = await new_world(
        graph_backend,
    )
    before = await world.add_entity("a", "A")
    ctx = ctx_for(world)
    await archive_item(ctx, "a", "entity", "cold")

    await restore_item(ctx, "a")

    after = await stored_entity(world, "a")
    assert after.status is EntityStatus.ACTIVE
    assert after.revision > before.revision
    assert after.updated_at == before.updated_at
    assert await archived_records(ctx) == []


async def test_restoring_an_entity_brings_its_edges_back_to_the_graph(
    graph_backend: GraphBackend,
) -> None:
    """A restored entity is a node of the graph again."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    await world.add_entity("b", "B")
    await add_fact(world, "a", "b")
    ctx = ctx_for(world)
    await archive_item(ctx, "b", "entity", "cold")

    await restore_item(ctx, "b")

    graph = await build_entity_graph(ctx)
    assert graph.neighbours["a"] == {"b": 1}


async def test_restoring_a_fact_returns_it_to_the_graph(
    graph_backend: GraphBackend,
) -> None:
    """A restored fact is an edge of the graph again."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    await world.add_entity("b", "B")
    await add_fact(world, "a", "b")
    ctx = ctx_for(world)
    fact_id = (await current_facts(ctx))[0].fact_id
    await archive_item(ctx, fact_id, "fact", "stale")

    await restore_item(ctx, fact_id)

    assert (await build_entity_graph(ctx)).neighbours["a"] == {"b": 1}


async def test_restoring_an_item_that_is_not_archived_is_refused(
    graph_backend: GraphBackend,
) -> None:
    """Only an archived item can be restored, and only once."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    ctx = ctx_for(world)

    with pytest.raises(ConsolidationError):
        await restore_item(ctx, "a")

    await archive_item(ctx, "a", "entity", "cold")
    await restore_item(ctx, "a")
    with pytest.raises(ConsolidationError):
        await restore_item(ctx, "a")


async def test_a_restored_record_is_kept_with_the_restored_state(
    graph_backend: GraphBackend,
) -> None:
    """The record of a restored item stays in the store with state RESTORED."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    ctx = ctx_for(world)
    await archive_item(ctx, "a", "entity", "cold")

    await restore_item(ctx, "a")

    record = await world.store.get(ArchiveRecord, "a")
    assert record is not None
    assert record.state is ArchiveState.RESTORED
    assert record.reason == "cold"
