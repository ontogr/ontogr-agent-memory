"""The prune and archive stage, on live graph tenants."""

from ontogr_mem.consolidation.archive import archive_item, restore_item
from ontogr_mem.consolidation.config import ConsolidationSettings, PruneSettings
from ontogr_mem.consolidation.graph import (
    archived_records,
    build_entity_graph,
    current_facts,
)
from ontogr_mem.consolidation.models import ArchiveRecord, StageStatus
from ontogr_mem.consolidation.signals import record_access
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.consolidation.stages.prune_archive import PruneArchiveStage
from ontogr_mem.consolidation.stages.scoring import ScoringStage
from ontogr_mem.data_models.entities import EntityStatus
from tests.integration.resolution.resolver_helpers import ex
from tests.support.consolidation import ConsWorld, new_world
from tests.support.graph_backend import GraphBackend
from tests.unit.temporal.helpers import at, obs


def live_settings(
    *,
    archive_below: float = 0.15,
    min_age_days: float = 30.0,
    retention_days: float = 180.0,
    delete_after_retention: bool = False,
) -> ConsolidationSettings:
    """Return settings with pruning turned on."""
    return ConsolidationSettings(
        _env_file=None,
        prune=PruneSettings(
            dry_run=False,
            archive_below=archive_below,
            min_age_days=min_age_days,
            retention_days=retention_days,
            delete_after_retention=delete_after_retention,
        ),
    )


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


async def entity_status(world: ConsWorld, entity_id: str) -> EntityStatus | None:
    """Return the stored status of ``entity_id``, or None when it is gone."""
    found = await world.resolution.get_entities([entity_id])
    return found[0].status if found else None


async def add_fact(world: ConsWorld, subject: str, obj: str) -> None:
    """Ingest one current fact from ``subject`` to ``obj`` into the temporal store."""
    await world.ensure_entities([subject, obj])
    await world.temporal.ingest_observations(
        [obs(f"o-{subject}-{obj}", obj, at(2010), subject=subject, slot=subject)]
    )


async def aged_world(
    graph_backend: GraphBackend, settings: ConsolidationSettings
) -> ConsWorld:
    """Return a world with a cold entity ``old`` and a hot entity ``keep``, scored."""
    world = await new_world(graph_backend, settings)
    await world.add_entity("old", "Old")
    await world.add_entity("keep", "Keep")
    world.clock.advance(days=400)
    for _ in range(21):
        await record_access(world.store, "keep", world.clock())
    await ScoringStage().run(ctx_for(world))
    return world


async def test_gate_skips_when_nothing_is_scored_or_archived(
    graph_backend: GraphBackend,
) -> None:
    """The stage has nothing to do on an empty tenant."""
    world = await new_world(graph_backend, live_settings())

    assert await PruneArchiveStage().gate(ctx_for(world)) == (
        "no scores and nothing archived"
    )


async def test_dry_run_is_the_default_and_changes_nothing(
    graph_backend: GraphBackend,
) -> None:
    """With default settings the cold item is reported and left in place."""
    world = await aged_world(graph_backend, ConsolidationSettings(_env_file=None))

    result = await PruneArchiveStage().run(ctx_for(world))

    assert result.status is StageStatus.DRY_RUN
    assert result.counts["to_archive"] == 1
    assert result.counts["archived"] == 0
    assert await entity_status(world, "old") is EntityStatus.ACTIVE
    assert await archived_records(ctx_for(world)) == []


async def test_cold_entity_is_archived_and_leaves_the_graph(
    graph_backend: GraphBackend,
) -> None:
    """A low-score entity past the minimum age is archived and leaves the graph."""
    world = await aged_world(graph_backend, live_settings())

    result = await PruneArchiveStage().run(ctx_for(world))

    assert result.status is StageStatus.OK
    assert result.counts["archived"] == 1
    assert await entity_status(world, "old") is EntityStatus.ARCHIVED
    graph = await build_entity_graph(ctx_for(world))
    assert set(graph.entities) == {"keep"}


async def test_hot_entity_is_not_archived(graph_backend: GraphBackend) -> None:
    """An entity whose score is above the threshold stays active."""
    world = await aged_world(graph_backend, live_settings())

    await PruneArchiveStage().run(ctx_for(world))

    assert await entity_status(world, "keep") is EntityStatus.ACTIVE


async def test_young_entity_is_archived_only_after_the_minimum_age(
    graph_backend: GraphBackend,
) -> None:
    """A low-score entity younger than the minimum age is archived later."""
    world = await new_world(graph_backend, live_settings(archive_below=1.0))
    await world.add_entity("new", "New")
    world.clock.advance(days=5)
    await ScoringStage().run(ctx_for(world))

    young = await PruneArchiveStage().run(ctx_for(world))

    assert young.counts["to_archive"] == 0
    assert await entity_status(world, "new") is EntityStatus.ACTIVE

    world.clock.advance(days=25)
    await ScoringStage().run(ctx_for(world))

    aged = await PruneArchiveStage().run(ctx_for(world))

    assert aged.counts["archived"] == 1
    assert await entity_status(world, "new") is EntityStatus.ARCHIVED


async def test_restore_brings_an_archived_entity_back(
    graph_backend: GraphBackend,
) -> None:
    """A restored entity is active again and no longer listed as archived."""
    world = await aged_world(graph_backend, live_settings())
    await PruneArchiveStage().run(ctx_for(world))

    await restore_item(ctx_for(world), "old")

    assert await entity_status(world, "old") is EntityStatus.ACTIVE
    assert await archived_records(ctx_for(world)) == []


async def test_retention_deletes_an_archived_entity_only_after_the_window(
    graph_backend: GraphBackend,
) -> None:
    """Retention waits for the full window, then deletes the entity and its record."""
    world = await aged_world(
        graph_backend, live_settings(retention_days=100, delete_after_retention=True)
    )
    await PruneArchiveStage().run(ctx_for(world))

    world.clock.advance(days=50)
    early = await PruneArchiveStage().run(ctx_for(world))

    assert early.counts["deleted"] == 0
    assert await entity_status(world, "old") is EntityStatus.ARCHIVED

    world.clock.advance(days=51)
    late = await PruneArchiveStage().run(ctx_for(world))

    assert late.counts["deleted"] == 1
    assert await entity_status(world, "old") is None
    assert await world.store.get(ArchiveRecord, "old") is None


async def test_retention_keeps_archived_entities_when_deletion_is_off(
    graph_backend: GraphBackend,
) -> None:
    """With delete_after_retention off, an item past retention stays archived."""
    world = await aged_world(graph_backend, live_settings(retention_days=1))
    await PruneArchiveStage().run(ctx_for(world))

    world.clock.advance(days=50)
    result = await PruneArchiveStage().run(ctx_for(world))

    assert result.counts["past_retention"] == 1
    assert result.counts["deleted"] == 0
    assert await entity_status(world, "old") is EntityStatus.ARCHIVED


async def test_retention_never_deletes_a_fact(graph_backend: GraphBackend) -> None:
    """A fact past retention stays archived, and the result says so."""
    world = await new_world(
        graph_backend, live_settings(retention_days=10, delete_after_retention=True)
    )
    await world.add_entity("a", "A")
    await world.add_entity("b", "B")
    await add_fact(world, "a", "b")
    fact_id = (await current_facts(ctx_for(world)))[0].fact_id
    await archive_item(ctx_for(world), fact_id, "fact", "stale")

    world.clock.advance(days=30)
    result = await PruneArchiveStage().run(ctx_for(world))

    assert result.counts["deleted"] == 0
    assert result.notes == [f"fact {fact_id} kept: deletion unsupported"]
    records = await archived_records(ctx_for(world))
    assert [r.target_id for r in records] == [fact_id]
    assert (await build_entity_graph(ctx_for(world))).facts == ()


async def test_entity_with_a_live_fact_is_kept_with_a_note(
    graph_backend: GraphBackend,
) -> None:
    """Deletion is refused while a live fact still points at the entity."""
    world = await new_world(
        graph_backend, live_settings(retention_days=10, delete_after_retention=True)
    )
    await world.add_entity("a", "A")
    await world.add_entity("b", "B")
    await world.seed_live_fact("a", "b")
    await archive_item(ctx_for(world), "a", "entity", "cold")

    world.clock.advance(days=30)
    result = await PruneArchiveStage().run(ctx_for(world))

    assert result.counts["deleted"] == 0
    assert result.notes == ["a kept: has facts"]
    assert await entity_status(world, "a") is EntityStatus.ARCHIVED


async def test_a_re_mentioned_item_is_skipped_by_retention_and_restored(
    graph_backend: GraphBackend,
) -> None:
    """An item accessed after it was archived is restored instead of deleted."""
    world = await aged_world(
        graph_backend, live_settings(retention_days=100, delete_after_retention=True)
    )
    await PruneArchiveStage().run(ctx_for(world))

    world.clock.advance(days=30)
    await record_access(world.store, "old", world.clock())
    await ScoringStage().run(ctx_for(world))

    world.clock.advance(days=80)
    result = await PruneArchiveStage().run(ctx_for(world))

    assert result.counts["deleted"] == 0
    assert result.counts["reactivated"] == 1
    assert await entity_status(world, "old") is EntityStatus.ACTIVE
    assert await archived_records(ctx_for(world)) == []


async def test_archiving_leaves_updated_at_untouched(
    graph_backend: GraphBackend,
) -> None:
    """Archiving is not a change to the entity, so its last update stays."""
    world = await aged_world(graph_backend, live_settings())
    before = (await world.resolution.get_entities(["old"]))[0]

    result = await PruneArchiveStage().run(ctx_for(world))

    assert result.counts["archived"] == 1
    after = (await world.resolution.get_entities(["old"]))[0]
    assert after.updated_at == before.updated_at


async def test_a_second_run_archives_nothing_more(graph_backend: GraphBackend) -> None:
    """Running the stage again changes no archive record."""
    world = await aged_world(graph_backend, live_settings())
    await PruneArchiveStage().run(ctx_for(world))
    first = await world.store.list(ArchiveRecord)

    second = await PruneArchiveStage().run(ctx_for(world))

    assert second.counts["to_archive"] == 0
    assert second.counts["archived"] == 0
    assert await world.store.list(ArchiveRecord) == first


async def test_a_mention_that_restored_an_entity_clears_its_archive_record(
    graph_backend: GraphBackend,
) -> None:
    """Resolution restores a matched archived entity, so it is no longer archived."""
    world = await new_world(graph_backend, live_settings())
    created = await world.resolution.resolve([ex("1", "Old", "Thing")])
    old_id = created.entities[0].entity_id
    world.clock.advance(days=400)
    await ScoringStage().run(ctx_for(world))
    await PruneArchiveStage().run(ctx_for(world))
    assert await entity_status(world, old_id) is EntityStatus.ARCHIVED

    again = await world.resolution.resolve([ex("1", "Old", "Thing", episode_id="e2")])
    assert again.entities[0].entity_id == old_id
    await ScoringStage().run(ctx_for(world))
    await PruneArchiveStage().run(ctx_for(world))

    assert await entity_status(world, old_id) is EntityStatus.ACTIVE
    assert await archived_records(ctx_for(world)) == []


async def test_the_archive_record_of_a_deleted_entity_is_dropped(
    graph_backend: GraphBackend,
) -> None:
    """An entity removed elsewhere does not make the stage fail on every run."""
    world = await aged_world(graph_backend, live_settings())
    await PruneArchiveStage().run(ctx_for(world))
    await world.resolution.delete("old")

    result = await PruneArchiveStage().run(ctx_for(world))

    assert result.status is StageStatus.OK
    assert await archived_records(ctx_for(world)) == []
