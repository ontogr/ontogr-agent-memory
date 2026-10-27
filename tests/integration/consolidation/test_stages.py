"""The three stages of P7a, and the signals they read, on live graph tenants."""

import pytest

from ontogr_mem.consolidation.models import (
    FeedbackEvent,
    FeedbackWeight,
    StageStatus,
)
from ontogr_mem.consolidation.signals import record_access, record_feedback
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.consolidation.stages.deduplication import DeduplicationStage
from ontogr_mem.consolidation.stages.feedback_weights import FeedbackWeightsStage
from ontogr_mem.consolidation.stages.ingest_cleanup import IngestCleanupStage
from tests.support.consolidation import ConsWorld, new_world
from tests.support.episodes import make_episode
from tests.support.graph_backend import GraphBackend


def ctx_for(
    world: ConsWorld, *, resolution: bool = True, temporal: bool = True
) -> StageContext:
    """Build a stage context over the world, optionally without a port."""
    return StageContext(
        run_id="r1",
        clock=world.clock,
        store=world.store,
        settings=world.settings,
        resolution=world.resolution if resolution else None,
        temporal=world.temporal if temporal else None,
    )


def test_cleanup_is_the_only_fatal_stage() -> None:
    """Only the ingest cleanup stage is fatal."""
    assert IngestCleanupStage.fatal
    assert not DeduplicationStage.fatal
    assert not FeedbackWeightsStage.fatal


async def test_cleanup_is_skipped_when_no_episode_is_pending(
    graph_backend: GraphBackend,
) -> None:
    """Cleanup is skipped when no episode is pending."""
    world = await new_world(
        graph_backend,
    )

    reason = await IngestCleanupStage().gate(ctx_for(world))

    assert reason == "no incomplete episodes"


async def test_cleanup_is_skipped_without_a_temporal_engine(
    graph_backend: GraphBackend,
) -> None:
    """Cleanup is skipped without a temporal engine."""
    world = await new_world(
        graph_backend,
    )

    reason = await IngestCleanupStage().gate(ctx_for(world, temporal=False))

    assert reason == "no temporal engine"


async def test_cleanup_leaves_an_unrecoverable_episode_and_does_not_fail(
    graph_backend: GraphBackend,
) -> None:
    """Cleanup leaves an unrecoverable episode pending without failing."""
    world = await new_world(
        graph_backend,
    )
    await world.episodes.write_episodes([make_episode("ep1")])
    await world.episodes.record_content_hash("ep1", "hash-ep1")
    await world.temporal_store.set_episode_pending("ep1")
    ctx = ctx_for(world)

    assert await IngestCleanupStage().gate(ctx) is None
    result = await IngestCleanupStage().run(ctx)

    assert result.status is StageStatus.OK
    assert result.counts["episodes_recovered"] == 0
    assert result.counts["episodes_not_recoverable"] == 1
    assert await IngestCleanupStage().gate(ctx) is None


async def test_dedup_gate_needs_a_changed_entity(graph_backend: GraphBackend) -> None:
    """Deduplication runs only when an entity changed."""
    world = await new_world(
        graph_backend,
    )

    assert await DeduplicationStage().gate(ctx_for(world, resolution=False)) == (
        "no resolution"
    )
    assert await DeduplicationStage().gate(ctx_for(world)) == (
        "no entity changed since its last check"
    )
    await world.add_entity("a", "Acme")
    assert await DeduplicationStage().gate(ctx_for(world)) is None


async def test_dedup_merges_entities_that_share_a_key_and_then_gates(
    graph_backend: GraphBackend,
) -> None:
    """Deduplication merges entities that share a key, then gates shut."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "Acme Corp")
    await world.add_entity("b", "Acme Corp")

    result = await DeduplicationStage().run(ctx_for(world))

    assert result.status is StageStatus.OK
    assert result.counts["pairs_merged"] == 1
    live = await world.resolution.store.active_entities()
    assert len(live) == 1
    assert await DeduplicationStage().gate(ctx_for(world)) == (
        "no entity changed since its last check"
    )


async def test_feedback_gate_needs_an_unapplied_event(
    graph_backend: GraphBackend,
) -> None:
    """Feedback weights run only when an unapplied event exists."""
    world = await new_world(
        graph_backend,
    )
    assert await FeedbackWeightsStage().gate(ctx_for(world)) == "no new feedback"

    await record_feedback(world.store, "x", 1.0, world.clock())

    assert await FeedbackWeightsStage().gate(ctx_for(world)) is None


async def test_feedback_moves_weights_by_alpha_and_marks_events(
    graph_backend: GraphBackend,
) -> None:
    """Feedback moves each weight by alpha and marks its events applied."""
    world = await new_world(
        graph_backend,
    )
    await record_feedback(world.store, "x", 1.0, world.clock())
    await record_feedback(world.store, "x", 1.0, world.clock())

    result = await FeedbackWeightsStage().run(ctx_for(world))

    assert result.counts == {"applied": 2, "already_applied": 0, "dropped_neutral": 0}
    weight = await world.store.get(FeedbackWeight, "x")
    assert weight is not None
    assert weight.weight == pytest.approx(0.595)
    events = await world.store.list(FeedbackEvent)
    assert all(e.applied for e in events)


async def test_implicit_feedback_counts_half(graph_backend: GraphBackend) -> None:
    """Implicit feedback moves the weight at half strength."""
    world = await new_world(
        graph_backend,
    )
    await record_feedback(world.store, "x", 1.0, world.clock(), implicit=True)

    await FeedbackWeightsStage().run(ctx_for(world))

    weight = await world.store.get(FeedbackWeight, "x")
    assert weight is not None
    assert weight.weight == pytest.approx(0.525)


async def test_a_second_run_changes_no_weight(graph_backend: GraphBackend) -> None:
    """Running the feedback stage again changes no weight."""
    world = await new_world(
        graph_backend,
    )
    await record_feedback(world.store, "x", 0.0, world.clock())
    await FeedbackWeightsStage().run(ctx_for(world))
    before = await world.store.get(FeedbackWeight, "x")

    assert await FeedbackWeightsStage().gate(ctx_for(world)) == "no new feedback"
    assert await world.store.get(FeedbackWeight, "x") == before


async def test_a_crash_between_weight_write_and_event_mark_is_repaired(
    graph_backend: GraphBackend,
) -> None:
    """An event applied twice after a crash changes no weight."""
    world = await new_world(
        graph_backend,
    )
    event = await record_feedback(world.store, "x", 1.0, world.clock())
    await FeedbackWeightsStage().run(ctx_for(world))
    weight = await world.store.get(FeedbackWeight, "x")
    # Put the event back as unapplied, as if the process died after the weight write.
    await world.store.put(event)

    result = await FeedbackWeightsStage().run(ctx_for(world))

    assert result.counts["already_applied"] == 1
    assert result.counts["applied"] == 0
    assert await world.store.get(FeedbackWeight, "x") == weight


async def test_idle_weights_decay_to_neutral_and_are_dropped(
    graph_backend: GraphBackend,
) -> None:
    """Weights that decay to neutral are dropped."""
    world = await new_world(
        graph_backend,
    )
    await record_feedback(world.store, "x", 1.0, world.clock())
    await FeedbackWeightsStage().run(ctx_for(world))
    world.clock.advance(days=400)
    await record_feedback(world.store, "y", 1.0, world.clock())

    result = await FeedbackWeightsStage().run(ctx_for(world))

    assert result.counts["dropped_neutral"] == 1
    assert await world.store.get(FeedbackWeight, "x") is None
    assert await world.store.get(FeedbackWeight, "y") is not None


async def test_an_event_past_its_attempts_is_not_tried_again(
    graph_backend: GraphBackend,
) -> None:
    """An event past its attempt limit is not tried again."""
    world = await new_world(
        graph_backend,
    )
    event = await record_feedback(world.store, "x", 1.0, world.clock())
    await world.store.put(event.model_copy(update={"attempts": 3}))

    assert await FeedbackWeightsStage().gate(ctx_for(world)) == "no new feedback"


async def test_signal_ids_follow_the_store_counter(graph_backend: GraphBackend) -> None:
    """Signal ids follow the store counter."""
    world = await new_world(
        graph_backend,
    )

    first = await record_feedback(world.store, "x", 1.0, world.clock())
    second = await record_access(world.store, "x", world.clock())

    assert first.event_id == "fb-00000001"
    assert second.event_id == "ac-00000002"
