"""The scoring stage, on live graph tenants."""

import pytest

from ontogr_mem.consolidation.models import Score, StageStatus
from ontogr_mem.consolidation.signals import record_access, record_feedback
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.consolidation.stages.feedback_weights import FeedbackWeightsStage
from ontogr_mem.consolidation.stages.scoring import ScoringStage
from tests.support.consolidation import ConsWorld, new_world
from tests.support.graph_backend import GraphBackend
from tests.unit.temporal.helpers import at, obs


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


async def score_of(world: ConsWorld, target_id: str) -> Score:
    """Return the stored score of ``target_id``, failing the test if it is absent."""
    score = await world.store.get(Score, target_id)
    assert score is not None
    return score


async def test_gate_skips_without_a_resolver_or_temporal_engine(
    graph_backend: GraphBackend,
) -> None:
    """Scoring is skipped when neither the resolver nor the temporal engine is wired."""
    world = await new_world(
        graph_backend,
    )
    ctx = StageContext(
        run_id="r1",
        clock=world.clock,
        store=world.store,
        settings=world.settings,
    )

    assert await ScoringStage().gate(ctx) == "no resolver or temporal engine"


@pytest.mark.parametrize(
    ("idle_days", "recency", "score"),
    [
        # Recency is 1.0 now and halves after each 30 days. Frequency is 0, and
        # salience is 0.25 from neutral feedback and no graph edge.
        (0, 1.0, 0.475),
        (30, 0.5, 0.275),
        (60, 0.25, 0.175),
    ],
)
async def test_recency_halves_every_thirty_days(
    graph_backend: GraphBackend, idle_days: int, recency: float, score: float
) -> None:
    """An entity's recency halves after each 30 days without a change."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    world.clock.advance(days=idle_days)

    await ScoringStage().run(ctx_for(world))

    stored = await score_of(world, "a")
    assert stored.kind == "entity"
    assert stored.recency == pytest.approx(recency)
    assert stored.score == pytest.approx(score)


@pytest.mark.parametrize(
    ("accesses", "frequency"),
    [(0, 0.0), (20, 1.0), (25, 1.0)],
)
async def test_frequency_is_zero_without_access_and_caps_at_twenty(
    graph_backend: GraphBackend, accesses: int, frequency: float
) -> None:
    """Frequency reaches 1.0 at 20 accesses and stays there."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    for _ in range(accesses):
        await record_access(world.store, "a", world.clock())

    await ScoringStage().run(ctx_for(world))

    stored = await score_of(world, "a")
    assert stored.frequency == pytest.approx(frequency)


@pytest.mark.parametrize(
    ("degree", "salience", "score"),
    [
        # Salience is half the capped degree plus half the neutral feedback of 0.5.
        (5, 0.5, 0.55),
        (10, 0.75, 0.625),
        (12, 0.75, 0.625),
    ],
)
async def test_salience_caps_at_degree_ten(
    graph_backend: GraphBackend, degree: int, salience: float, score: float
) -> None:
    """The degree part of salience caps at ten neighbours, and facts are scored."""
    world = await new_world(
        graph_backend,
    )
    neighbours = [f"n{i}" for i in range(degree)]
    await world.add_entity("hub", "Hub")
    for name in neighbours:
        await world.add_entity(name, name.upper())
    await world.ensure_entities(["hub", *neighbours])
    await world.temporal.ingest_observations(
        [
            obs(f"o{i}", name, at(2010), subject="hub", slot=f"s{i}")
            for i, name in enumerate(neighbours)
        ]
    )

    await ScoringStage().run(ctx_for(world))

    stored = await score_of(world, "hub")
    assert stored.salience == pytest.approx(salience)
    assert stored.score == pytest.approx(score)
    scores = await world.store.list(Score)
    assert sum(1 for s in scores if s.kind == "fact") == degree
    assert sum(1 for s in scores if s.kind == "entity") == degree + 1


async def test_feedback_raises_salience_and_score(graph_backend: GraphBackend) -> None:
    """One good rating lifts the weight from 0.5 to 0.55, and so the score."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    await record_feedback(world.store, "a", 1.0, world.clock())
    await FeedbackWeightsStage().run(ctx_for(world))

    await ScoringStage().run(ctx_for(world))

    stored = await score_of(world, "a")
    assert stored.salience == pytest.approx(0.275)
    assert stored.score == pytest.approx(0.4825)


async def test_scores_of_vanished_entities_are_removed(
    graph_backend: GraphBackend,
) -> None:
    """An archived entity leaves the graph, so its score is removed."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    await world.add_entity("b", "B")
    await ScoringStage().run(ctx_for(world))
    await world.resolution.archive("b")

    result = await ScoringStage().run(ctx_for(world))

    assert result.status is StageStatus.OK
    assert result.counts["removed"] == 1
    assert await world.store.get(Score, "b") is None
    assert await world.store.get(Score, "a") is not None


async def test_a_second_run_changes_no_score(graph_backend: GraphBackend) -> None:
    """Running the scoring stage again writes the same scores."""
    world = await new_world(
        graph_backend,
    )
    await world.add_entity("a", "A")
    await world.add_entity("b", "B")
    await ScoringStage().run(ctx_for(world))
    first = await world.store.list(Score)

    result = await ScoringStage().run(ctx_for(world))

    assert result.counts["removed"] == 0
    assert await world.store.list(Score) == first
