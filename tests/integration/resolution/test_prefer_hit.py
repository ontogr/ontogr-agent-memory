"""Behavior of an ambiguous name hit when several stored entities match."""

from ontogr_mem.data_models.entities import Alias, AliasSource, DecisionMethod
from ontogr_mem.llm import RoleModels
from ontogr_mem.resolution.policy import ResolutionMode, ResolutionPolicy, TypePolicy
from ontogr_mem.resolution.resolver import EntityResolver
from ontogr_mem.resolution.resolver_store import ResolverStore
from tests.integration.resolution.resolver_helpers import (
    count_active,
    ex,
    latest_mention,
    make,
)
from tests.support.graph_backend import GraphBackend
from tests.support.type_world import WORLD
from tests.unit.resolution.helpers import oracle_models


async def _namesakes(
    graph_backend: GraphBackend,
    *,
    policy: ResolutionPolicy | None = None,
    models: RoleModels | None = None,
    inline: bool = False,
) -> tuple[ResolverStore, EntityResolver, str, str]:
    """Store two people with one name and different emails."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(store, policy=policy, models=models, inline=inline)
    first = await resolver.resolve(
        [ex("1", "Ana", "Person", episode_id="ep1", email="a@x.org")], WORLD.links
    )
    second = await resolver.resolve(
        [ex("1", "Ana", "Person", episode_id="ep2", email="b@x.org")], WORLD.links
    )
    assert first.entities[0].entity_id != second.entities[0].entity_id
    return store, resolver, first.entities[0].entity_id, second.entities[0].entity_id


class TestPreferredHit:
    """A name shared by several compatible entities picks one without a verifier."""

    async def test_prefers_the_namesake_with_more_mentions(self, graph_backend):
        """An ambiguous mention attaches to the namesake seen more often."""
        store, resolver, first_id, second_id = await _namesakes(graph_backend)
        await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep3", email="b@x.org")],
            WORLD.links,
        )
        await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep4", email="b@x.org")],
            WORLD.links,
        )

        result = await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep5")], WORLD.links
        )

        assert result.entities[0].entity_id == second_id
        assert result.decisions[0].reason == "preferred_hit"
        assert await count_active(store) == 2

    async def test_equal_mention_counts_pick_the_oldest(self, graph_backend):
        """A tie on mention count goes to the older entity."""
        store, resolver, first_id, _second_id = await _namesakes(graph_backend)

        result = await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep3")], WORLD.links
        )

        assert result.entities[0].entity_id == first_id
        assert result.decisions[0].reason == "preferred_hit"
        assert await count_active(store) == 2

    async def test_a_key_hit_is_preferred_to_an_alias_hit(self, graph_backend):
        """A canonical name beats an alias on an otherwise tied hit."""
        store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(store)
        first = await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep1", email="a@x.org")],
            WORLD.links,
        )
        second = await resolver.resolve(
            [ex("1", "Anya", "Person", episode_id="ep2", email="b@x.org")],
            WORLD.links,
        )
        await store.write_entity_batch(
            [],
            [
                Alias(
                    alias_id="alias-anya-ana",
                    entity_id=second.entities[0].entity_id,
                    text="Ana",
                    strict_key="ana",
                    loose_key="ana",
                    source=AliasSource.EXTRACTED,
                )
            ],
        )

        result = await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep3")], WORLD.links
        )

        assert result.entities[0].entity_id == first.entities[0].entity_id
        assert result.decisions[0].method is DecisionMethod.KEY
        assert result.decisions[0].reason == "preferred_hit"

    async def test_a_configured_verifier_replaces_the_preferred_rule(
        self, graph_backend
    ):
        """With inline verification on, the ambiguous mention reaches the model."""
        models, scripted = oracle_models(lambda _a, _b: True)
        store, resolver, _first_id, _second_id = await _namesakes(
            graph_backend, models=models, inline=True
        )

        await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep3")], WORLD.links
        )

        assert scripted.calls
        decision = await latest_mention(store, "ep3")
        assert decision.decision_reason != "preferred_hit"

    async def test_verify_always_replaces_the_preferred_rule(self, graph_backend):
        """A VERIFY_ALWAYS override also sends the ambiguous mention to a model."""
        policy = ResolutionPolicy(
            types={"Person": TypePolicy(mode=ResolutionMode.VERIFY_ALWAYS)}
        )
        models, scripted = oracle_models(lambda _a, _b: True)
        store, resolver, _first_id, _second_id = await _namesakes(
            graph_backend, policy=policy, models=models, inline=True
        )

        await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep3")], WORLD.links
        )

        assert scripted.calls
        decision = await latest_mention(store, "ep3")
        assert decision.decision_reason != "preferred_hit"

    async def test_a_single_matching_email_picks_that_namesake(self, graph_backend):
        """An email that matches one namesake resolves the ambiguity."""
        store, resolver, first_id, _second_id = await _namesakes(graph_backend)

        result = await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep3", email="a@x.org")],
            WORLD.links,
        )

        assert result.entities[0].entity_id == first_id
        assert result.decisions[0].reason in {"key_hit", "identity_attr"}
        assert await count_active(store) == 2
