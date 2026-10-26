"""Lineage reads on a live graph tenant: evidence, episode effects, belief chains."""

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.temporal.lineage import (
    Effect,
    Role,
    belief_chain,
    episode_effects,
    why,
)
from tests.integration.temporal.test_engine import make_engine, typed
from tests.support.clock import FixedClock
from tests.support.episodes import make_episode
from tests.support.graph_backend import GraphBackend
from tests.support.temporal import open_temporal
from tests.unit.temporal.helpers import at


NOW = FixedClock().now()


class TestTemporalLineage:
    """Who supports a fact, what an episode changed, and how belief moved."""

    async def test_why_lists_creator_then_confirmer_with_their_episodes(
        self, graph_backend: GraphBackend
    ):
        """The first episode is the creator and the repeat is a confirmer."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        engine = make_engine(store)
        await episodes.write_episodes(
            [make_episode("ep1"), make_episode("ep2", minutes=5)]
        )
        await engine.ingest([typed()], make_episode("ep1"))
        await engine.ingest([typed(episode_id="ep2")], make_episode("ep2", minutes=5))
        (fact,) = await store.current_facts()

        result = await why(store, fact.fact_id)

        assert result is not None
        assert [(row.role, row.episode_id) for row in result.evidence] == [
            (Role.CREATOR, "ep1"),
            (Role.CONFIRMER, "ep2"),
        ]

    async def test_why_of_an_unknown_fact_is_none(self, graph_backend: GraphBackend):
        """A fact that is not live has no evidence to show."""
        _, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )

        assert await why(store, "missing") is None

    async def test_episode_effects_lists_what_the_episode_confirmed(
        self, graph_backend: GraphBackend
    ):
        """A repeat episode is reported as confirming the fact it restated."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        engine = make_engine(store)
        await episodes.write_episodes(
            [make_episode("ep1"), make_episode("ep2", minutes=5)]
        )
        await engine.ingest([typed()], make_episode("ep1"))
        await engine.ingest([typed(episode_id="ep2")], make_episode("ep2", minutes=5))
        (fact,) = await store.current_facts()

        effects = await episode_effects(store, "ep2")

        assert effects == (
            Effect(fact.fact_id, fact.slot_key, Role.CONFIRMER, FactStatus.ACTIVE),
        )

    async def test_belief_chain_orders_versions_and_links_the_closing_audit_row(
        self, graph_backend: GraphBackend
    ):
        """A replacement shows the old value closed by the new one."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
                "rome",
            ),
        )
        engine = make_engine(store)
        await episodes.write_episodes(
            [make_episode("ep1"), make_episode("ep2", minutes=5)]
        )
        await engine.ingest([typed()], make_episode("ep1"))
        await engine.ingest(
            [
                typed(
                    fact_id="f2",
                    text="Ana lives in Rome",
                    entity_refs=("ana", "rome"),
                    valid_from=at(2024),
                    episode_id="ep2",
                )
            ],
            make_episode("ep2"),
        )

        chain = await belief_chain(store, subject_id="ana", relation_type_id=1, now=NOW)

        assert [(link.view.object_id, link.view.version) for link in chain] == [
            ("paris", 1),
            ("paris", 2),
            ("rome", 1),
        ]
        assert chain[0].closed_by is None
        assert chain[1].closed_by is not None
        assert chain[2].closed_by is None
