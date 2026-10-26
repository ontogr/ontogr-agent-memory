"""Bounded graph reads against the in-memory tenant stores."""

from dataclasses import replace

import pytest

from ontogr_mem.data_models import Alias
from ontogr_mem.retrieval.errors import RetrievalError
from ontogr_mem.retrieval.neighbourhood import neighbourhood
from ontogr_mem.temporal.reader import FactReader
from tests.support.episodes import make_episode
from tests.support.graph_backend import GraphBackend
from tests.support.p9i import NOW, add_fact, make_world
from tests.unit.temporal.helpers import at, obs


class TestNeighbourhood:
    """Walk the visible fact graph with stable bounds and annotations."""

    async def test_depth_limits_the_walk_and_edges_keep_their_statements(
        self, graph_backend: GraphBackend
    ):
        """A depth-one walk returns the seed and its direct neighbour."""
        episodes, resolver, store, _, engine, keywords = await make_world(
            graph_backend, ("A", "B", "C", "D")
        )
        for index, (left, right) in enumerate((("A", "B"), ("B", "C"), ("C", "D"))):
            await add_fact(
                episodes,
                engine,
                observation_id=f"o{index}",
                episode_id=f"ep{index}",
                subject=left,
                object_id=right,
                slot=f"edge-{index}",
                valid_from=at(2020),
                statement=f"{left} knows {right}.",
            )
        search = keywords

        result = await neighbourhood(
            store,
            resolver,
            search,
            FactReader(store),
            now=NOW,
            seeds=("A",),
            depth=1,
        )

        assert [node.entity_id for node in result.nodes] == ["A", "B"]
        assert [
            (edge.subject_id, edge.object_id, edge.statement) for edge in result.edges
        ] == [("A", "B", "A knows B.")]
        assert result.seeds == ("A",)
        assert result.truncated is False

    async def test_keyword_search_supplies_a_seed_for_an_alias(
        self, graph_backend: GraphBackend
    ):
        """A keyword hit on an alias starts the walk at its active entity."""
        episodes, resolver, store, _, engine, keywords = await make_world(
            graph_backend, ("ana", "acme")
        )
        await resolver.write_entity_batch(
            [],
            [
                Alias(
                    alias_id="alias-acme",
                    entity_id="acme",
                    text="Acme Labs",
                    strict_key="acme labs",
                    loose_key="acme labs",
                )
            ],
        )
        await add_fact(
            episodes,
            engine,
            observation_id="o1",
            episode_id="ep1",
            subject="ana",
            object_id="acme",
            slot="works-at",
            valid_from=at(2020),
            statement="Ana works at Acme Labs.",
        )

        result = await neighbourhood(
            store,
            resolver,
            keywords,
            FactReader(store),
            now=NOW,
            query="Acme",
        )

        assert result.seeds == ("acme",)
        assert {node.entity_id for node in result.nodes} == {"acme", "ana"}

    async def test_cap_returns_breadth_first_prefix_and_reports_truncation(
        self, graph_backend: GraphBackend
    ):
        """A full cap is truncated only when an unseen neighbour is waiting."""
        ids = ("hub", *(f"n{index:03}" for index in range(300)))
        episodes, resolver, store, _, engine, keywords = await make_world(
            graph_backend, ids
        )
        observation_rows = []
        episode_rows = []
        for index in range(300):
            episode_id = f"ep{index:03}"
            episode_rows.append(make_episode(episode_id))
            observation_rows.append(
                replace(
                    obs(
                        f"o{index:03}",
                        f"n{index:03}",
                        at(2020),
                        subject="hub",
                        slot=f"edge-{index:03}",
                        episode=episode_id,
                    ),
                    statement=f"hub knows n{index:03}.",
                    quote=f"hub knows n{index:03}.",
                )
            )
        await episodes.write_episodes(episode_rows)
        await engine.ingest_observations(observation_rows)
        search = keywords

        capped = await neighbourhood(
            store,
            resolver,
            search,
            FactReader(store),
            now=NOW,
            seeds=("hub",),
            depth=1,
            max_nodes=100,
        )
        complete = await neighbourhood(
            store,
            resolver,
            search,
            FactReader(store),
            now=NOW,
            seeds=("hub",),
            depth=1,
            max_nodes=301,
        )

        assert [node.entity_id for node in capped.nodes] == [
            "hub",
            *(f"n{index:03}" for index in range(99)),
        ]
        assert capped.truncated is True
        assert len(complete.nodes) == 301
        assert complete.truncated is False

    async def test_default_seeds_are_the_three_most_connected_entities(
        self, graph_backend: GraphBackend
    ):
        """Seed selection ranks visible fact counts and breaks ties by id."""
        ids = (
            "a-hub",
            "b-leaf",
            "c-leaf",
            "d-leaf",
            "e-hub",
            "f-leaf",
            "g-leaf",
        )
        episodes, resolver, store, _, engine, keywords = await make_world(
            graph_backend, ids
        )
        for index, (left, right) in enumerate(
            (
                ("a-hub", "b-leaf"),
                ("a-hub", "c-leaf"),
                ("a-hub", "d-leaf"),
                ("e-hub", "f-leaf"),
                ("e-hub", "g-leaf"),
            )
        ):
            await add_fact(
                episodes,
                engine,
                observation_id=f"o{index}",
                episode_id=f"ep{index}",
                subject=left,
                object_id=right,
                slot=f"edge-{index}",
                valid_from=at(2020),
                statement=f"{left} knows {right}.",
            )

        result = await neighbourhood(
            store,
            resolver,
            keywords,
            FactReader(store),
            now=NOW,
            query="",
        )

        assert result.seeds == ("a-hub", "e-hub", "b-leaf")

    async def test_default_view_hides_withdrawn_facts(
        self, graph_backend: GraphBackend
    ):
        """History exposes a retracted edge that the default view omits."""
        episodes, resolver, store, clock, engine, keywords = await make_world(
            graph_backend, ("ana", "acme")
        )
        await add_fact(
            episodes,
            engine,
            observation_id="o1",
            episode_id="ep1",
            subject="ana",
            object_id="acme",
            slot="employer",
            valid_from=at(2020),
            statement="Ana works at Acme.",
        )
        clock.value = clock.value.replace(day=23)
        await engine.retract_episodes(["ep1"])
        reader = FactReader(store)
        search = keywords

        current = await neighbourhood(
            store, resolver, search, reader, now=clock.value, seeds=("ana",)
        )
        history = await neighbourhood(
            store,
            resolver,
            search,
            reader,
            now=clock.value,
            seeds=("ana",),
            include_history=True,
        )

        assert current.edges == ()
        assert [edge.status.value for edge in history.edges] == ["active", "withdrawn"]

    @pytest.mark.parametrize(
        ("depth", "max_nodes"),
        ((5, 100), (-1, 100), (2, 0), (2, 501)),
    )
    async def test_out_of_range_bounds_raise_retrieval_error(
        self, graph_backend: GraphBackend, depth: int, max_nodes: int
    ):
        """Invalid walk bounds fail before the read begins."""
        episodes, resolver, store, _, _, keywords = await make_world(
            graph_backend, ("ana",)
        )

        with pytest.raises(RetrievalError):
            await neighbourhood(
                store,
                resolver,
                keywords,
                FactReader(store),
                now=NOW,
                depth=depth,
                max_nodes=max_nodes,
            )
