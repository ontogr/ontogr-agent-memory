"""Entity cards and tenant statistics on the in-memory twins."""

from datetime import timedelta

import pytest

from ontogr_mem.data_models import EntityStatus
from ontogr_mem.errors import AmbiguousNameError
from ontogr_mem.summaries import entity_card, tenant_stats
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.temporal.reader import FactReader
from tests.support.episodes import make_episode
from tests.support.p9i import NOW, add_fact, alias, entity, make_world
from tests.unit.temporal.helpers import at


class TestEntityCard:
    """Resolve entities and return their aliases and relevant facts."""

    async def test_by_id_returns_card_fields_and_sorted_aliases(self):
        """The card includes canonical fields and aliases in stable order."""
        episodes, resolver, store, _, engine = await make_world(("ana", "acme"))
        await resolver.write_entity_batch(
            [entity("ana", name="Ana")],
            [
                alias("ana", "Annie", "z-alias"),
                alias("ana", "Ana", "a-alias"),
            ],
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

        result = await entity_card(
            store,
            resolver,
            FactReader(store),
            now=NOW,
            entity_id="ana",
        )

        assert result is not None
        assert (result.entity_id, result.name, result.type_label) == (
            "ana",
            "Ana",
            "Person",
        )
        assert result.status is EntityStatus.ACTIVE
        assert result.aliases == ("Ana", "Annie")
        assert [fact.statement for fact in result.facts] == ["Ana works at Acme."]

    async def test_name_lookup_is_case_insensitive_and_ambiguous_names_list_ids(self):
        """Strict alias lookup handles case and reports all colliding entities."""
        _, resolver, store, _, _ = await make_world(("ana", "ana-2"))
        await resolver.write_entity_batch(
            [entity("ana", name="Ana"), entity("ana-2", name="Ana 2")],
            [alias("ana", "Ana", "a1"), alias("ana-2", "ANA", "a2")],
        )

        with pytest.raises(AmbiguousNameError) as raised:
            await entity_card(store, resolver, FactReader(store), now=NOW, name="ana")

        assert raised.value.ids == ("ana", "ana-2")

    async def test_name_lookup_follows_a_merged_entity_to_its_survivor(self):
        """An alias on a merged entity resolves to the survivor's card."""
        _, resolver, store, _, _ = await make_world(("survivor", "absorbed"))
        await resolver.write_entity_batch(
            [
                entity("survivor", name="Ana"),
                entity("absorbed", name="Ana Example").model_copy(
                    update={
                        "status": EntityStatus.MERGED,
                        "merged_into": "survivor",
                    }
                ),
            ],
            [alias("absorbed", "Ana Example", "merged-name")],
        )

        result = await entity_card(
            store, resolver, FactReader(store), now=NOW, name="ANA EXAMPLE"
        )

        assert result is not None
        assert result.entity_id == "survivor"
        assert result.name == "Ana"

    async def test_fact_limit_reports_truncation_and_unknown_target_returns_none(self):
        """The card marks a cut list and leaves not-found mapping to its caller."""
        episodes, resolver, store, _, engine = await make_world(
            ("ana", "acme", "globex")
        )
        for index, object_id in enumerate(("acme", "globex")):
            await add_fact(
                episodes,
                engine,
                observation_id=f"o{index}",
                episode_id=f"ep{index}",
                subject="ana",
                object_id=object_id,
                slot=f"employer-{index}",
                valid_from=at(2020),
                statement=f"Ana works at {object_id}.",
            )

        result = await entity_card(
            store,
            resolver,
            FactReader(store),
            now=NOW,
            entity_id="ana",
            max_facts=1,
        )
        missing = await entity_card(
            store,
            resolver,
            FactReader(store),
            now=NOW,
            entity_id="missing",
        )

        assert result is not None
        assert len(result.facts) == 1
        assert result.facts_truncated is True
        assert missing is None

    @pytest.mark.parametrize(
        ("entity_id", "name"),
        ((None, None), ("ana", "Ana")),
    )
    async def test_requires_exactly_one_lookup_key(self, entity_id, name):
        """Missing and competing lookup keys are rejected as invalid input."""
        _, resolver, store, _, _ = await make_world(("ana",))

        with pytest.raises(ValueError):
            await entity_card(
                store,
                resolver,
                FactReader(store),
                now=NOW,
                entity_id=entity_id,
                name=name,
            )


class TestTenantStats:
    """Count persisted fact versions, entities and paged episodes."""

    async def test_counts_all_fact_versions_entity_statuses_and_episodes(self):
        """Stats use history rows, every entity state and all episode pages."""
        episodes, resolver, store, clock, engine = await make_world(
            ("ana", "acme", "globex")
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
        clock.value += timedelta(days=1)
        await add_fact(
            episodes,
            engine,
            observation_id="o2",
            episode_id="ep2",
            subject="ana",
            object_id="globex",
            slot="employer",
            valid_from=at(2025),
            statement="Ana works at Globex.",
        )
        archived = entity("globex").model_copy(
            update={"status": EntityStatus.ARCHIVED}
        )
        await resolver.save_entity(archived)
        await episodes.write_episodes(
            [make_episode(f"extra-{index:03}") for index in range(99)]
        )

        result = await tenant_stats(store, resolver, episodes, now=clock.value)
        facts = await store.query(TemporalQuery(now=clock.value, include_history=True))

        assert result.fact_versions == len(facts)
        assert result.fact_versions == 3
        assert result.facts_by_status == {"active": 2, "closed": 1}
        assert result.entities == 3
        assert result.entities_by_status == {"active": 2, "archived": 1}
        assert result.episodes == 101
