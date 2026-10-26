"""Keyword behavior shared by the twin and both graph backends."""

from dataclasses import replace

import pytest

from ontogr_mem.data_models.entities import EntityStatus
from ontogr_mem.retrieval.keyword import KeywordKind
from tests.contracts.resolver_store import alias, entity, merge_plan
from tests.contracts.temporal_store import observation
from tests.support.episodes import make_episode


class KeywordSearchContract:
    """A fresh keyword harness provides the stores of one tenant."""

    async def test_aliases_rank_entities_once_and_filter_archived(self, harness):
        """Multiple aliases contribute the best hit, not duplicate entities."""
        await harness.resolver.write_entity_batch(
            [
                entity("both", name="Berlin Mitte"),
                entity("one"),
                entity("archived", status=EntityStatus.ARCHIVED),
            ],
            [
                alias("both", "Berlin Mitte"),
                alias("both", "Berlin"),
                alias("one", "Berlin"),
                alias("archived", "Berlin Mitte"),
            ],
        )
        hits = await harness.keyword.search(
            KeywordKind.ENTITY, "Berlin Mitte", limit=10
        )
        assert [hit.id for hit in hits] == ["both", "one"]
        assert [hit.rank for hit in hits] == [0, 1]
        assert hits[0].score > hits[1].score
        assert await harness.keyword.search(KeywordKind.ENTITY, "Berl", limit=10) == []
        assert (
            len(await harness.keyword.search(KeywordKind.ENTITY, "Berlin", limit=1))
            == 1
        )

    async def test_observation_statement_and_quote_are_searchable(self, harness):
        """Either text field can find an observation after a write."""
        await harness.episodes.write_episodes([make_episode("ep-1")])
        await harness.temporal.append_observations(
            [
                replace(
                    observation("obs-1"),
                    statement="Berlin office",
                    quote="quiet courtyard",
                )
            ]
        )
        for query in ("office", "courtyard"):
            hits = await harness.keyword.search(
                KeywordKind.OBSERVATION, query, limit=10
            )
            assert [hit.id for hit in hits] == ["obs-1"]

    async def test_episode_text_json_and_deletion(self, harness):
        """Structured content is searched as JSON and deletes are visible."""
        for episode in (
            make_episode("text", content="Berlin Mitte"),
            make_episode("json", content={"city": "Berlin"}),
        ):
            await harness.episodes.write_episodes([episode])
            await harness.episodes.record_content_hash(episode.id, "hash")
        hits = await harness.keyword.search(KeywordKind.EPISODE, "Berlin", limit=10)
        assert {hit.id for hit in hits} == {"text", "json"}
        await harness.episodes.remove_episode("text")
        hits = await harness.keyword.search(KeywordKind.EPISODE, "Berlin", limit=10)
        assert [hit.id for hit in hits] == ["json"]

    @pytest.mark.parametrize(
        "query",
        ["", "() : *", "a AND (b", "x:y", '"unterminated', "-minus", "the and is"],
    )
    async def test_empty_and_operator_queries_do_not_raise(self, harness, query):
        """User syntax is tokenized rather than interpreted by a query parser."""
        assert await harness.keyword.search(KeywordKind.EPISODE, query, limit=10) == []

    async def test_internal_apostrophes_and_ascii_case(self, harness):
        """Names with apostrophes remain complete tokens on both backends."""
        await harness.resolver.write_entity_batch(
            [entity("person")], [alias("person", "O'Brien")]
        )
        hits = await harness.keyword.search(KeywordKind.ENTITY, "O'BRIEN", limit=10)
        assert [hit.id for hit in hits] == ["person"]

    async def test_merged_aliases_return_the_active_survivor(self, harness):
        """A merged entity never leaks its old id through alias search."""
        survivor, absorbed = entity("survivor"), entity("absorbed")
        name = alias("absorbed", "Rarecompanyname")
        await harness.resolver.write_entity_batch([survivor, absorbed], [name])
        await harness.resolver.apply_merge(
            merge_plan("merge-1", survivor, absorbed, aliases=(name,))
        )

        hits = await harness.keyword.search(
            KeywordKind.ENTITY, "Rarecompanyname", limit=10
        )

        assert [hit.id for hit in hits] == ["survivor"]

    async def test_deleted_entities_disappear_from_alias_search(self, harness):
        """Deleting an entity removes its searchable aliases."""
        await harness.resolver.write_entity_batch(
            [entity("deleted")], [alias("deleted", "Rarecompanyname")]
        )
        assert [
            hit.id
            for hit in await harness.keyword.search(
                KeywordKind.ENTITY, "Rarecompanyname", limit=10
            )
        ] == ["deleted"]

        await harness.resolver.delete_entity("deleted")

        assert (
            await harness.keyword.search(
                KeywordKind.ENTITY, "Rarecompanyname", limit=10
            )
            == []
        )

    async def test_reset_observations_disappear_without_deleting_other_episodes(
        self, harness
    ):
        """Observation removal is reflected immediately in fulltext reads."""
        await harness.episodes.write_episodes(
            [make_episode("ep-1"), make_episode("ep-2")]
        )
        await harness.temporal.append_observations(
            [
                replace(observation("o1"), statement="uniqueterm", episode_id="ep-1"),
                replace(observation("o2"), statement="uniqueterm", episode_id="ep-2"),
            ]
        )
        assert {
            hit.id
            for hit in await harness.keyword.search(
                KeywordKind.OBSERVATION, "uniqueterm", limit=10
            )
        } == {"o1", "o2"}

        await harness.temporal.remove_episode_observations("ep-1")

        assert [
            hit.id
            for hit in await harness.keyword.search(
                KeywordKind.OBSERVATION, "uniqueterm", limit=10
            )
        ] == ["o2"]

    async def test_tenants_do_not_share_results(self, harness, other_harness):
        """A shared index never returns another tenant's documents."""
        await other_harness.episodes.write_episodes(
            [make_episode("foreign", content="secretword")]
        )
        await other_harness.episodes.record_content_hash("foreign", "hash")
        assert (
            await harness.keyword.search(KeywordKind.EPISODE, "secretword", limit=10)
            == []
        )
