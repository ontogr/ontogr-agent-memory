"""Contract for `EpisodeStore` implementations."""

from collections.abc import Awaitable, Callable
from datetime import timedelta

import pytest

from ontogr_mem.data_models import TemporalStatus
from ontogr_mem.data_models.episode import EpisodeMeta
from tests.support.episodes import T0, Harness, make_episode


class EpisodeStoreContract:
    """Checks every `EpisodeStore` must pass.

    Subclasses provide the `make_harness` fixture, which returns a factory that
    creates a store for a fresh tenant on each call.
    """

    @pytest.fixture
    def make_harness(self) -> Callable[[], Awaitable[Harness]]:
        """Return a factory for stores on fresh tenants."""
        raise NotImplementedError

    @pytest.fixture
    async def harness(self, make_harness) -> Harness:
        """A store on a fresh tenant."""
        return await make_harness()

    async def test_episode_meta_round_trips_every_field(self, harness):
        """Metadata preserves optional fields and the order of fact ids."""
        meta = EpisodeMeta(
            "e1",
            session_id="s1",
            thread="t1",
            subject_ref="subject-1",
            fact_ids=("f2", "fé1"),
        )
        await harness.store.write_episodes([make_episode("e1")])

        await harness.store.write_episode_meta(meta)

        assert await harness.store.get_episode_meta("e1") == meta

    async def test_episode_ids_by_subject_returns_sorted_matching_ids(self, harness):
        """Subject lookup returns sorted episode ids and ignores absent metadata."""
        await harness.store.write_episodes(
            [make_episode("e2"), make_episode("e1"), make_episode("e3")]
        )
        await harness.store.write_episode_meta(EpisodeMeta("e2", subject_ref="ana"))
        await harness.store.write_episode_meta(EpisodeMeta("e1", subject_ref="ana"))
        await harness.store.write_episode_meta(EpisodeMeta("e3", subject_ref="bob"))

        assert await harness.store.episode_ids_by_subject("ana") == ["e1", "e2"]
        assert await harness.store.episode_ids_by_subject("missing") == []

    async def test_a_second_meta_write_replaces_the_first(self, harness):
        """A replacement clears optional fields and replaces all fact ids."""
        await harness.store.write_episodes([make_episode("e1")])
        await harness.store.write_episode_meta(
            EpisodeMeta("e1", "s1", "t1", "subject-1", ("f1", "f2"))
        )
        replacement = EpisodeMeta("e1")

        await harness.store.write_episode_meta(replacement)

        assert await harness.store.get_episode_meta("e1") == replacement

    async def test_missing_episode_meta_reads_as_none(self, harness):
        """Unknown episodes and episodes without metadata return None."""
        assert await harness.store.get_episode_meta("e1") is None

        await harness.store.write_episodes([make_episode("e1")])

        assert await harness.store.get_episode_meta("e1") is None

    async def test_episode_meta_of_an_unknown_episode_is_refused(self, harness):
        """Metadata cannot exist without its episode."""
        with pytest.raises(ValueError, match="unknown episode"):
            await harness.store.write_episode_meta(EpisodeMeta("e1"))

        assert await harness.store.get_episode_meta("e1") is None

    async def test_removing_an_episode_removes_its_meta(self, harness):
        """Deleted metadata does not return when the episode is written again."""
        episode = make_episode("e1")
        await harness.store.write_episodes([episode])
        await harness.store.write_episode_meta(EpisodeMeta("e1", session_id="s1"))

        await harness.store.remove_episode("e1")

        assert await harness.store.get_episode_meta("e1") is None
        await harness.store.write_episodes([episode])
        assert await harness.store.get_episode_meta("e1") is None

    async def test_episode_meta_does_not_cross_tenants(self, make_harness):
        """Metadata reads, writes and deletion stay within one tenant."""
        first = await make_harness()
        second = await make_harness()
        for harness in (first, second):
            await harness.store.write_episodes([make_episode("e1")])
        first_meta = EpisodeMeta("e1", session_id="s1")
        second_meta = EpisodeMeta("e1", session_id="s2")

        await first.store.write_episode_meta(first_meta)
        await second.store.write_episode_meta(second_meta)

        assert await first.store.get_episode_meta("e1") == first_meta
        assert await second.store.get_episode_meta("e1") == second_meta
        await first.store.remove_episode("e1")
        assert await first.store.get_episode_meta("e1") is None
        assert await second.store.get_episode_meta("e1") == second_meta

    async def test_round_trips_every_field(self, harness):
        """A written episode reads back equal, with microsecond times."""
        episode = make_episode("e1", content={"b": [1, 2], "a": "é"})

        await harness.store.write_episodes([episode])
        await harness.store.record_content_hash("e1", "hash-e1")
        stored = await harness.store.get_episode("e1")

        assert stored == episode

    async def test_text_content_round_trips(self, harness):
        """Plain text content stays a string."""
        episode = make_episode("e1", content="line one\nline two")

        await harness.store.write_episodes([episode])
        await harness.store.record_content_hash("e1", "hash-e1")

        assert await harness.store.get_episode("e1") == episode

    async def test_missing_episode_reads_as_none(self, harness):
        """An unknown id returns None."""
        assert await harness.store.get_episode("nope") is None
        assert await harness.store.get_content_hash("nope") is None

    async def test_get_episodes_returns_the_requested_episodes_ordered_by_id(
        self, harness
    ):
        """The requested episodes come back ordered by id, with every field."""
        episodes = [
            make_episode("e1", minutes=1, content="first"),
            make_episode("e2", minutes=2, content={"b": [1, 2], "a": "é"}),
            make_episode("e3", minutes=3, actor="u2", metadata={"session_id": "s2"}),
        ]
        await harness.store.write_episodes(episodes)
        for episode in episodes:
            await harness.store.record_content_hash(episode.id, episode.content_hash)

        stored = await harness.store.get_episodes(["e3", "e1", "e2"])

        assert stored == episodes

    async def test_get_episodes_omits_unknown_ids(self, harness):
        """An id that no episode carries is left out of the result."""
        episode = make_episode("e1")
        await harness.store.write_episodes([episode])

        stored = await harness.store.get_episodes(["e1", "nope"])

        assert [e.id for e in stored] == ["e1"]

    async def test_get_episodes_with_no_ids_returns_an_empty_list(self, harness):
        """No ids means no episodes."""
        assert await harness.store.get_episodes([]) == []

    async def test_list_episodes_uses_newest_first_keyset_pages(self, harness):
        """Pages order by ingest time and id, without repeating the cursor row."""
        episodes = [
            make_episode("a", ingested_minutes=1),
            make_episode("b", ingested_minutes=2),
            make_episode("c", ingested_minutes=2),
            make_episode("d", ingested_minutes=3),
        ]
        await harness.store.write_episodes(episodes)

        first = await harness.store.list_episodes(limit=2)
        second = await harness.store.list_episodes(
            limit=2,
            before=(first[-1].ingested_at, first[-1].id),
        )

        assert [episode.id for episode in first] == ["d", "c"]
        assert [episode.id for episode in second] == ["b", "a"]

    async def test_list_episodes_filters_by_metadata_session(self, harness):
        """Session filtering omits missing metadata and unknown sessions."""
        await harness.store.write_episodes(
            [make_episode("one"), make_episode("two", minutes=1)]
        )
        await harness.store.write_episode_meta(EpisodeMeta("one", session_id="s1"))
        await harness.store.write_episode_meta(EpisodeMeta("two", session_id="s2"))

        assert [
            episode.id
            for episode in await harness.store.list_episodes(limit=10, session_id="s1")
        ] == ["one"]
        assert await harness.store.list_episodes(limit=10, session_id="unknown") == []

    async def test_list_episodes_without_session_includes_missing_metadata(
        self, harness
    ):
        """An unfiltered list includes episodes without an EpisodeMeta row."""
        await harness.store.write_episodes([make_episode("one"), make_episode("two")])
        await harness.store.write_episode_meta(EpisodeMeta("one", session_id="s1"))

        assert [
            episode.id for episode in await harness.store.list_episodes(limit=10)
        ] == ["two", "one"]

    async def test_list_episodes_stays_within_its_tenant(self, make_harness):
        """The same tenant-scoped episode id does not expose another record."""
        first = await make_harness()
        second = await make_harness()
        await first.store.write_episodes([make_episode("same", content="first")])
        await second.store.write_episodes([make_episode("same", content="second")])

        assert [
            episode.content for episode in await first.store.list_episodes(limit=1)
        ] == ["first"]
        assert [
            episode.content for episode in await second.store.list_episodes(limit=1)
        ] == ["second"]

    async def test_get_episodes_returns_a_repeated_id_once(self, harness):
        """A repeated id comes back as one episode."""
        episode = make_episode("e1")
        await harness.store.write_episodes([episode])
        await harness.store.record_content_hash("e1", episode.content_hash)

        stored = await harness.store.get_episodes(["e1", "e1"])

        assert stored == [episode]

    async def test_get_episodes_returns_six_hundred_episodes(self, harness):
        """A read larger than one batch returns every episode."""
        episodes = [
            make_episode(f"e{i:04d}", minutes=i, ref=f"r{i}") for i in range(600)
        ]
        await harness.store.write_episodes(episodes)

        stored = await harness.store.get_episodes([episode.id for episode in episodes])

        assert [episode.id for episode in stored] == [
            episode.id for episode in episodes
        ]

    async def test_get_episodes_does_not_return_another_tenant_episode(
        self, make_harness
    ):
        """The same episode id in two tenants never reads across."""
        first = await make_harness()
        second = await make_harness()
        await first.store.write_episodes([make_episode("e1", content="first")])
        await second.store.write_episodes([make_episode("e1", content="second")])

        first_read = await first.store.get_episodes(["e1"])
        second_read = await second.store.get_episodes(["e1"])

        assert [e.content for e in first_read] == ["first"]
        assert [e.content for e in second_read] == ["second"]

    async def test_content_hash_is_set_only_after_it_is_recorded(self, harness):
        """The hash is None after the write and present after recording."""
        await harness.store.write_episodes([make_episode("e1")])

        assert await harness.store.get_content_hash("e1") is None

        await harness.store.record_content_hash("e1", "abc")

        assert await harness.store.get_content_hash("e1") == "abc"

    async def test_rewrite_keeps_first_ingested_at_and_hash(self, harness):
        """Writing an episode again does not move ingested_at or drop the hash."""
        first = make_episode("e1", ingested_minutes=0)
        again = make_episode("e1", ingested_minutes=30, content="changed")
        await harness.store.write_episodes([first])
        await harness.store.record_content_hash("e1", "abc")

        await harness.store.write_episodes([again])
        stored = await harness.store.get_episode("e1")

        assert stored.ingested_at == first.ingested_at
        assert stored.content == "changed"
        assert stored.content_hash == "abc"

    async def test_rewrite_keeps_temporal_state(self, harness):
        """Temporal fields set by another writer survive a second write."""
        await harness.store.write_episodes([make_episode("e1")])
        await harness.set_temporal_state("e1", TemporalStatus.COMMITTED, "h1")

        await harness.store.write_episodes([make_episode("e1")])
        stored = await harness.store.get_episode("e1")

        assert stored.temporal_status is TemporalStatus.COMMITTED
        assert stored.temporal_committed_hash == "h1"

    async def test_remove_deletes_orphaned_sources_and_actors(self, harness):
        """Removing the only episode of a source and actor deletes both."""
        await harness.store.write_episodes([make_episode("e1")])

        await harness.store.remove_episode("e1")

        assert await harness.store.get_episode("e1") is None
        assert await harness.node_ids("Source") == set()
        assert await harness.node_ids("Actor") == set()

    async def test_remove_keeps_sources_and_actors_in_use(self, harness):
        """A source or actor shared with another episode stays."""
        await harness.store.write_episodes(
            [
                make_episode("e1", ref="shared", actor="u1"),
                make_episode("e2", ref="shared", actor="u1", minutes=1),
                make_episode("e3", ref="own", actor="u2", minutes=2),
            ]
        )

        await harness.store.remove_episode("e3")
        await harness.store.remove_episode("e1")

        assert await harness.node_ids("Episode") == {"e2"}
        assert await harness.node_ids("Source") == {"chat|shared"}
        assert await harness.node_ids("Actor") == {"u1"}

    async def test_remove_of_missing_episode_is_a_no_op(self, harness):
        """Removing an unknown id does nothing."""
        await harness.store.remove_episode("nope")

    async def test_latest_before_filters_and_orders(self, harness):
        """Only earlier, fully ingested episodes come back, oldest first."""
        episodes = [make_episode(f"e{i}", minutes=i) for i in range(6)]
        await harness.store.write_episodes(episodes)
        for episode in episodes:
            if episode.id != "e2":
                await harness.store.record_content_hash(episode.id, "h")

        result = await harness.store.latest_before(
            T0 + timedelta(minutes=4, microseconds=7), 3, exclude_id="e3"
        )

        assert [e.id for e in result] == ["e0", "e1", "e4"]

    async def test_latest_before_orders_ties_by_ingest_time_then_id(self, harness):
        """Episodes sharing occurred_at come back in a fixed order."""
        episodes = [
            make_episode("b", minutes=1, ingested_minutes=2),
            make_episode("a", minutes=1, ingested_minutes=2),
            make_episode("c", minutes=1, ingested_minutes=1),
            make_episode("z", minutes=2),
        ]
        await harness.store.write_episodes(episodes)
        for episode in episodes:
            await harness.store.record_content_hash(episode.id, "h")

        result = await harness.store.latest_before(T0 + timedelta(minutes=5), 10)

        assert [e.id for e in result] == ["c", "a", "b", "z"]

    async def test_tenants_do_not_see_each_other(self, make_harness):
        """The same episode id in two tenants never reads across."""
        first = await make_harness()
        second = await make_harness()
        await first.store.write_episodes([make_episode("e1", content="first")])
        await first.store.record_content_hash("e1", "h")
        await second.store.write_episodes([make_episode("e1", content="second")])

        assert (await first.store.get_episode("e1")).content == "first"
        assert (await second.store.get_episode("e1")).content == "second"
        assert await second.store.get_content_hash("e1") is None
        assert await first.store.latest_before(T0 + timedelta(days=1), 10) != []
        assert await second.store.latest_before(T0 + timedelta(days=1), 10) == []
        await first.store.remove_episode("e1")
        assert (await second.store.get_episode("e1")).content == "second"

    async def test_writes_a_thousand_episodes_in_one_call(self, harness):
        """A large batch is stored completely."""
        episodes = [
            make_episode(f"e{i:04d}", minutes=i, ref=f"r{i}") for i in range(1000)
        ]

        await harness.store.write_episodes(episodes)

        assert len(await harness.node_ids("Episode")) == 1000
        assert (await harness.store.get_episode("e0999")).id == "e0999"
