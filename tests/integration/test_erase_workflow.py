"""Behavior checks for Erase cleanup, residue detection and recovery."""

import pytest

from ontogr_mem.data_models import FactStatus
from ontogr_mem.llm.cache import CacheEntry, MemoryResponseCache
from ontogr_mem.relational_store.jobs.erase_journal import ErasureStatus
from ontogr_mem.service.contracts import ForgetMode
from ontogr_mem.service.forget.workflow import EraseWorkflow
from ontogr_mem.service.runtime import TenantRuntime
from ontogr_mem.vectorstore.models import VectorKind
from tests.support.erase_workflow import EraseWorld, make_erase_world


class TestEraseWorkflow:
    """Exercise Erase through memory-backed stores and their public ports."""

    async def test_erase_reports_effects_and_prunes_surviving_entity(self):
        """Counts reflect storage changes and unsupported entity text is removed."""
        world = await make_erase_world()
        before_vector = await world.runtime.stores.vectors.fetch_vectors(
            VectorKind.NODE, ["person"]
        )
        await world.cache.put(
            "tagged-entry",
            CacheEntry(
                prompt="unrelated prompt",
                text="unrelated completion",
                input_tokens=1,
                output_tokens=1,
                tags=frozenset({"episode:e1"}),
            ),
        )
        await world.cache.put(
            "text-entry",
            CacheEntry(
                prompt=world.needle,
                text="completion without a tag",
                input_tokens=1,
                output_tokens=1,
            ),
        )
        workflow = EraseWorkflow(
            world.journal,
            lambda _: _runtime(world),
            world.runtime.clock,
            cache=world.cache,
        )

        result = await workflow.run(
            "caller",
            "erase-test",
            world.plan,
            ForgetMode.ERASE,
        )

        assert result.status == ErasureStatus.COMPLETED.value
        assert result.residue == ()
        assert result.text_scanned is True
        assert result.blanked_entity_ids == ("person",)
        assert result.removed.episodes == 1
        assert result.removed.facts == 1
        assert result.removed.observations == 1
        assert result.removed.entities == 1
        assert result.removed.vector_documents == 4
        assert result.removed.cache_entries == 2
        assert result.removed.entity_text_updates == 1

        stores = world.runtime.stores
        assert await stores.episodes.get_episode("e1") is None
        assert await stores.temporal_store.observations_of_episode("e1") == []
        facts = await stores.temporal_store.facts_by_ids(world.plan.fact_ids)
        assert len(facts) == 2
        assert facts[-1].status is FactStatus.WITHDRAWN
        assert await stores.resolver_store.has_entity("organization") is False
        person_rows = await stores.resolver_store.get_entities(["person"])
        assert len(person_rows) == 1
        person = person_rows[0]
        assert person.canonical_name == "Bob"
        assert person.summary == ""
        assert person.descriptions == []
        assert person.attributes == {}
        assert person.summary_stale is True
        aliases = await stores.resolver_store.aliases_of("person")
        assert {alias.text for alias in aliases} == {"Bob"}
        after_vector = await stores.vectors.fetch_vectors(VectorKind.NODE, ["person"])
        assert before_vector["person"] != after_vector["person"]
        assert await world.cache.count_tag("episode:e1") == 0
        assert await world.cache.scan(world.needle) == 0
        journal_row = await world.journal.get(result.erasure_id)
        assert journal_row is not None
        assert journal_row.status is ErasureStatus.COMPLETED

    async def test_text_residue_fails_erase_after_cleanup(self):
        """A cache entry that retains erased text is reported as residue."""

        class RetainingCache(MemoryResponseCache):
            async def evict_text(self, needle: str) -> int:
                """Leave matching entries in place and report no removals."""
                return 0

        cache = RetainingCache()
        world = await make_erase_world(cache=cache)
        await cache.put(
            "untagged-entry",
            CacheEntry(
                prompt=world.needle,
                text="retained completion",
                input_tokens=1,
                output_tokens=1,
            ),
        )
        workflow = EraseWorkflow(
            world.journal,
            lambda _: _runtime(world),
            world.runtime.clock,
            cache=cache,
        )

        result = await workflow.run(
            "caller",
            "erase-test",
            world.plan,
            ForgetMode.ERASE,
        )

        assert result.status == ErasureStatus.FAILED.value
        assert result.residue == ("model_cache",)
        assert result.removed.cache_entries == 0
        assert await world.runtime.stores.episodes.get_episode("e1") is None
        record = await world.journal.get(result.erasure_id)
        assert record is not None
        assert record.status is ErasureStatus.FAILED
        assert record.residue == ("model_cache",)

    async def test_recovery_finishes_cleanup_once_and_skips_closed_rows(self):
        """Recovery repeats completed steps safely and closes the journal row."""
        world = await make_erase_world()

        def crash_after_entities(step: str) -> None:
            if step == "entities":
                raise RuntimeError("simulated interruption")

        interrupted = EraseWorkflow(
            world.journal,
            lambda _: _runtime(world),
            world.runtime.clock,
            cache=world.cache,
            after_step=crash_after_entities,
        )
        with pytest.raises(RuntimeError, match="simulated interruption"):
            await interrupted.run(
                "caller",
                "erase-test",
                world.plan,
                ForgetMode.ERASE,
            )

        running = await world.journal.running("erase-test")
        assert len(running) == 1
        assert running[0].last_step == "entities"
        resumed = EraseWorkflow(
            world.journal,
            lambda _: _runtime(world),
            world.runtime.clock,
            cache=world.cache,
        )

        results = await resumed.recover("erase-test")

        assert len(results) == 1
        assert results[0].status == ErasureStatus.COMPLETED.value
        assert results[0].text_scanned is False
        assert await world.runtime.stores.episodes.get_episode("e1") is None
        assert (
            await world.runtime.stores.temporal_store.observations_of_episode("e1")
            == []
        )
        assert (
            await world.runtime.stores.resolver_store.has_entity("organization")
        ) is False
        person = (await world.runtime.stores.resolver_store.get_entities(["person"]))[0]
        assert person.canonical_name == "Bob"
        assert person.summary == ""
        assert await world.journal.running("erase-test") == []
        assert await resumed.recover("erase-test") == []


async def _runtime(world: EraseWorld) -> TenantRuntime:
    """Return the already-built runtime for the test tenant."""
    return world.runtime
