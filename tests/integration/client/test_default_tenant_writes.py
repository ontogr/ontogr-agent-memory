"""End-to-end checks of text writes on a fresh tenant with the default ontology."""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models.jobs import JobState
from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from tests.support.falkordb import falkordb_test_settings
from tests.support.tenants import new_tenant
from tests.support.text_writes import (
    ScriptedEntity,
    ScriptedFact,
    open_memory,
    text_models,
)


FIRST = "Ana works at Acme."
SECOND = "Ana works at Beta."
UNKNOWN_RELATION = "Ana mentors Ben."
MODEL_LABEL = "Ana works at Zeta."
EARLIER = datetime(2025, 1, 1, tzinfo=UTC)
LATER = datetime(2025, 6, 1, tzinfo=UTC)
SCRIPT = {
    FIRST: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Acme", "Organization")],
        [ScriptedFact("Ana works at Acme", 0, 1, "WORKS_AT", "Ana works at Acme")],
    ),
    SECOND: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Beta", "Organization")],
        [ScriptedFact("Ana works at Beta", 0, 1, "WORKS_AT", "Ana works at Beta")],
    ),
    UNKNOWN_RELATION: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Ben", "Person")],
        [ScriptedFact("Ana mentors Ben", 0, 1, "mentors", "Ana mentors Ben")],
    ),
    MODEL_LABEL: (
        [
            ScriptedEntity("Ana", "Person"),
            ScriptedEntity("Zeta", None, proposed_label="Startup"),
        ],
        [ScriptedFact("Ana works at Zeta", 0, 1, "WORKS_AT", "Ana works at Zeta")],
    ),
}


@pytest.fixture(autouse=True)
async def falkordb_environment(
    monkeypatch: pytest.MonkeyPatch, falkordb_store: FalkorDBGraphStore
) -> AsyncGenerator[None]:
    """Point the service settings at the compose FalkorDB for each test."""
    settings = falkordb_test_settings()
    monkeypatch.setenv("MEMORY_FALKORDB_HOST", settings.host)
    monkeypatch.setenv("MEMORY_FALKORDB_PORT", str(settings.port))
    yield


class TestDefaultTenantWrites:
    """Text writes on a fresh tenant behave as the default ontology defines."""

    async def test_default_relation_is_stored_and_recalled(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A fact on a default relation is stored and found by search."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                receipt = await memory.write_text(
                    FIRST, reference_time=EARLIER, wait_seconds=10
                )
                await memory.drain()
                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert len(job.result["fact_ids"]) == 1
                context = await memory.search("Where does Ana work?")
                assert [fact.statement for fact in context.facts] == [
                    "Ana works at Acme"
                ]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_single_valued_relation_keeps_only_the_latest_object(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A newer single-valued fact replaces the older object."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_text(FIRST, reference_time=EARLIER, wait_seconds=10)
                await memory.write_text(SECOND, reference_time=LATER, wait_seconds=10)
                await memory.drain()
                view = await memory.get_entity("Ana")
                assert [fact.statement for fact in view.facts] == ["Ana works at Beta"]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_unknown_relation_is_refused_and_counted(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """An unknown relation is refused and counted in the job result."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                receipt = await memory.write_text(
                    UNKNOWN_RELATION, reference_time=EARLIER, wait_seconds=10
                )
                await memory.drain()
                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert job.result["ontology_refused"] == 1
                assert job.result["fact_ids"] == []
                context = await memory.search("Who does Ana mentor?")
                assert context.facts == []
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_catalog_typed_entity_reads_back_with_an_empty_type_label(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A catalog typed entity reads back with an empty type label."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_text(FIRST, reference_time=EARLIER, wait_seconds=10)
                await memory.drain()
                view = await memory.get_entity("Acme")
                assert view.entity.type_label == ""
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_model_proposed_entity_type_is_stored_unchecked(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A model proposed entity type is stored unchanged."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, text_models(SCRIPT)) as memory:
                await memory.write_text(
                    MODEL_LABEL, reference_time=EARLIER, wait_seconds=10
                )
                await memory.drain()
                view = await memory.get_entity("Zeta")
                assert view.entity.type_label == "Startup"
        finally:
            await falkordb_store.erase_tenant(tenant)
