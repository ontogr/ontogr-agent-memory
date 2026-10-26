"""Public memory service behavior over isolated in-memory adapters."""

from datetime import UTC, datetime

import pytest

from ontogr_mem.service.contracts import (
    AddMemory,
    GetEntity,
    InvalidateMemory,
    ListEpisodes,
    MemoryKind,
    SearchMemory,
    UpdateMemory,
)
from ontogr_mem.service.errors import (
    LLMUnavailableError,
    ObjectNotFoundError,
)
from ontogr_mem.service.principal import Grant
from tests.support.memory_service import memory_service_harness


class TestMemoryService:
    """Exercise verbs through the service boundary."""

    async def test_status_lists_grants_and_reads_hide_denied_tenants(self):
        """Granted Tenants are listed and denied reads use the not-found error."""
        harness = await memory_service_harness()

        status = await harness.service.get_status(harness.principal)

        assert "t1" in status.tenants
        operations = await harness.operations.list_operations(tenant=None)
        assert [(row.kind.value, row.outcome.value) for row in operations] == [
            ("retrieve", "ok")
        ]
        with pytest.raises(ObjectNotFoundError) as denied:
            await harness.service.list_episodes(
                harness.principal.model_copy(update={"default_tenant": "t2"}),
                ListEpisodes(),
            )
        assert str(denied.value) == "tenant or object not found"
        granted = harness.principal.model_copy(
            update={"grants": {"t2": harness.principal.grants["t1"]}}
        )
        empty = await harness.service.list_episodes(granted, ListEpisodes(tenant="t2"))
        assert empty.episodes == []

    async def test_status_includes_default_tenant_for_wildcard_grant(self):
        """A wildcard grant exposes the principal's default tenant in status."""
        harness = await memory_service_harness()
        principal = harness.principal.model_copy(
            update={"grants": {"*": frozenset({Grant.READ})}}
        )

        status = await harness.service.get_status(principal)

        assert status.tenants == [principal.default_tenant]

    async def test_structured_writes_are_readable_through_entity_and_episode_verbs(
        self,
    ):
        """A fact write stores its relation statement and its session metadata."""
        harness = await memory_service_harness()
        effective_at = datetime(2024, 3, 1, tzinfo=UTC)

        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                valid_from=effective_at,
                session_id="session-one",
                wait_seconds=5,
                idempotency_key="fact-one",
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)

        assert job.status.value == "done"
        assert job.result is not None and len(job.result["fact_ids"]) == 1
        entity = await harness.service.get_entity(
            harness.principal, GetEntity(tenant="t1", name="ANA")
        )
        assert entity.entity.name == "Ana"
        assert entity.facts[0].statement == "Ana works at Acme"
        episodes = await harness.service.list_episodes(
            harness.principal, ListEpisodes(tenant="t1", session_id="session-one")
        )
        assert [episode.session_id for episode in episodes.episodes] == ["session-one"]

    async def test_search_returns_fact_and_rendered_context(self):
        """A stored fact is available in the structured and prompt response."""
        harness = await memory_service_harness()
        await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                wait_seconds=5,
            ),
        )

        result = await harness.service.search_memory(
            harness.principal,
            SearchMemory(tenant="t1", query="Ana works at Acme", format="both"),
        )

        assert result.state == "found"
        assert result.facts and result.facts[0].statement == "Ana works at Acme"
        assert result.rendered

    async def test_text_write_without_a_model_fails_before_queueing(self):
        """A free-text request does not create a job without an LLM client."""
        harness = await memory_service_harness()

        with pytest.raises(LLMUnavailableError):
            await harness.service.add_memory(
                harness.principal,
                AddMemory(tenant="t1", kind=MemoryKind.TEXT, content="Ana works here."),
            )

        assert await harness.state.tenant_ids() == []

    async def test_update_closes_the_current_fact_and_adds_its_replacement(self):
        """An update closes one fact and makes its replacement current."""
        harness = await memory_service_harness()
        original = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                valid_from=datetime(2024, 3, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        original_job = await harness.service.get_job(harness.principal, original.job_id)
        assert original_job.result is not None
        fact_id = original_job.result["fact_ids"][0]

        receipt = await harness.service.update_memory(
            harness.principal,
            UpdateMemory(
                tenant="t1",
                fact_id=fact_id,
                object="Globex",
                effective_at=datetime(2025, 6, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )

        job = await harness.service.get_job(harness.principal, receipt.job_id)
        entity = await harness.service.get_entity(
            harness.principal, GetEntity(tenant="t1", name="Ana")
        )
        assert job.status.value == "done"
        if job.result is None:
            raise AssertionError("completed update has no result")
        assert job.result["closed_fact_id"] == fact_id
        assert entity.facts[0].object.name == "Globex"

    async def test_invalidate_closes_a_fact_and_keeps_its_history(self):
        """An invalidated fact remains readable before its applied end time."""
        harness = await memory_service_harness()
        forget_principal = harness.principal.model_copy(
            update={"grants": {"t1": harness.principal.grants["t1"] | {Grant.FORGET}}}
        )
        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Initech",
                valid_from=datetime(2024, 3, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)
        assert job.result is not None
        fact_id = job.result["fact_ids"][0]

        invalidated = await harness.service.invalidate(
            forget_principal,
            InvalidateMemory(
                tenant="t1",
                fact_id=fact_id,
                at=datetime(2025, 9, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        historical = await harness.service.search_memory(
            harness.principal,
            SearchMemory(
                tenant="t1",
                query="Ana works at Initech",
                valid_at=datetime(2025, 8, 1, tzinfo=UTC),
            ),
        )
        current = await harness.service.search_memory(
            harness.principal,
            SearchMemory(tenant="t1", query="Ana works at Initech"),
        )
        history = await harness.service.search_memory(
            harness.principal,
            SearchMemory(
                tenant="t1",
                query="Ana works at Initech",
                include_history=True,
            ),
        )

        assert invalidated.status.value == "done"
        assert historical.facts and historical.facts[0].fact_id == fact_id
        assert current.facts == []
        assert any(
            fact.valid_to == datetime(2025, 9, 1, tzinfo=UTC) for fact in history.facts
        )

    async def test_update_refuses_a_closed_fact(self):
        """An already closed fact cannot be replaced a second time."""
        harness = await memory_service_harness()
        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                valid_from=datetime(2024, 3, 1, tzinfo=UTC),
                valid_to=datetime(2025, 1, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)
        assert job.result is not None
        fact_id = job.result["fact_ids"][0]

        receipt = await harness.service.update_memory(
            harness.principal,
            UpdateMemory(
                tenant="t1",
                fact_id=fact_id,
                object="Globex",
                effective_at=datetime(2025, 6, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        assert receipt.status.value == "failed"
        assert receipt.error is not None
        assert receipt.error.code.value == "conflict"
        assert receipt.error.message == "fact is closed, not active"

    async def test_episode_pages_do_not_repeat_episodes(self):
        """A next cursor starts after the last episode on the preceding page."""
        harness = await memory_service_harness()
        for object_name in ("Acme", "Globex", "Initech"):
            await harness.service.add_memory(
                harness.principal,
                AddMemory(
                    tenant="t1",
                    kind=MemoryKind.FACT,
                    subject="Ana",
                    relation="works_at",
                    object=object_name,
                    wait_seconds=5,
                ),
            )

        first = await harness.service.list_episodes(
            harness.principal, ListEpisodes(tenant="t1", limit=2)
        )
        second = await harness.service.list_episodes(
            harness.principal,
            ListEpisodes(tenant="t1", limit=2, cursor=first.next_cursor),
        )

        first_ids = {episode.episode_id for episode in first.episodes}
        second_ids = {episode.episode_id for episode in second.episodes}
        assert len(first.episodes) == 2
        assert len(second.episodes) == 1
        assert first_ids.isdisjoint(second_ids)
        assert second.next_cursor is None

    async def test_update_rejects_an_effective_time_before_fact_start(self):
        """The update contract rejects an end that precedes the fact start."""
        harness = await memory_service_harness()
        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                valid_from=datetime(2024, 3, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)
        assert job.result is not None
        fact_id = job.result["fact_ids"][0]

        receipt = await harness.service.update_memory(
            harness.principal,
            UpdateMemory(
                tenant="t1",
                fact_id=fact_id,
                object="Globex",
                effective_at=datetime(2024, 2, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )

        job = await harness.service.get_job(harness.principal, receipt.job_id)
        assert job.status.value == "failed"
        assert job.error is not None
        assert job.error.code.value == "invalid_argument"
        episodes = await harness.service.list_episodes(
            harness.principal, ListEpisodes(tenant="t1", limit=10)
        )
        assert len(episodes.episodes) == 1
