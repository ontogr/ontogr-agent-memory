"""Public feedback, consolidation, and tenant-read service behavior."""

import dataclasses
from datetime import UTC, datetime

import pytest

from ontogr_mem.relational_store.audit import AuditAction
from ontogr_mem.retrieval.errors import KeywordTimeoutError
from ontogr_mem.retrieval.models import MENTION_CAP
from ontogr_mem.service.contracts import (
    AddMemory,
    ConsolidateMemory,
    GetEntity,
    MemoryFeedback,
    MemoryKind,
    SearchMemory,
)
from ontogr_mem.service.errors import InvalidArgumentError, ObjectNotFoundError
from ontogr_mem.service.principal import Grant
from tests.support import memory_service as memory_service_support
from tests.support.memory_service import memory_service_harness


class TestMemoryServiceExtended:
    """Exercise the additional public service verbs on isolated tenants."""

    async def test_feedback_records_entity_and_fact_scores(self):
        """Feedback scores map from -1..1 to persisted 0..1 ratings."""
        harness = await memory_service_harness()
        receipt = await harness.service.add_memory(
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
        job = await harness.service.get_job(harness.principal, receipt.job_id)
        assert job.result is not None
        fact_id = job.result["fact_ids"][0]
        entity = await harness.service.get_entity(
            harness.principal, GetEntity(tenant="t1", name="Ana")
        )
        writer = harness.principal.model_copy(
            update={"grants": {"t1": frozenset({Grant.READ, Grant.WRITE})}}
        )

        entity_result = await harness.service.memory_feedback(
            writer,
            MemoryFeedback(tenant="t1", entity_id=entity.entity.entity_id, score=1.0),
        )
        fact_result = await harness.service.memory_feedback(
            writer,
            MemoryFeedback(tenant="t1", fact_id=fact_id, score=-1.0),
        )

        assert entity_result == {"recorded": 1, "rating": 1.0, "tenant_id": "t1"}
        assert fact_result == {"recorded": 2, "rating": 0.0, "tenant_id": "t1"}

    async def test_consolidate_memory_runs_as_a_write_job(self):
        """Consolidation returns its stage report through the job result."""
        harness = await memory_service_harness()

        receipt = await harness.service.consolidate_memory(
            harness.principal, ConsolidateMemory(tenant="t1", wait_seconds=5)
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)

        assert job.status.value == "done"
        assert job.result is not None
        assert isinstance(job.result["stages"], list)
        assert job.result["status"]
        reflection = next(
            stage for stage in job.result["stages"] if stage["stage"] == "reflection"
        )
        assert reflection["status"] == "skipped"

    async def test_history_entity_graph_stats_and_operations_are_readable(self):
        """Tenant reads return domain data, while stats and lists add no rows."""
        harness = await memory_service_harness()
        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="lives_in",
                object="Berlin",
                valid_from=datetime(2024, 1, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)
        assert job.result is not None
        fact_id = job.result["fact_ids"][0]
        entity_view = await harness.service.get_entity(
            harness.principal, GetEntity(tenant="t1", name="Ana")
        )
        principal = harness.principal.model_copy(
            update={"grants": {"t1": frozenset({Grant.READ, Grant.WRITE, Grant.ADMIN})}}
        )
        before = await harness.operations.list_operations(tenant="t1", limit=100)

        history = await harness.service.history(principal, "t1", fact_id)
        card = await harness.service.entity(principal, "t1", name="Ana")
        graph = await harness.service.graph(
            principal, "t1", seeds=(entity_view.entity.entity_id,), depth=1
        )
        stats = await harness.service.stats(principal, "t1")
        operations = await harness.service.operations(principal, "t1", limit=1)
        audit_rows = await harness.service.audit_log(principal, "t1", limit=1)
        after = await harness.operations.list_operations(tenant="t1", limit=100)

        assert history.fact_id == fact_id and len(history.versions) == 1
        assert card.name == "Ana" and card.facts[0].fact_id == fact_id
        assert {node.entity_id for node in graph.nodes} == {
            entity_view.entity.entity_id,
            entity_view.facts[0].object.entity_id,
        }
        assert stats["fact_versions"] == 1
        usage = stats["usage"]
        if not isinstance(usage, dict) or not isinstance(usage.get("operations"), int):
            raise AssertionError("stats usage must be a mapping")
        assert usage["operations"] >= 1
        assert len(operations) == 1
        assert audit_rows == []
        assert len(after) == len(before) + 3

    async def test_export_is_audited_and_requires_export_grant(self):
        """Export writes a scalar-detail audit event and checks its grant."""
        harness = await memory_service_harness()
        writer = harness.principal.model_copy(
            update={
                "grants": {"t1": frozenset({Grant.READ, Grant.WRITE, Grant.EXPORT})}
            }
        )
        admin = harness.principal.model_copy(
            update={"grants": {"t1": frozenset({Grant.ADMIN})}}
        )

        with pytest.raises(ObjectNotFoundError):
            await harness.service.export(harness.principal, "t1")
        exported = await harness.service.export(writer, "t1")
        audit_rows = await harness.service.audit_log(admin, "t1", limit=10)

        assert exported.jsonl.endswith("\n")
        assert exported.counts["meta"] == 1
        assert len(audit_rows) == 1
        row = audit_rows[0]
        assert row.entry.action is AuditAction.EXPORT
        assert row.entry.detail == {
            "count_meta": 1,
            "count_entity": 0,
            "count_fact": 0,
            "count_episode": 0,
        }
        assert all(
            value is None or isinstance(value, (str, int, bool))
            for value in row.entry.detail.values()
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param({"graph_depth": 5}, id="graph_depth_above_cap"),
            pytest.param(
                {"mentions": [f"Name {i}" for i in range(MENTION_CAP + 1)]},
                id="mentions_above_cap",
            ),
        ],
    )
    async def test_search_rejects_retrieval_caps_as_invalid_argument(self, overrides):
        """Search reports a request above a retrieval cap as a client error."""
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

        with pytest.raises(InvalidArgumentError):
            await harness.service.search_memory(
                harness.principal,
                SearchMemory(tenant="t1", query="Where does Ana work?", **overrides),
            )

    async def test_graph_keyword_timeout_passes_through_as_a_server_fault(
        self, monkeypatch
    ):
        """A keyword timeout during seed search is not reported as bad input."""

        class TimeoutKeywords:
            async def search(self, kind, query, *, limit):
                raise KeywordTimeoutError("keyword index timed out")

        original_memory_stores = memory_service_support.memory_stores

        async def stores_with_timeout_keywords(tenant_id, embedder, *, tracer=None):
            stores = await original_memory_stores(tenant_id, embedder, tracer=tracer)
            return dataclasses.replace(stores, keywords=TimeoutKeywords())

        monkeypatch.setattr(
            memory_service_support, "memory_stores", stores_with_timeout_keywords
        )
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

        with pytest.raises(KeywordTimeoutError):
            await harness.service.graph(
                harness.principal, "t1", query="Where does Ana work?"
            )

    async def test_graph_rejects_depth_above_the_cap_as_invalid_argument(self):
        """A graph depth above the retrieval cap is reported as a client error."""
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

        with pytest.raises(InvalidArgumentError):
            await harness.service.graph(
                harness.principal, "t1", query="Where does Ana work?", depth=99
            )
