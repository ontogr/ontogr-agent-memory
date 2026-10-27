"""Trace spans, ids, and usage as callers observe service and gateway work."""

import asyncio
import logging
from contextlib import AbstractContextManager
from types import TracebackType

import pytest

from ontogr_mem.data_models.jobs import JobProgress
from ontogr_mem.llm import LLMGate, LLMRequest, ModelSize
from ontogr_mem.llm.cache import MemoryResponseCache
from ontogr_mem.observability import InMemoryTracer, Span, SpanName
from ontogr_mem.request_context import RequestContext, bind_context
from ontogr_mem.service.contracts import (
    AddMemory,
    MemoryContext,
    MemoryKind,
    SearchMemory,
)
from ontogr_mem.service.errors import ObjectNotFoundError
from ontogr_mem.service.principal import Grant
from tests.support.graph_backend import GraphBackend
from tests.support.llm import ScriptedLLM
from tests.support.memory_service import memory_service_harness


class TestOperationSpans:
    """Exercise tracing through real service verbs and in-memory adapters."""

    async def test_write_records_its_authorization_child_and_trace_id(
        self, graph_backend: GraphBackend, caplog
    ):
        """A write span owns its auth check and carries the accepted trace id."""
        tracer = InMemoryTracer()
        harness = await memory_service_harness(graph_backend, tracer=tracer)
        caplog.set_level(logging.INFO)

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
            trace_id="req-42",
        )
        await harness.service.drain()

        job = await harness.state.get_job(receipt.job_id)
        progress = JobProgress(stage="index", done=1, total=1)
        await harness.state.set_job_progress(receipt.job_id, progress)
        job_view = await harness.service.get_job(harness.principal, receipt.job_id)
        operations = await harness.operations.list_operations(tenant="t1")
        spans = tracer.finished()
        roots = [
            span
            for span in spans
            if span.name == SpanName.WRITE
            and span.attributes.get("trace_id") == "req-42"
        ]
        checks = [span for span in spans if span.name == SpanName.AUTH_CHECK]
        assert receipt.trace_id == "req-42"
        assert job is not None and job.trace_id == "req-42"
        assert job_view.trace_id == "req-42"
        assert job_view.progress == progress
        assert any(row.trace_id == "req-42" for row in operations)
        assert roots
        assert any(
            check.parent_id == root.span_id for check in checks for root in roots
        )
        assert any("trace=req-42" in record.getMessage() for record in caplog.records)
        assert all(span.error is None for span in spans)

    async def test_invalid_trace_id_is_replaced_before_storage_and_logging(
        self, graph_backend: GraphBackend, caplog
    ):
        """A trace id with whitespace cannot forge another log line."""
        tracer = InMemoryTracer()
        harness = await memory_service_harness(graph_backend, tracer=tracer)
        caplog.set_level(logging.INFO)

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
            trace_id="bad id\nforged",
        )

        job = await harness.state.get_job(receipt.job_id)
        assert job is not None
        assert receipt.trace_id == job.trace_id
        assert len(job.trace_id) == 32
        assert all(character in "0123456789abcdef" for character in job.trace_id)
        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert f"trace={job.trace_id}" in messages
        assert "forged" not in messages

    async def test_retrieve_span_does_not_contain_query_text(
        self, graph_backend: GraphBackend
    ):
        """The embedding span reports a count without retaining search text."""
        tracer = InMemoryTracer()
        harness = await memory_service_harness(graph_backend, tracer=tracer)
        marker = "PRIVATE-EPISODE-TEXT"
        await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                statement=marker,
                wait_seconds=5,
            ),
        )

        await harness.service.search_memory(
            harness.principal,
            SearchMemory(tenant="t1", query=marker),
        )

        spans = tracer.finished()
        embed_spans = [span for span in spans if span.name == SpanName.LLM_EMBED]
        assert embed_spans
        assert all(span.attributes.get("count") == 1 for span in embed_spans)
        assert all("text" not in span.attributes for span in embed_spans)
        assert marker not in repr([span.attributes for span in spans])
        assert marker not in repr([span.error for span in spans])

    async def test_denied_write_records_a_denied_auth_child(
        self, graph_backend: GraphBackend
    ):
        """A grant denial remains visible on the failed verb span."""
        tracer = InMemoryTracer()
        harness = await memory_service_harness(graph_backend, tracer=tracer)
        principal = harness.principal.model_copy(update={"grants": {}})

        with pytest.raises(ObjectNotFoundError):
            await harness.service.add_memory(
                principal,
                AddMemory(
                    tenant="t1",
                    kind=MemoryKind.FACT,
                    subject="Ana",
                    relation="works_at",
                    object="Acme",
                ),
            )

        spans = tracer.finished()
        denied = [
            span
            for span in spans
            if span.name == SpanName.AUTH_CHECK
            and span.attributes.get("outcome") == "denied"
        ]
        failed_writes = [
            span
            for span in spans
            if span.name == SpanName.WRITE and span.error is not None
        ]
        assert denied
        assert any(
            write.error is not None and write.error.startswith("ObjectNotFoundError:")
            for write in failed_writes
        )
        assert any(
            check.parent_id == write.span_id
            for check in denied
            for write in failed_writes
        )

    async def test_a_failing_tracer_does_not_change_a_write_result(
        self, graph_backend: GraphBackend
    ):
        """Tracer creation failures leave the operation running normally."""

        class BrokenTracer:
            def span(
                self, name: SpanName | str, **attributes: object
            ) -> AbstractContextManager[Span]:
                raise RuntimeError("tracer unavailable")

        harness = await memory_service_harness(graph_backend, tracer=BrokenTracer())

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

        assert receipt.error is None
        assert receipt.tenant == "t1"
        result = await harness.service.search_memory(
            harness.principal,
            SearchMemory(tenant="t1", query="Ana works at Acme"),
        )
        assert isinstance(result, MemoryContext)

    async def test_span_exit_failure_does_not_replace_the_verb_error(
        self, graph_backend: GraphBackend
    ):
        """Tracer shutdown failures do not hide a denied operation's error."""

        class BrokenExitTracer:
            def span(
                self, name: SpanName | str, **attributes: object
            ) -> AbstractContextManager[Span]:
                class SpanContext:
                    def __enter__(self) -> Span:
                        return self

                    def __exit__(
                        self,
                        exc_type: type[BaseException] | None,
                        exc: BaseException | None,
                        traceback: TracebackType | None,
                    ) -> bool:
                        raise RuntimeError("tracer exit unavailable")

                    def set(self, name: str, value: object) -> None:
                        pass

                    def error(self, error: BaseException) -> None:
                        pass

                return SpanContext()

        harness = await memory_service_harness(graph_backend, tracer=BrokenExitTracer())
        principal = harness.principal.model_copy(update={"grants": {}})

        with pytest.raises(ObjectNotFoundError):
            await harness.service.add_memory(
                principal,
                AddMemory(
                    tenant="t1",
                    kind=MemoryKind.FACT,
                    subject="Ana",
                    relation="works_at",
                    object="Acme",
                ),
            )

    async def test_concurrent_tenant_writes_keep_separate_span_trees(
        self, graph_backend: GraphBackend
    ):
        """Concurrent operations do not attach checks to another Tenant's root."""
        tracer = InMemoryTracer()
        harness = await memory_service_harness(graph_backend, tracer=tracer)
        principal = harness.principal.model_copy(
            update={
                "grants": {
                    "t1": frozenset({Grant.READ, Grant.WRITE}),
                    "t2": frozenset({Grant.READ, Grant.WRITE}),
                }
            }
        )

        async def write(tenant: str):
            return await harness.service.add_memory(
                principal,
                AddMemory(
                    tenant=tenant,
                    kind=MemoryKind.FACT,
                    subject="Ana",
                    relation="works_at",
                    object="Acme",
                    wait_seconds=5,
                ),
                trace_id=f"trace-{tenant}",
            )

        await asyncio.gather(write("t1"), write("t2"))
        await harness.service.drain()

        spans = tracer.finished()
        for tenant in ("t1", "t2"):
            # Each request opens a root write span and a child span for the
            # queued job; only the root has no parent.
            root = next(
                span
                for span in spans
                if span.name == SpanName.WRITE
                and span.parent_id is None
                and span.attributes.get("tenant") == tenant
                and span.attributes.get("trace_id") == f"trace-{tenant}"
            )
            check = next(
                span
                for span in spans
                if span.name == SpanName.AUTH_CHECK
                and span.attributes.get("tenant") == tenant
            )
            assert check.parent_id == root.span_id


class TestGatewaySpans:
    """The model-call span reports only safe model and usage metadata."""

    async def test_completion_cache_hit_reports_zero_input_tokens(self):
        """A completion span records cache outcome and never stores its prompt."""
        tracer = InMemoryTracer()
        gate = LLMGate(
            ScriptedLLM(["answer"]),
            max_input_tokens=100,
            model_name="test-model",
            cache=MemoryResponseCache(),
            tracer=tracer,
        )
        context = RequestContext(principal_id="caller", tenant="tenant_a")
        request = LLMRequest(
            system="system",
            prompt="PRIVATE-EPISODE-TEXT",
            model_size=ModelSize.SMALL,
        )

        with bind_context(context):
            await gate.complete(request)
            await gate.complete(request)

        spans = [
            span for span in tracer.finished() if span.name == SpanName.LLM_COMPLETE
        ]
        assert len(spans) == 2
        assert spans[0].attributes["model"] == "test-model"
        assert spans[0].attributes["size"] == "small"
        assert spans[1].attributes["cache.hit"] is True
        assert spans[1].attributes["tokens.in"] == 0
        assert all(
            "PRIVATE-EPISODE-TEXT" not in repr(span.attributes) for span in spans
        )
