"""Tracer tests cover no-op behavior, finished spans, parents and retention."""

import asyncio

from ontogr_mem.observability import InMemoryTracer, NoOpTracer, SpanName


class TestNoOpTracer:
    """The default tracer avoids allocating state for each span."""

    def test_reuses_one_span_and_supports_context_manager_use(self):
        """Every no-op span context yields the same inert span object."""
        tracer = NoOpTracer()

        with tracer.span("x") as first:
            first.set("count", 2)
        with tracer.span("y") as second:
            second.error(ValueError("ignored"))

        assert first is second


class TestInMemoryTracer:
    """The in-memory tracer returns a bounded snapshot of finished spans."""

    def test_records_attributes_duration_and_error_summary(self):
        """Finished spans expose redacted attributes, duration and error text."""
        tracer = InMemoryTracer()

        with tracer.span(SpanName.WRITE, tenant="acme") as span:
            span.set("count", 3)
            span.error(ValueError("failed\nprivate detail"))

        finished = tracer.finished()

        assert len(finished) == 1
        assert finished[0].name == SpanName.WRITE.value
        assert finished[0].attributes == {"tenant": "acme", "count": 3}
        assert finished[0].duration_ms >= 0
        assert finished[0].error == "ValueError: failed"

    def test_nested_spans_keep_their_parent_id(self):
        """A child span points to the active span that opened it."""
        tracer = InMemoryTracer()

        with tracer.span(SpanName.WRITE), tracer.span(SpanName.LLM_COMPLETE):
            pass

        spans = tracer.finished()

        assert spans[0].parent_id == spans[1].span_id
        assert spans[1].parent_id is None

    async def test_concurrent_tasks_keep_separate_parent_chains(self):
        """Each task's child belongs to that task's active parent span."""
        tracer = InMemoryTracer()

        async def operation(index: int) -> None:
            with tracer.span(SpanName.WRITE, count=index):
                await asyncio.sleep(0)
                with tracer.span(SpanName.LLM_COMPLETE):
                    await asyncio.sleep(0)

        await asyncio.gather(operation(1), operation(2))
        spans = tracer.finished()
        parent_ids = {
            span.attributes["count"]: span.span_id
            for span in spans
            if span.name == SpanName.WRITE.value
        }
        child_parent_ids = {
            span.parent_id for span in spans if span.name == SpanName.LLM_COMPLETE.value
        }

        assert len(set(parent_ids.values())) == 2
        assert child_parent_ids == set(parent_ids.values())

    def test_keeps_only_the_last_1000_finished_spans(self):
        """The oldest completed spans leave the bounded buffer first."""
        tracer = InMemoryTracer()

        for index in range(1005):
            with tracer.span(SpanName.WRITE, count=index):
                pass

        finished = tracer.finished()

        assert len(finished) == 1000
        assert finished[0].attributes["count"] == 5
        assert finished[-1].attributes["count"] == 1004

    def test_tracer_instances_do_not_share_parent_state(self):
        """A span in another tracer remains a root under an active parent."""
        first = InMemoryTracer()
        second = InMemoryTracer()

        with first.span(SpanName.WRITE), second.span(SpanName.LLM_COMPLETE):
            pass

        assert second.finished()[0].parent_id is None
