"""Embedding retries, batching, circuit protection, and strict result handling."""

import asyncio
from collections.abc import Sequence

import pytest

from ontogr_mem.relational_store import (
    MemoryOperationStore,
    OperationKind,
    OperationOutcome,
)
from ontogr_mem.request_context import RequestContext, bind_context
from ontogr_mem.vectorstore import (
    EmbeddingError,
    EmbeddingFailed,
    EmbeddingGateway,
    EmbeddingTimeoutError,
    FailReason,
    PermanentEmbeddingError,
    VectorStoreSettings,
)
from tests.support.clock import FixedClock
from tests.support.embedder import (
    FailingEmbedder,
    FlakyEmbedder,
    MockEmbedder,
    RecordingEmbedder,
    no_sleep,
)


def make_gateway(embedder, **settings: object) -> EmbeddingGateway:
    """Build a gateway with no delay between retry attempts."""
    return EmbeddingGateway(
        embedder,
        VectorStoreSettings(**settings),
        sleep=no_sleep,
    )


class TestEmbeddingGateway:
    """The embedding gateway keeps per-text outcomes while guarding calls."""

    async def test_transient_failures_split_and_eventually_embed_each_text(
        self,
    ) -> None:
        """A flaky batched provider recovers through retries and splitting."""
        embedder = FlakyEmbedder(failures=2)
        gateway = make_gateway(embedder)

        outcomes = await gateway.embed(["a", "b"])

        assert len(outcomes) == 2
        assert all(outcome.vector is not None for outcome in outcomes)
        assert all(outcome.reason is None for outcome in outcomes)
        assert embedder.calls >= 3

    async def test_exhausted_single_text_has_a_transient_failure_detail(self) -> None:
        """A single text returns an explicit failure after three tries."""
        embedder = FlakyEmbedder(failures=10)
        gateway = make_gateway(embedder, batch_size=1)

        outcomes = await gateway.embed(["a"])

        assert outcomes[0].reason is FailReason.EMBED_FAILED
        assert "rate limited" in outcomes[0].detail
        assert embedder.calls == 3

    async def test_default_batch_size_caps_provider_requests_at_thirty_six(
        self,
    ) -> None:
        """The default batch size remains 36 for arbitrarily large input lists."""
        embedder = RecordingEmbedder()
        gateway = make_gateway(embedder)

        outcomes = await gateway.embed([f"text-{i}" for i in range(130)])

        assert len(outcomes) == 130
        assert all(outcome.vector is not None for outcome in outcomes)
        assert max(map(len, embedder.batches)) == 36

    async def test_breaker_opens_after_two_exhausted_single_text_failures(self) -> None:
        """The breaker refuses later requests without calling the provider."""
        embedder = FlakyEmbedder(failures=100)
        gateway = make_gateway(
            embedder,
            batch_size=1,
            breaker_threshold=2,
        )

        first = await gateway.embed(["a"])
        second = await gateway.embed(["b"])
        calls_before_refusal = embedder.calls
        third = await gateway.embed(["c"])

        assert first[0].reason is FailReason.EMBED_FAILED
        assert second[0].reason is FailReason.EMBED_FAILED
        assert third[0].reason is FailReason.EMBED_FAILED
        assert third[0].detail == "circuit open"
        assert embedder.calls == calls_before_refusal

    async def test_splitting_large_batch_does_not_trip_the_breaker(self) -> None:
        """A provider rejecting only large batches is not counted as failing."""
        embedder = FailingEmbedder(transient_over=1)
        gateway = make_gateway(embedder, breaker_threshold=2)

        outcomes = await gateway.embed([f"text-{i}" for i in range(36)])
        after_split = await gateway.embed(["another"])

        assert all(outcome.vector is not None for outcome in outcomes)
        assert after_split[0].vector is not None

    async def test_permanent_failures_do_not_trip_the_breaker(self) -> None:
        """A provider that answered with permanent errors can still recover."""

        class PermanentThenGood(MockEmbedder):
            """Reject two calls permanently, then return normal vectors."""

            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
                self.calls += 1
                if self.calls <= 2:
                    raise PermanentEmbeddingError("bad credentials")
                return await super().embed_batch(texts)

        embedder = PermanentThenGood()
        gateway = make_gateway(embedder, breaker_threshold=2)

        first = await gateway.embed(["a"])
        second = await gateway.embed(["b"])
        third = await gateway.embed(["c"])

        assert first[0].reason is FailReason.EMBED_FAILED
        assert second[0].reason is FailReason.EMBED_FAILED
        assert third[0].vector is not None
        assert embedder.calls == 3

    async def test_zero_request_budget_returns_timeout_without_provider_call(
        self,
    ) -> None:
        """An expired request context maps to an outcome and stops the call."""
        embedder = RecordingEmbedder()
        gateway = make_gateway(embedder)
        context = RequestContext("owner", "tenant-a", latency_budget_s=0.0)

        with bind_context(context):
            outcomes = await gateway.embed(["a"])

        assert outcomes[0].reason is FailReason.EMBED_TIMEOUT
        assert embedder.batches == []

    async def test_context_deadline_cancels_a_slow_provider_call(self) -> None:
        """The request budget bounds an in-progress embedding call."""

        class SlowEmbedder(MockEmbedder):
            """Wait longer than the request deadline before embedding."""

            async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
                await asyncio.sleep(1.0)
                return await super().embed_batch(texts)

        gateway = make_gateway(SlowEmbedder(), attempt_timeout_s=5.0)
        context = RequestContext("owner", "tenant-a", latency_budget_s=0.05)

        with bind_context(context):
            outcomes = await gateway.embed(["a"])

        assert outcomes[0].reason is FailReason.EMBED_TIMEOUT
        assert "deadline" in outcomes[0].detail

    async def test_probe_respects_an_expired_request_deadline(self) -> None:
        """An expired request cannot probe the provider."""
        embedder = RecordingEmbedder()
        gateway = make_gateway(embedder)

        with (
            bind_context(RequestContext("owner", "tenant-a", latency_budget_s=0.0)),
            pytest.raises(EmbeddingTimeoutError, match="deadline"),
        ):
            await gateway.probe()

        assert embedder.batches == []

    async def test_probe_respects_an_open_circuit_breaker(self) -> None:
        """An open breaker refuses probes without another provider call."""
        embedder = FlakyEmbedder(failures=100)
        gateway = make_gateway(
            embedder,
            batch_size=1,
            breaker_threshold=1,
        )

        outcome = await gateway.embed(["a"])
        calls_before_probe = embedder.calls

        assert outcome[0].reason is FailReason.EMBED_FAILED
        with pytest.raises(EmbeddingError, match="circuit open"):
            await gateway.probe()

        assert embedder.calls == calls_before_probe

    async def test_embed_all_raises_at_first_failed_text(self) -> None:
        """The strict helper reports the first outcome without a vector."""
        gateway = make_gateway(MockEmbedder())

        with pytest.raises(EmbeddingFailed) as error:
            await gateway.embed_all(["a", "", "b"])

        assert error.value.index == 1
        assert "empty text" in str(error.value)

    async def test_embed_all_returns_vectors_for_every_text(self) -> None:
        """The strict helper returns unit-scaled vectors in input order."""
        gateway = make_gateway(MockEmbedder())

        vectors = await gateway.embed_all(["a", "b"])

        direct = await MockEmbedder().embed_batch(["a", "b"])
        assert vectors == [pytest.approx(tuple(vector)) for vector in direct]

    async def test_embeddings_do_not_make_operation_cost_unknown(self) -> None:
        """Embedding calls do not add unpriced model usage to the context."""
        store = MemoryOperationStore(FixedClock())
        operation_id = await store.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="tenant-a",
        )
        context = RequestContext("owner", "tenant-a", operation_id=operation_id)
        gateway = make_gateway(MockEmbedder())

        with bind_context(context):
            await gateway.embed(["a"])
        await store.finish(
            operation_id,
            outcome=OperationOutcome.OK,
            usage=context.usage.snapshot(),
        )

        record = await store.get(operation_id)
        assert record is not None
        assert record.usage.cost_usd == 0.0
