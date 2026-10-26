"""Tenant-scoped completion caching and operation usage through the live gate."""

import asyncio

import pytest
from pydantic import BaseModel

from ontogr_mem.llm import (
    DeadlineExceeded,
    LLMGate,
    LLMRequest,
    LLMResponse,
    ModelSize,
)
from ontogr_mem.llm.cache import MemoryResponseCache
from ontogr_mem.llm.pricing import Price, PriceTable
from ontogr_mem.relational_store import (
    MemoryOperationStore,
    OperationKind,
    OperationOutcome,
)
from ontogr_mem.request_context import RequestContext, bind_context
from tests.support.clock import FixedClock
from tests.support.llm import ScriptedLLM


class Person(BaseModel):
    """A structured completion used to verify cache invalidation."""

    name: str


def request(prompt: str = "find a person") -> LLMRequest:
    """Build a stable completion request."""
    return LLMRequest("system", prompt, ModelSize.SMALL)


def gate(client, cache: MemoryResponseCache, **kwargs) -> LLMGate:
    """Build a gate with a real cache and no retry delay."""

    async def no_sleep(seconds: float) -> None:
        return None

    return LLMGate(
        client,
        max_input_tokens=10_000,
        cache=cache,
        sleep=no_sleep,
        **kwargs,
    )


class TestLLMGateCache:
    """The gate partitions cache entries and accounts only model misses."""

    async def test_same_request_hits_the_same_tenant_cache(self) -> None:
        """A repeated request under one Tenant reuses the stored completion."""
        client = ScriptedLLM(["stored"])
        cache = MemoryResponseCache()
        model_gate = gate(client, cache)
        context = RequestContext("owner", "tenant-a")

        with bind_context(context):
            first = await model_gate.complete(request())
            second = await model_gate.complete(request())

        assert first == second
        assert second.text == "stored"
        assert len(client.requests) == 1

    async def test_different_tenants_do_not_share_cache_entries(self) -> None:
        """The same prompt for another Tenant reaches the model again."""
        client = ScriptedLLM(["first", "second"])
        model_gate = gate(client, MemoryResponseCache())

        with bind_context(RequestContext("owner", "tenant-a")):
            first = await model_gate.complete(request())
        with bind_context(RequestContext("owner", "tenant-b")):
            second = await model_gate.complete(request())

        assert first.text == "first"
        assert second.text == "second"
        assert len(client.requests) == 2

    async def test_context_without_tenant_bypasses_the_cache(self) -> None:
        """Unscoped requests do not share a cache bucket."""
        client = ScriptedLLM(["first", "second"])
        model_gate = gate(client, MemoryResponseCache())
        context = RequestContext("owner", None)

        with bind_context(context):
            first = await model_gate.complete(request())
            second = await model_gate.complete(request())

        assert first.text == "first"
        assert second.text == "second"
        assert len(client.requests) == 2

    async def test_tag_eviction_makes_the_completion_reach_the_model_again(
        self,
    ) -> None:
        """Evicting a context tag removes its associated completion."""
        client = ScriptedLLM(["first", "second"])
        cache = MemoryResponseCache()
        model_gate = gate(client, cache)
        context = RequestContext("owner", "tenant-a", cache_tags={"episode:e1"})

        with bind_context(context):
            await model_gate.complete(request())
        assert await cache.evict_tag("episode:e1") == 1
        with bind_context(context):
            response = await model_gate.complete(request())

        assert response.text == "second"
        assert len(client.requests) == 2

    async def test_cache_hit_adds_the_current_episode_tag(self) -> None:
        """Every episode that reuses a reply can later evict that reply."""
        client = ScriptedLLM(["stored"])
        cache = MemoryResponseCache()
        model_gate = gate(client, cache)

        with bind_context(
            RequestContext("owner", "tenant-a", cache_tags={"episode:first"})
        ):
            first = await model_gate.complete(request())
        with bind_context(
            RequestContext("owner", "tenant-a", cache_tags={"episode:second"})
        ):
            second = await model_gate.complete(request())

        assert second == first
        assert len(client.requests) == 1
        assert await cache.count_tag("episode:first") == 1
        assert await cache.count_tag("episode:second") == 1
        assert await cache.evict_tag("episode:second") == 1

    async def test_expired_context_cannot_return_a_cached_response(self) -> None:
        """A cache hit does not let an expired request exceed its budget."""
        client = ScriptedLLM(["stored"])
        model_gate = gate(client, MemoryResponseCache())

        with bind_context(RequestContext("owner", "tenant-a")):
            await model_gate.complete(request())

        with (
            bind_context(RequestContext("owner", "tenant-a", latency_budget_s=0.0)),
            pytest.raises(DeadlineExceeded),
        ):
            await model_gate.complete(request())

        assert len(client.requests) == 1

    async def test_slow_cache_lookup_is_bounded_by_the_request_deadline(self) -> None:
        """A cache read is cancelled when the request budget expires."""

        class SlowCache(MemoryResponseCache):
            """Wait during reads so the gateway must enforce the caller budget."""

            async def get(self, key: str):
                await asyncio.sleep(1.0)
                return await super().get(key)

        client = ScriptedLLM(["unused"])
        model_gate = gate(client, SlowCache())
        context = RequestContext("owner", "tenant-a", latency_budget_s=0.05)

        with bind_context(context), pytest.raises(DeadlineExceeded):
            await model_gate.complete(request())

        assert client.requests == []

    async def test_model_size_partitions_cache_entries(self) -> None:
        """Different model tiers do not reuse each other's completion."""
        client = ScriptedLLM(["small answer", "medium answer"])
        model_gate = gate(client, MemoryResponseCache())
        context = RequestContext("owner", "tenant-a")

        with bind_context(context):
            small = await model_gate.complete(request())
            medium = await model_gate.complete(
                LLMRequest("system", "find a person", ModelSize.MEDIUM)
            )

        assert small.text == "small answer"
        assert medium.text == "medium answer"
        assert len(client.requests) == 2

    async def test_unpriced_miss_usage_lands_on_the_operation_row(self) -> None:
        """A cache hit adds no usage and the unknown model cost stays unknown."""
        store = MemoryOperationStore(FixedClock())
        operation_id = await store.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="tenant-a",
        )
        context = RequestContext("owner", "tenant-a", operation_id=operation_id)
        client = ScriptedLLM([LLMResponse("answer", 7, 3)])
        model_gate = gate(client, MemoryResponseCache())

        with bind_context(context):
            await model_gate.complete(request())
            await model_gate.complete(request())
        await store.finish(
            operation_id,
            outcome=OperationOutcome.OK,
            usage=context.usage.snapshot(),
        )

        record = await store.get(operation_id)
        assert record is not None
        assert record.usage.tokens_in == 7
        assert record.usage.tokens_out == 3
        assert record.usage.cost_usd is None

    async def test_priced_miss_usage_is_not_increased_by_a_hit(self) -> None:
        """The operation total uses price data from the cache miss only."""
        store = MemoryOperationStore(FixedClock())
        operation_id = await store.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="tenant-a",
        )
        context = RequestContext("owner", "tenant-a", operation_id=operation_id)
        client = ScriptedLLM([LLMResponse("answer", 1000, 500)])
        model_gate = gate(
            client,
            MemoryResponseCache(),
            model_name="gpt-4o-mini",
            prices=PriceTable({"gpt-4o": Price(2.0, 6.0)}),
        )

        with bind_context(context):
            await model_gate.complete(request())
            await model_gate.complete(request())
        await store.finish(
            operation_id,
            outcome=OperationOutcome.OK,
            usage=context.usage.snapshot(),
        )

        record = await store.get(operation_id)
        assert record is not None
        assert record.usage.tokens_in == 1000
        assert record.usage.tokens_out == 500
        assert record.usage.cost_usd == 0.005

    async def test_structured_validation_evicts_bad_reply_before_next_call(
        self,
    ) -> None:
        """A later structured request cannot reuse an invalid cached answer."""
        client = ScriptedLLM(["not json", {"name": "Ana"}, {"name": "Bea"}])
        cache = MemoryResponseCache()
        model_gate = gate(client, cache)
        context = RequestContext("owner", "tenant-a")

        with bind_context(context):
            first = await model_gate.structured(
                Person,
                "system",
                "find a person",
                model_size=ModelSize.MEDIUM,
            )
            second = await model_gate.structured(
                Person,
                "system",
                "find a person",
                model_size=ModelSize.MEDIUM,
            )

        assert first == Person(name="Ana")
        assert second == Person(name="Bea")
        assert len(client.requests) == 3
