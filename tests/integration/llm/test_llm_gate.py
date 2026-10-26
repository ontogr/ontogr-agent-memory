"""End-to-end behavior of the chat gateway around scripted model clients."""

import asyncio

import pytest
from pydantic import BaseModel

from ontogr_mem.llm import (
    CircuitOpen,
    DeadlineExceeded,
    InputTooLargeError,
    LLMCallError,
    LLMGate,
    LLMRequest,
    LLMResponse,
    ModelSize,
    StructuredOutputError,
)
from ontogr_mem.llm.settings import LLMSettings
from ontogr_mem.resilience import CircuitBreaker
from tests.support.llm import ScriptedLLM


class Person(BaseModel):
    """A person returned by the structured-output tests."""

    name: str


class FakeTime:
    """A clock the breaker test advances without waiting."""

    def __init__(self) -> None:
        """Start at time zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current fake time."""
        return self.now


async def no_sleep(seconds: float) -> None:
    """Skip model retry delays in deterministic tests."""


def make_request(prompt: str = "hello", size: ModelSize = ModelSize.SMALL):
    """Build a small request with stable defaults."""
    return LLMRequest(system="system", prompt=prompt, model_size=size)


def make_gate(client, **settings: object) -> LLMGate:
    """Build a gate with no-op retry delays and validated settings."""
    return LLMGate(
        client,
        max_input_tokens=10_000,
        settings=LLMSettings(**settings),
        sleep=no_sleep,
    )


class TestLLMGate:
    """Retry, deadline, batch, structured output, and concurrency behavior."""

    async def test_retries_transient_failures_then_returns_the_reply(self) -> None:
        """Transient provider errors retry up to the configured attempt limit."""
        client = ScriptedLLM(
            [LLMCallError("connection reset"), LLMCallError("busy"), "ready"]
        )
        gate = make_gate(client, max_attempts=3)

        response = await gate.complete(make_request())

        assert response.text == "ready"
        assert len(client.requests) == 3

    async def test_exhaustion_chains_the_last_provider_error(self) -> None:
        """Exhausting retries raises a typed error chained to the last failure."""
        failures = [
            LLMCallError("first"),
            LLMCallError("second"),
            LLMCallError("third"),
        ]
        client = ScriptedLLM(failures)
        gate = make_gate(client, max_attempts=3)

        with pytest.raises(LLMCallError) as error:
            await gate.complete(make_request())

        assert error.value.__cause__ is failures[-1]
        assert len(client.requests) == 3

    async def test_terminal_provider_error_is_not_retried(self) -> None:
        """An authentication failure is returned on its first attempt."""
        client = ScriptedLLM([LLMCallError("HTTP 401 unauthorized"), "unused"])
        gate = make_gate(client, max_attempts=3)

        with pytest.raises(LLMCallError, match="401"):
            await gate.complete(make_request())

        assert len(client.requests) == 1

    async def test_untyped_client_error_propagates_without_retry(self) -> None:
        """A programming error is not treated as a provider failure."""
        client = ScriptedLLM([ValueError("bad request object"), "unused"])
        gate = make_gate(client, max_attempts=3)

        with pytest.raises(ValueError, match="bad request object"):
            await gate.complete(make_request())

        assert len(client.requests) == 1

    async def test_per_call_timeout_is_not_retried(self) -> None:
        """A provider call that outlasts its timeout ends after one attempt."""

        class SlowClient:
            """Waits longer than the configured timeout."""

            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, request: LLMRequest) -> LLMResponse:
                self.calls += 1
                await asyncio.sleep(1.0)
                return LLMResponse("late", 1, 1)

        client = SlowClient()
        gate = LLMGate(
            client,
            max_input_tokens=100,
            timeout_s=0.01,
            settings=LLMSettings(max_attempts=3),
            sleep=no_sleep,
        )

        with pytest.raises(TimeoutError):
            await gate.complete(make_request())

        assert client.calls == 1

    async def test_breaker_refuses_calls_until_its_recovery_period_passes(self) -> None:
        """A tripped breaker blocks calls, then admits one recovery probe."""
        clock = FakeTime()
        breaker = CircuitBreaker(2, 30, clock)
        client = ScriptedLLM(
            [LLMCallError("first"), LLMCallError("second"), "recovered"]
        )
        gate = LLMGate(
            client,
            max_input_tokens=100,
            settings=LLMSettings(max_attempts=2),
            breaker=breaker,
            sleep=no_sleep,
        )

        with pytest.raises(LLMCallError):
            await gate.complete(make_request())
        with pytest.raises(CircuitOpen):
            await gate.complete(make_request())
        assert len(client.requests) == 2

        clock.now = 31
        assert (await gate.complete(make_request())).text == "recovered"
        assert len(client.requests) == 3

    async def test_cancelling_a_half_open_probe_releases_its_slot(self) -> None:
        """A cancelled recovery call does not leave the breaker stuck."""

        class BlockingClient:
            """Block the first call and answer the next one."""

            def __init__(self) -> None:
                self.entered = asyncio.Event()
                self.calls = 0

            async def complete(self, request: LLMRequest) -> LLMResponse:
                self.calls += 1
                if self.calls == 1:
                    self.entered.set()
                    await asyncio.Event().wait()
                return LLMResponse("recovered", 1, 1)

        clock = FakeTime()
        breaker = CircuitBreaker(1, 10, clock)
        breaker.record_failure()
        clock.now = 11
        client = BlockingClient()
        gate = LLMGate(
            client,
            max_input_tokens=100,
            breaker=breaker,
            sleep=no_sleep,
        )
        task = asyncio.create_task(gate.complete(make_request()))
        await asyncio.wait_for(client.entered.wait(), timeout=1.0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        response = await gate.complete(make_request())

        assert response.text == "recovered"
        assert client.calls == 2

    async def test_zero_request_budget_refuses_before_calling_client(self) -> None:
        """An exhausted request budget stops the provider call."""
        from ontogr_mem.request_context import RequestContext, bind_context

        client = ScriptedLLM(["unused"])
        gate = make_gate(client)
        context = RequestContext("owner", "tenant-a", latency_budget_s=0.0)

        with bind_context(context), pytest.raises(DeadlineExceeded):
            await gate.complete(make_request())

        assert client.requests == []

    async def test_request_deadline_timeout_raises_deadline_error(self) -> None:
        """A shorter request budget is distinguished from the call timeout."""
        from ontogr_mem.request_context import RequestContext, bind_context

        class SlowClient:
            """Takes longer than the request budget."""

            async def complete(self, request: LLMRequest) -> LLMResponse:
                await asyncio.sleep(1.0)
                return LLMResponse("late", 1, 1)

        gate = LLMGate(
            SlowClient(),
            max_input_tokens=100,
            timeout_s=5.0,
            settings=LLMSettings(),
            sleep=no_sleep,
        )
        context = RequestContext("owner", "tenant-a", latency_budget_s=0.05)

        with bind_context(context), pytest.raises(DeadlineExceeded):
            await gate.complete(make_request())

    async def test_complete_many_keeps_results_and_failures_in_request_order(
        self,
    ) -> None:
        """One failed completion is data and does not discard other results."""
        client = ScriptedLLM(["first", LLMCallError("HTTP 403"), "third"])
        gate = make_gate(client)

        items = await gate.complete_many(
            [make_request("one"), make_request("two"), make_request("three")]
        )

        assert [
            (item.index, item.response.text if item.response else None)
            for item in items
        ] == [
            (0, "first"),
            (1, None),
            (2, "third"),
        ]
        assert isinstance(items[1].error, LLMCallError)

    async def test_complete_many_keeps_oversized_input_as_an_item_error(self) -> None:
        """An input-cap error does not prevent the other items from completing."""
        client = ScriptedLLM(["ok"])
        gate = LLMGate(
            client,
            max_input_tokens=5,
            settings=LLMSettings(),
            sleep=no_sleep,
        )

        items = await gate.complete_many([make_request("x"), make_request("x" * 20)])

        assert items[0].response is not None
        assert items[0].response.text == "ok"
        assert items[1].response is None
        assert isinstance(items[1].error, InputTooLargeError)

    async def test_structured_retries_with_validation_feedback(self) -> None:
        """Malformed JSON and invalid fields are fed back before success."""
        client = ScriptedLLM(["not json", {"name": 5}, {"name": "Ana"}])
        gate = make_gate(client)

        person = await gate.structured(Person, "system", "find a person")

        assert person == Person(name="Ana")
        assert "name" in client.requests[2].prompt

    async def test_structured_failure_names_the_output_model(self) -> None:
        """Exhausted validation attempts raise a typed model-specific error."""
        client = ScriptedLLM(["bad", "still bad", "also bad"])
        gate = make_gate(client)

        with pytest.raises(StructuredOutputError, match="Person"):
            await gate.structured(Person, "system", "find a person")

    async def test_structured_accepts_markdown_fenced_json(self) -> None:
        """The shared JSON loader accepts a fenced object."""
        gate = make_gate(ScriptedLLM(['```json\n{"name": "Ana"}\n```']))

        assert await gate.structured(Person, "system", "find a person") == Person(
            name="Ana"
        )

    async def test_default_per_size_concurrency_limit_is_four(self) -> None:
        """Four calls enter together while a fifth waits for a permit."""

        class BlockingClient:
            """Tracks active calls and releases them as a group."""

            def __init__(self) -> None:
                self.entered = 0
                self.running = 0
                self.peak = 0
                self.four_entered = asyncio.Event()
                self.release = asyncio.Event()

            async def complete(self, request: LLMRequest) -> LLMResponse:
                self.entered += 1
                self.running += 1
                self.peak = max(self.peak, self.running)
                if self.entered == 4:
                    self.four_entered.set()
                await self.release.wait()
                self.running -= 1
                return LLMResponse("ok", 1, 1)

        client = BlockingClient()
        gate = make_gate(client)
        tasks = [asyncio.create_task(gate.complete(make_request())) for _ in range(5)]
        await asyncio.wait_for(client.four_entered.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert client.entered == 4

        client.release.set()
        await asyncio.gather(*tasks)

        assert client.peak == 4
        assert client.entered == 5
