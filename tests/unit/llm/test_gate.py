"""Tests for the model call gate."""

import asyncio

import pytest

from ontogr_mem.llm import (
    InputTooLargeError,
    LLMGate,
    LLMRequest,
    LLMResponse,
    ModelSize,
    estimate_tokens,
)


class SlowClient:
    """Tracks the highest number of calls in flight."""

    def __init__(self, delay: float) -> None:
        """Create the helper."""
        self.delay = delay
        self.calls = 0
        self.running = 0
        self.peak = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete."""
        self.calls += 1
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(self.delay)
        self.running -= 1
        return LLMResponse("ok", 1, 1)


def request(size: ModelSize = ModelSize.SMALL, prompt: str = "hi") -> LLMRequest:
    """Request."""
    return LLMRequest(system="", prompt=prompt, model_size=size)


class TestLLMGate:
    """Behavior of LLMGate."""

    @pytest.mark.parametrize(("text", "tokens"), [("", 0), ("abcd", 1), ("abcde", 2)])
    def test_estimates_one_token_per_four_characters_rounded_up(self, text, tokens):
        """Estimates one token per four characters rounded up."""
        assert estimate_tokens(text) == tokens

    async def test_refuses_large_prompt_before_calling_the_client(self):
        """Refuses large prompt before calling the client."""
        client = SlowClient(0)
        gate = LLMGate(client, max_input_tokens=2)

        with pytest.raises(InputTooLargeError):
            await gate.complete(request(prompt="x" * 9))

        assert client.calls == 0

    async def test_limits_parallel_calls_per_model_size(self):
        """Limits parallel calls per model size."""
        client = SlowClient(0.01)
        gate = LLMGate(client, max_input_tokens=100, concurrency={ModelSize.SMALL: 2})

        await asyncio.gather(*(gate.complete(request()) for _ in range(6)))

        assert client.peak == 2

    async def test_slow_call_raises_timeout(self):
        """Slow call raises timeout."""
        gate = LLMGate(SlowClient(1), max_input_tokens=100, timeout_s=0.01)

        with pytest.raises(TimeoutError):
            await gate.complete(request())
