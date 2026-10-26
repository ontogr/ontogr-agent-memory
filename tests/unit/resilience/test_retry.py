"""Retry classification and bounded delay behavior for model calls."""

import random

import pytest

from ontogr_mem.llm import LLMCallError
from ontogr_mem.resilience import Kind, backoff_delay, classify


TERMINAL_MESSAGES = [
    "HTTP 400",
    "HTTP 401",
    "HTTP 402",
    "HTTP 403",
    "HTTP 404",
    "HTTP 422",
    "insufficient quota",
    "insufficient_quota",
    "quota exceeded",
    "quota_exceeded",
    "billing",
    "credit balance",
    "out of credits",
    "invalid api key",
    "unauthorized",
    "unauthorised",
    "permission denied",
    "authentication",
    "api_key is not set",
    "content policy",
    "context length",
    "too large",
]


class TestClassify:
    """Terminal and transient provider errors are distinguished by content."""

    @pytest.mark.parametrize("message", TERMINAL_MESSAGES)
    def test_terminal_messages_are_not_retryable(self, message: str) -> None:
        """Every documented status or terminal phrase is recognized."""
        assert classify(LLMCallError(message)) is Kind.TERMINAL

    def test_message_without_terminal_pattern_is_transient(self) -> None:
        """An unrecognized client failure is eligible for retry."""
        assert classify(LLMCallError("connection reset")) is Kind.TRANSIENT

    def test_terminal_cause_is_found(self) -> None:
        """A wrapped provider error keeps its terminal classification."""
        cause = ValueError("credit balance depleted")
        error = LLMCallError("provider failed")
        error.__cause__ = cause

        assert classify(error) is Kind.TERMINAL

    def test_cause_cycle_terminates(self) -> None:
        """Repeated causes do not make classification loop forever."""
        first = LLMCallError("connection reset")
        second = ValueError("still transient")
        first.__cause__ = second
        second.__cause__ = first

        assert classify(first) is Kind.TRANSIENT


class TestBackoffDelay:
    """Equal-jitter delays stay within the exponential cap."""

    @pytest.mark.parametrize("attempt", [0, 1, 10])
    def test_delay_is_between_half_and_all_of_the_capped_value(
        self, attempt: int
    ) -> None:
        """Each delay is positive and no greater than its configured cap."""
        base_s, max_s = 0.5, 8.0
        capped = min(max_s, base_s * 2**attempt)

        delay = backoff_delay(
            attempt,
            base_s=base_s,
            max_s=max_s,
            rng=random.Random(45),
        )

        assert 0.5 * capped <= delay <= capped
