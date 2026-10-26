"""Tests for the service error codes and the error classes."""

import pytest

from ontogr_mem.errors import AgentMemoryError
from ontogr_mem.service.errors import (
    ConfirmationRequiredError,
    ConflictError,
    ErrorCode,
    InternalError,
    InvalidArgumentError,
    LLMUnavailableError,
    ObjectNotFoundError,
    PayloadTooLargeError,
    RateLimitedError,
    ServiceError,
)


ERROR_CLASSES = (
    InvalidArgumentError,
    ObjectNotFoundError,
    ConflictError,
    LLMUnavailableError,
    RateLimitedError,
    PayloadTooLargeError,
    ConfirmationRequiredError,
    InternalError,
)


class TestErrorCodes:
    """The code set and the one-class-per-code rule."""

    def test_each_code_has_exactly_one_error_class(self):
        """Every code has one error subclass and no code is claimed twice."""
        codes = [error.code for error in ServiceError.__subclasses__()]

        assert len(codes) == len(set(codes))
        assert set(codes) == set(ErrorCode)

    @pytest.mark.parametrize("error_class", ERROR_CLASSES)
    def test_every_error_is_an_ontogr_mem_error(self, error_class):
        """A caller can catch any service error through the package root."""
        assert issubclass(error_class, AgentMemoryError)


class TestServiceError:
    """Message, hint and string form of a service error."""

    def test_keeps_message_and_hint(self):
        """The error keeps both parts and reads as the message."""
        error = InvalidArgumentError("bad", hint="fix it")

        assert error.message == "bad"
        assert error.hint == "fix it"
        assert str(error) == "bad"

    def test_hint_defaults_to_empty(self):
        """A hint is optional."""
        assert InvalidArgumentError("bad").hint == ""

    def test_payload_too_large_is_a_value_error(self):
        """A validator can raise the size error directly."""
        assert isinstance(PayloadTooLargeError("x"), ValueError)
