"""Typed service errors with a stable code per error class."""

from enum import StrEnum
from typing import ClassVar

from ontogr_mem.errors import AgentMemoryError


class ErrorCode(StrEnum):
    """Stable error codes returned by the memory service."""

    INVALID_ARGUMENT = "invalid_argument"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    LLM_UNAVAILABLE = "llm_unavailable"
    RATE_LIMITED = "rate_limited"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    CONFIRMATION_REQUIRED = "confirmation_required"
    INTERNAL = "internal"


class ServiceError(AgentMemoryError):
    """An error a caller can act on: a stable code, a message and a hint.

    Attributes:
        code: The wire code of the error class.
        message: What went wrong, safe to show a caller.
        hint: What the caller can do about it, empty when there is nothing.
    """

    code: ClassVar[ErrorCode]

    def __init__(self, message: str, hint: str = "") -> None:
        """Keep the message and the hint; ``str(error)`` is the message.

        Args:
            message: What went wrong.
            hint: What the caller can do about it.
        """
        super().__init__(message)
        self.message = message
        self.hint = hint


class InvalidArgumentError(ServiceError):
    """A request field is missing, malformed or out of range."""

    code: ClassVar[ErrorCode] = ErrorCode.INVALID_ARGUMENT


class ObjectNotFoundError(ServiceError):
    """A tenant or an object does not exist, or the caller cannot see it."""

    code: ClassVar[ErrorCode] = ErrorCode.NOT_FOUND


class ConflictError(ServiceError):
    """The request collides with the current state, such as a duplicate job."""

    code: ClassVar[ErrorCode] = ErrorCode.CONFLICT


class LLMUnavailableError(ServiceError):
    """A verb that needs a model was called without one configured."""

    code: ClassVar[ErrorCode] = ErrorCode.LLM_UNAVAILABLE


class RateLimitedError(ServiceError):
    """The caller spent its tokens for this minute."""

    code: ClassVar[ErrorCode] = ErrorCode.RATE_LIMITED


class PayloadTooLargeError(ServiceError, ValueError):
    """A request field is over its byte limit."""

    code: ClassVar[ErrorCode] = ErrorCode.PAYLOAD_TOO_LARGE


class ConfirmationRequiredError(ServiceError):
    """A destructive verb was called without the token of its dry run."""

    code: ClassVar[ErrorCode] = ErrorCode.CONFIRMATION_REQUIRED


class InternalError(ServiceError):
    """An unexpected failure inside the service."""

    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL
