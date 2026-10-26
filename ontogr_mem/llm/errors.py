"""Errors raised by the language model layer."""

from ontogr_mem.errors import AgentMemoryError


class LLMError(AgentMemoryError):
    """Base class for language model call failures."""


class LLMCallError(LLMError):
    """A client failed. Clients raise this and chain the original cause."""


class InputTooLargeError(LLMError):
    """A prompt exceeds the hard input-token cap."""


class CircuitOpenError(LLMError):
    """The model breaker refused a call while the provider is unavailable."""


class DeadlineExceededError(LLMError):
    """The request latency budget expired before the model call completed."""


class StructuredOutputError(LLMError):
    """The model did not return a value that validates against its schema."""
