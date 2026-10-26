"""Errors raised by operation storage."""

from ontogr_mem.errors import AgentMemoryError


class OperationStateError(AgentMemoryError):
    """An operation is unknown or is no longer running."""
