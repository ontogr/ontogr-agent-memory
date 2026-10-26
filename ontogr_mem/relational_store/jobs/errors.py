"""Errors from job persistence and runner configuration."""

from ontogr_mem.errors import AgentMemoryError, StoreError


class IdempotencyConflictError(StoreError):
    """An idempotency key was reused with a different request."""


class EphemeralStateError(AgentMemoryError):
    """An in-memory state store was requested without explicit permission."""


class UnknownJobOpError(AgentMemoryError, ValueError):
    """A job names an operation that has no handler."""
