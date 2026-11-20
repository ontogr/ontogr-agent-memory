"""Errors from job persistence and runner configuration."""

from ontogr_mem.errors import AgentMemoryError, StoreError


class IdempotencyConflictError(StoreError):
    """An idempotency key was reused with a different request."""


class UnknownJobOpError(AgentMemoryError, ValueError):
    """A job names an operation that has no handler."""


class LeaseLostError(StoreError):
    """A write was refused because this process no longer owns the lease."""
