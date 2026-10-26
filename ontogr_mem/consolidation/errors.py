"""Errors raised by the consolidation package."""

from ontogr_mem.errors import AgentMemoryError


class ConsolidationError(AgentMemoryError):
    """Base class of the consolidation errors."""


class StoreError(ConsolidationError):
    """The consolidation store failed or returned a document it cannot read."""


class RegistryError(ConsolidationError):
    """The stage list is invalid."""
