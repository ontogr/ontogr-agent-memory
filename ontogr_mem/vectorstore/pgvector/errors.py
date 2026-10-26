"""Errors raised only by the pgvector backend."""

from ontogr_mem.vectorstore.errors import VectorStoreError


class InvalidIdentifierError(VectorStoreError):
    """A tenant id or a SQL identifier cannot be used safely."""


class TenantNotProvisionedError(VectorStoreError):
    """The tenant schema or its tables do not exist."""


class RecallBelowThresholdError(VectorStoreError):
    """The index returned fewer results than the exact scan for too many queries."""
