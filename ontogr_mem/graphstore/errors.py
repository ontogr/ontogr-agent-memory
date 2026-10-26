"""Errors raised by the graph store."""

from ontogr_mem.errors import StoreError


class GraphStoreError(StoreError):
    """Base class for every error the graph store raises."""


class ValidationError(GraphStoreError, ValueError):
    """Input rejected before any query was sent."""


class DatabaseUnavailableError(GraphStoreError):
    """The server could not be reached or the pool is exhausted."""


class CircuitOpenError(DatabaseUnavailableError):
    """The circuit breaker is open; the request was not sent."""


class StoreQueryError(GraphStoreError):
    """The server rejected or failed a query."""


class QueryTimeoutError(StoreQueryError):
    """The server killed the query after its timeout."""


class ConstraintViolationError(StoreQueryError):
    """A uniqueness constraint rejected a write."""


class AlreadyExistsError(StoreQueryError):
    """The object the statement creates already exists on the server."""


class NotFoundError(StoreQueryError):
    """The object the statement names does not exist on the server."""


class DeleteRefusedError(StoreQueryError):
    """The delete was refused because a selected node still has edges."""


class SchemaDriftError(GraphStoreError):
    """The live indexes or constraints differ from the declared schema."""
