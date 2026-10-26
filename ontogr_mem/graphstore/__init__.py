"""Shared graph adapter package, not a kernel, with backend-neutral execution."""

from ontogr_mem.graphstore.config import ResilienceSettings
from ontogr_mem.graphstore.errors import (
    CircuitOpenError,
    ConstraintViolationError,
    DatabaseUnavailableError,
    GraphStoreError,
    QueryTimeoutError,
    SchemaDriftError,
    StoreQueryError,
    ValidationError,
)
from ontogr_mem.graphstore.executor import QuerySpec, Rows
from ontogr_mem.graphstore.indices import (
    RangeIndex,
    SchemaPart,
    UniqueConstraint,
)
from ontogr_mem.graphstore.protocol import GraphStore, ScopedRunner
from ontogr_mem.graphstore.types import (
    Edge,
    EdgeSelector,
    GraphHandle,
    GraphStats,
    Guard,
    Health,
    Node,
    NodeRow,
    NodeSelector,
    WriteError,
    WriteResult,
)
from ontogr_mem.observability.metrics import Metrics


__all__ = [
    "CircuitOpenError",
    "ConstraintViolationError",
    "DatabaseUnavailableError",
    "Edge",
    "EdgeSelector",
    "GraphHandle",
    "GraphStats",
    "GraphStore",
    "GraphStoreError",
    "Guard",
    "Health",
    "Metrics",
    "Node",
    "NodeRow",
    "NodeSelector",
    "QuerySpec",
    "QueryTimeoutError",
    "RangeIndex",
    "ResilienceSettings",
    "Rows",
    "SchemaDriftError",
    "SchemaPart",
    "ScopedRunner",
    "StoreQueryError",
    "UniqueConstraint",
    "ValidationError",
    "WriteError",
    "WriteResult",
]
