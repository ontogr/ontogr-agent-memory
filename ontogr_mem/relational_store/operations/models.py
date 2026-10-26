"""Operation kinds, outcomes, records and usage summaries."""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.data_models.usage import OperationUsage


MINTED_KEY_PATTERN: Final = re.compile(r"mk_[A-Za-z0-9_-]+")


class OperationKind(StrEnum):
    """Kind of work recorded by an operation row."""

    WRITE = "write"
    RETRIEVE = "retrieve"
    INVALIDATE = "invalidate"
    ERASE = "erase"
    EXPORT = "export"
    HISTORY = "history"


class OperationOutcome(StrEnum):
    """Lifecycle state or terminal result of an operation."""

    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"


_DEFAULT_OPERATION_USAGE = OperationUsage()


@dataclass(frozen=True, slots=True)
class OperationRecord:
    """Stored details and lifecycle state for one operation.

    Attributes:
        id: Unique operation id.
        parent_id: Parent operation id, if this operation has one.
        kind: Work type recorded by this row.
        principal_id: Principal that started the operation.
        tenant: Tenant that owns the operation, or None when unscoped.
        session_id: Caller session id, if present.
        trace_id: Trace id shared by related operations.
        started_at: Time the operation started.
        ended_at: Time the operation ended, or None while running.
        outcome: Current lifecycle state or terminal result.
        error_class: Exception class for a failed operation, if available.
        error: Redacted one-line error summary, if available.
        usage: Token counts and known model cost.
    """

    id: str
    parent_id: str | None
    kind: OperationKind
    principal_id: str
    tenant: str | None
    session_id: str | None
    trace_id: str
    started_at: datetime
    ended_at: datetime | None
    outcome: OperationOutcome
    error_class: str | None
    error: str | None
    usage: OperationUsage

    def __post_init__(self) -> None:
        """Validate the required principal and optional Tenant id."""
        if not self.principal_id:
            raise ValueError("principal id must not be empty")
        if self.tenant is not None:
            validate_tenant_id(self.tenant)

    @property
    def cost_unknown(self) -> bool:
        """Return whether one or more model calls had unknown cost."""
        return self.usage.cost_usd is None


@dataclass(frozen=True, slots=True)
class TenantUsage:
    """Aggregate operation and model usage for one Tenant.

    Attributes:
        operations: Number of operation rows, including running rows.
        tokens_in: Total input tokens.
        tokens_out: Total output tokens.
        cost_usd: Sum of priced operation costs.
        unpriced_operations: Number of operation rows with unknown cost.
    """

    operations: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    unpriced_operations: int


def summarise_error(error: BaseException) -> tuple[str, str]:
    """Return an exception class and a redacted, single-line summary.

    The summary keeps only the first line, masks minted keys and is limited to
    200 characters.
    """
    lines = str(error).splitlines()
    first_line = lines[0].strip() if lines else ""
    return type(error).__name__, MINTED_KEY_PATTERN.sub("mk_***", first_line)[:200]
