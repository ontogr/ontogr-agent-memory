"""Async storage contract for operation rows."""

from typing import Protocol

from ontogr_mem.data_models.usage import OperationUsage
from ontogr_mem.relational_store.operations.models import (
    _DEFAULT_OPERATION_USAGE,
    OperationKind,
    OperationOutcome,
    OperationRecord,
    TenantUsage,
)


class OperationStore(Protocol):
    """Persist operations from all Tenants and report their usage."""

    async def start(
        self,
        *,
        kind: OperationKind,
        principal_id: str,
        tenant: str | None,
        session_id: str | None = None,
        parent_id: str | None = None,
        trace_id: str = "",
    ) -> str:
        """Create a running operation and return its id."""
        ...

    async def finish(
        self,
        operation_id: str,
        *,
        outcome: OperationOutcome,
        usage: OperationUsage = _DEFAULT_OPERATION_USAGE,
        error: BaseException | None = None,
    ) -> None:
        """Finish a running operation with its final usage and optional error.

        Only ``ok`` and ``failed`` are terminal outcomes. An unknown or already
        finished id raises OperationStateError.
        """
        ...

    async def get(self, operation_id: str) -> OperationRecord | None:
        """Return an operation by id, or None when no row exists."""
        ...

    async def list_operations(
        self, *, tenant: str | None = None, limit: int = 50
    ) -> list[OperationRecord]:
        """Return newest operations first, optionally filtered by Tenant."""
        ...

    async def usage(self, tenant: str) -> TenantUsage:
        """Return totals for every operation row belonging to a Tenant."""
        ...
