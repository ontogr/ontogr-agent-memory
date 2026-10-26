"""In-memory operation storage for one event loop."""

from dataclasses import replace
from uuid import uuid4

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.data_models.usage import OperationUsage
from ontogr_mem.relational_store.operations.errors import OperationStateError
from ontogr_mem.relational_store.operations.models import (
    _DEFAULT_OPERATION_USAGE,
    OperationKind,
    OperationOutcome,
    OperationRecord,
    TenantUsage,
    summarise_error,
)


class MemoryOperationStore:
    """Keep operation rows in insertion order without durable storage."""

    def __init__(self, clock: Clock) -> None:
        """Create an empty operation store using the supplied clock."""
        self._clock = clock
        self._records: dict[str, OperationRecord] = {}

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
        if not principal_id:
            raise ValueError("principal id must not be empty")
        if tenant is not None:
            validate_tenant_id(tenant)
        operation_id = uuid4().hex
        self._records[operation_id] = OperationRecord(
            id=operation_id,
            parent_id=parent_id,
            kind=kind,
            principal_id=principal_id,
            tenant=tenant,
            session_id=session_id,
            trace_id=trace_id,
            started_at=self._clock.now(),
            ended_at=None,
            outcome=OperationOutcome.RUNNING,
            error_class=None,
            error=None,
            usage=OperationUsage(),
        )
        return operation_id

    async def finish(
        self,
        operation_id: str,
        *,
        outcome: OperationOutcome,
        usage: OperationUsage = _DEFAULT_OPERATION_USAGE,
        error: BaseException | None = None,
    ) -> None:
        """Finish a running operation with its final usage and optional error."""
        if outcome is OperationOutcome.RUNNING:
            raise ValueError("finish outcome must be terminal")
        if outcome is OperationOutcome.OK and error is not None:
            raise ValueError("successful operation cannot have an error")
        record = self._records.get(operation_id)
        if record is None or record.outcome is not OperationOutcome.RUNNING:
            raise OperationStateError(f"operation {operation_id!r} is not running")

        error_class, error_summary = (
            (None, None) if error is None else summarise_error(error)
        )
        self._records[operation_id] = replace(
            record,
            ended_at=self._clock.now(),
            outcome=outcome,
            error_class=error_class,
            error=error_summary,
            usage=usage,
        )

    async def get(self, operation_id: str) -> OperationRecord | None:
        """Return an operation by id, or None when no row exists."""
        return self._records.get(operation_id)

    async def list_operations(
        self, *, tenant: str | None = None, limit: int = 50
    ) -> list[OperationRecord]:
        """Return newest operations first, optionally filtered by Tenant."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if tenant is not None:
            validate_tenant_id(tenant)
        records = list(reversed(self._records.values()))
        if tenant is not None:
            records = [record for record in records if record.tenant == tenant]
        return sorted(records, key=lambda record: record.started_at, reverse=True)[
            :limit
        ]

    async def usage(self, tenant: str) -> TenantUsage:
        """Return totals for every operation row belonging to a Tenant."""
        validate_tenant_id(tenant)
        records = [
            record for record in self._records.values() if record.tenant == tenant
        ]
        return TenantUsage(
            operations=len(records),
            tokens_in=sum(record.usage.tokens_in for record in records),
            tokens_out=sum(record.usage.tokens_out for record in records),
            cost_usd=sum(
                (record.usage.cost_usd or 0.0)
                for record in records
                if record.usage.cost_usd is not None
            ),
            unpriced_operations=sum(record.cost_unknown for record in records),
        )
