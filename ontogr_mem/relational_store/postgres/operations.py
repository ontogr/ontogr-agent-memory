"""PostgreSQL operation rows over a shared connection pool."""

from typing import TYPE_CHECKING
from uuid import uuid4

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.data_models.timestamps import OPEN_MICROS, to_micros
from ontogr_mem.data_models.usage import OperationUsage
from ontogr_mem.relational_store.engine import pg_connection
from ontogr_mem.relational_store.operations.errors import OperationStateError
from ontogr_mem.relational_store.operations.models import (
    _DEFAULT_OPERATION_USAGE,
    OperationKind,
    OperationOutcome,
    OperationRecord,
    TenantUsage,
    summarise_error,
)
from ontogr_mem.relational_store.operations.rows import (
    OPERATION_COLUMNS,
    record_from_row,
)


if TYPE_CHECKING:
    import asyncpg


_SELECT_ROWS = f"SELECT {OPERATION_COLUMNS} FROM operation_rows"


class PostgresOperationStore:
    """Persist operations in Postgres behind a shared pool.

    Args:
        pool: Process-wide connection pool; the caller runs
            ``ensure_postgres_schema`` before constructing the store.
        clock: Source of timestamps for new and finished operations.
    """

    def __init__(self, pool: "asyncpg.Pool", clock: Clock) -> None:
        """Bind the store to its pool without taking a connection."""
        self._pool = pool
        self._clock = clock

    async def mark_interrupted_runs(self) -> None:
        """Mark operations left running by a previous process as failed."""
        async with pg_connection(self._pool) as connection:
            await connection.execute(
                "UPDATE operation_rows SET ended_at = $1, outcome = 'failed', "
                "error_class = 'Interrupted', "
                "error = 'operation was still running when the store opened' "
                "WHERE outcome = 'running'",
                to_micros(self._clock.now()),
            )

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
        async with pg_connection(self._pool) as connection:
            await connection.execute(
                "INSERT INTO operation_rows (id, parent_id, kind, principal_id, "
                "tenant, session_id, trace_id, started_at, ended_at, outcome, "
                "tokens_in, tokens_out, cost_usd) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'running', 0, 0, 0.0)",
                operation_id,
                parent_id,
                kind.value,
                principal_id,
                tenant,
                session_id,
                trace_id,
                to_micros(self._clock.now()),
                OPEN_MICROS,
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
        error_class, error_summary = (
            (None, None) if error is None else summarise_error(error)
        )
        async with pg_connection(self._pool) as connection:
            row = await connection.fetchrow(
                "UPDATE operation_rows SET ended_at = $1, outcome = $2, "
                "error_class = $3, error = $4, tokens_in = $5, tokens_out = $6, "
                "cost_usd = $7 WHERE id = $8 AND outcome = 'running' "
                "RETURNING id",
                to_micros(self._clock.now()),
                outcome.value,
                error_class,
                error_summary,
                usage.tokens_in,
                usage.tokens_out,
                usage.cost_usd,
                operation_id,
            )
            if row is None:
                raise OperationStateError(f"operation {operation_id!r} is not running")

    async def get(self, operation_id: str) -> OperationRecord | None:
        """Return an operation by id, or None when no row exists."""
        async with pg_connection(self._pool) as connection:
            row = await connection.fetchrow(
                f"{_SELECT_ROWS} WHERE id = $1", operation_id
            )
            return None if row is None else record_from_row(row)

    async def list_operations(
        self, *, tenant: str | None = None, limit: int = 50
    ) -> list[OperationRecord]:
        """Return newest operations first, optionally filtered by Tenant."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if tenant is not None:
            validate_tenant_id(tenant)
        async with pg_connection(self._pool) as connection:
            if tenant is None:
                rows = await connection.fetch(
                    f"{_SELECT_ROWS} ORDER BY started_at DESC, seq DESC LIMIT $1",
                    limit,
                )
            else:
                rows = await connection.fetch(
                    f"{_SELECT_ROWS} WHERE tenant = $1 "
                    "ORDER BY started_at DESC, seq DESC LIMIT $2",
                    tenant,
                    limit,
                )
            return [record_from_row(row) for row in rows]

    async def usage(self, tenant: str) -> TenantUsage:
        """Return totals for every operation row belonging to a Tenant."""
        validate_tenant_id(tenant)
        async with pg_connection(self._pool) as connection:
            row = await connection.fetchrow(
                "SELECT COUNT(*), COALESCE(SUM(tokens_in), 0), "
                "COALESCE(SUM(tokens_out), 0), "
                "COALESCE(SUM(CASE WHEN cost_usd IS NULL THEN 0 "
                "ELSE cost_usd END), 0.0), "
                "COALESCE(SUM(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END), 0) "
                "FROM operation_rows WHERE tenant = $1",
                tenant,
            )
            if row is None:
                return TenantUsage(0, 0, 0, 0.0, 0)
            return TenantUsage(
                operations=int(row[0]),
                tokens_in=int(row[1]),
                tokens_out=int(row[2]),
                cost_usd=float(row[3]),
                unpriced_operations=int(row[4]),
            )
