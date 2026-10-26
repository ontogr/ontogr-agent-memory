"""SQLite-backed operation rows on a caller-owned connection."""

import sqlite3
from uuid import uuid4

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.data_models.timestamps import OPEN_MICROS, to_micros
from ontogr_mem.data_models.usage import OperationUsage
from ontogr_mem.errors import StoreError
from ontogr_mem.relational_store.operations.errors import OperationStateError
from ontogr_mem.relational_store.operations.models import (
    _DEFAULT_OPERATION_USAGE,
    OperationKind,
    OperationOutcome,
    OperationRecord,
    TenantUsage,
    summarise_error,
)
from ontogr_mem.relational_store.operations.rows import record_from_row
from ontogr_mem.relational_store.schema.runner import ensure_sqlite_schema


_COLUMNS = (
    "id, parent_id, kind, principal_id, tenant, session_id, trace_id, "
    "started_at, ended_at, outcome, error_class, error, tokens_in, tokens_out, "
    "cost_usd"
)


class SqliteOperationStore:
    """Persist operations on a caller-owned SQLite connection.

    Args:
        connection: SQLite connection opened in autocommit mode.
        clock: Source of timestamps for new and finished operations.
    """

    def __init__(self, connection: sqlite3.Connection, clock: Clock) -> None:
        """Apply migrations and mark operations left running as interrupted."""
        self._connection = connection
        self._clock = clock
        try:
            ensure_sqlite_schema(connection)
            connection.execute(
                "UPDATE operation_rows SET ended_at = ?, outcome = 'failed', "
                "error_class = 'Interrupted', "
                "error = 'operation was still running when the store opened' "
                "WHERE outcome = 'running'",
                (to_micros(clock.now()),),
            )
        except sqlite3.Error as exc:
            raise StoreError("could not initialize operation storage") from exc

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
        try:
            self._connection.execute(
                "INSERT INTO operation_rows (id, parent_id, kind, principal_id, "
                "tenant, session_id, trace_id, started_at, ended_at, outcome, "
                "tokens_in, tokens_out, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', 0, 0, 0.0)",
                (
                    operation_id,
                    parent_id,
                    kind.value,
                    principal_id,
                    tenant,
                    session_id,
                    trace_id,
                    to_micros(self._clock.now()),
                    OPEN_MICROS,
                ),
            )
        except sqlite3.Error as exc:
            raise StoreError("could not start operation") from exc
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
        try:
            cursor = self._connection.execute(
                "UPDATE operation_rows SET ended_at = ?, outcome = ?, "
                "error_class = ?, error = ?, tokens_in = ?, tokens_out = ?, "
                "cost_usd = ? WHERE id = ? AND outcome = 'running'",
                (
                    to_micros(self._clock.now()),
                    outcome.value,
                    error_class,
                    error_summary,
                    usage.tokens_in,
                    usage.tokens_out,
                    usage.cost_usd,
                    operation_id,
                ),
            )
        except sqlite3.Error as exc:
            raise StoreError("could not finish operation") from exc
        if cursor.rowcount == 0:
            raise OperationStateError(f"operation {operation_id!r} is not running")

    async def get(self, operation_id: str) -> OperationRecord | None:
        """Return an operation by id, or None when no row exists."""
        try:
            row = self._connection.execute(
                f"SELECT {_COLUMNS} FROM operation_rows WHERE id = ?",
                (operation_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError("could not get operation") from exc
        return None if row is None else record_from_row(row)

    async def list_operations(
        self, *, tenant: str | None = None, limit: int = 50
    ) -> list[OperationRecord]:
        """Return newest operations first, optionally filtered by Tenant."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if tenant is not None:
            validate_tenant_id(tenant)
        query = f"SELECT {_COLUMNS} FROM operation_rows"
        parameters: tuple[object, ...]
        if tenant is None:
            parameters = (limit,)
        else:
            query += " WHERE tenant = ?"
            parameters = (tenant, limit)
        query += " ORDER BY started_at DESC, rowid DESC LIMIT ?"
        try:
            rows = self._connection.execute(query, parameters).fetchall()
        except sqlite3.Error as exc:
            raise StoreError("could not list operations") from exc
        return [record_from_row(row) for row in rows]

    async def usage(self, tenant: str) -> TenantUsage:
        """Return totals for every operation row belonging to a Tenant."""
        validate_tenant_id(tenant)
        try:
            row = self._connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(tokens_in), 0), "
                "COALESCE(SUM(tokens_out), 0), "
                "COALESCE(SUM(CASE WHEN cost_usd IS NULL THEN 0 "
                "ELSE cost_usd END), 0.0), "
                "COALESCE(SUM(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END), 0) "
                "FROM operation_rows WHERE tenant = ?",
                (tenant,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError("could not total operation usage") from exc
        if row is None:
            return TenantUsage(0, 0, 0, 0.0, 0)
        return TenantUsage(
            operations=int(row[0]),
            tokens_in=int(row[1]),
            tokens_out=int(row[2]),
            cost_usd=float(row[3]),
            unpriced_operations=int(row[4]),
        )
