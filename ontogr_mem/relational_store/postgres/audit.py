"""PostgreSQL audit log over a shared connection pool."""

from typing import TYPE_CHECKING

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.data_models.timestamps import to_micros
from ontogr_mem.errors import StoreError
from ontogr_mem.relational_store.audit.chain import (
    GENESIS_HASH,
    canonical_detail,
    chain_hash,
)
from ontogr_mem.relational_store.audit.models import (
    AuditEntry,
    AuditRow,
)
from ontogr_mem.relational_store.audit.rows import (
    AUDIT_COLUMNS,
    audit_row_from_row,
)
from ontogr_mem.relational_store.engine import pg_connection


if TYPE_CHECKING:
    import asyncpg


_SELECT_ROWS = f"SELECT {AUDIT_COLUMNS} FROM audit_rows"

# Appends serialize on one advisory key so concurrent writers cannot read the
# same chain head; the number is arbitrary but must never be reused elsewhere.
_CHAIN_LOCK_KEY = 90210


class PostgresAuditLog:
    """Store audit rows in Postgres behind a shared pool.

    Args:
        pool: Process-wide connection pool; the caller runs
            ``ensure_postgres_schema`` before constructing the log.
        clock: Source of timezone-aware timestamps for new rows.
    """

    def __init__(self, pool: "asyncpg.Pool", clock: Clock) -> None:
        """Bind the log to its pool without taking a connection."""
        self._pool = pool
        self._clock = clock

    async def append(self, entry: AuditEntry) -> AuditRow:
        """Append one event under the chain lock and return its linked row.

        Raises:
            ValueError: If the clock returns a naive datetime.
            StoreError: If Postgres cannot append the row.
        """
        recorded_at = self._clock.now()
        stored_at = to_micros(recorded_at)
        detail = canonical_detail(entry.detail)
        try:
            async with (
                pg_connection(self._pool) as connection,
                connection.transaction(),
            ):
                await connection.execute(
                    "SELECT pg_advisory_xact_lock($1)", _CHAIN_LOCK_KEY
                )
                previous = await connection.fetchval(
                    "SELECT row_hash FROM audit_rows ORDER BY seq DESC LIMIT 1"
                )
                previous_hash = GENESIS_HASH if previous is None else previous
                row_hash = chain_hash(previous_hash, recorded_at, entry)
                seq = await connection.fetchval(
                    "INSERT INTO audit_rows (recorded_at, action, outcome, "
                    "principal_id, tenant, acting_for, target, trace_id, "
                    "detail, previous_hash, row_hash) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) "
                    "RETURNING seq",
                    stored_at,
                    entry.action.value,
                    entry.outcome.value,
                    entry.principal_id,
                    entry.tenant,
                    entry.acting_for,
                    entry.target,
                    entry.trace_id,
                    detail,
                    previous_hash,
                    row_hash,
                )
        except Exception as exc:
            raise StoreError("could not append the audit row") from exc
        if seq is None:
            raise StoreError("Postgres did not return the inserted audit sequence")
        return AuditRow(
            seq=seq,
            recorded_at=recorded_at,
            entry=entry,
            previous_hash=previous_hash,
            row_hash=row_hash,
        )

    async def verify(self) -> int | None:
        """Return the first broken row, or None when the chain verifies.

        This detects sequence gaps, edits, and deletions in the middle. A
        deleted final row is undetectable until another row is appended. A
        database owner can rewrite later hashes and leave a verifying chain.

        Raises:
            StoreError: If Postgres cannot read the audit rows.
        """
        try:
            rows = await self._fetch_ordered()
        except Exception as exc:
            raise StoreError("could not read the audit chain") from exc
        previous_hash = GENESIS_HASH
        for expected_seq, stored_row in enumerate(rows, start=1):
            try:
                row = audit_row_from_row(stored_row)
            except (TypeError, ValueError):
                return int(stored_row[0])
            if row.seq != expected_seq:
                return row.seq
            if row.previous_hash != previous_hash:
                return row.seq
            if chain_hash(previous_hash, row.recorded_at, row.entry) != row.row_hash:
                return row.seq
            previous_hash = row.row_hash
        return None

    async def list_rows(
        self, *, tenant: str | None = None, limit: int = 100
    ) -> list[AuditRow]:
        """Return newest rows first, optionally filtered to one Tenant.

        Raises:
            ValueError: If ``limit`` is less than one.
            InvalidTenantError: If ``tenant`` is not a valid Tenant id.
            StoreError: If Postgres cannot read the audit rows.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if tenant is not None:
            validate_tenant_id(tenant)
        try:
            if tenant is None:
                rows = await self._fetch(
                    f"{_SELECT_ROWS} ORDER BY seq DESC LIMIT $1", limit
                )
            else:
                rows = await self._fetch(
                    f"{_SELECT_ROWS} WHERE tenant = $1 ORDER BY seq DESC LIMIT $2",
                    tenant,
                    limit,
                )
            return [audit_row_from_row(row) for row in rows]
        except Exception as exc:
            raise StoreError("could not list audit rows") from exc

    async def _fetch_ordered(self) -> list["asyncpg.Record"]:
        """Return every audit row oldest first."""
        async with pg_connection(self._pool) as connection:
            return await connection.fetch(f"{_SELECT_ROWS} ORDER BY seq")

    async def _fetch(self, query: str, *values: object) -> list["asyncpg.Record"]:
        """Return rows for one select with bound values."""
        async with pg_connection(self._pool) as connection:
            return await connection.fetch(query, *values)
