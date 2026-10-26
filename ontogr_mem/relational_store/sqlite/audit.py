"""SQLite audit log implementation using a caller-owned connection."""

import sqlite3
from contextlib import suppress

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
from ontogr_mem.relational_store.audit.rows import audit_row_from_row
from ontogr_mem.relational_store.schema.runner import ensure_sqlite_schema


_SELECT_ROWS = (
    "SELECT seq, recorded_at, action, outcome, principal_id, tenant, acting_for, "
    "target, trace_id, detail, previous_hash, row_hash FROM audit_rows"
)


class SqliteAuditLog:
    """Store audit rows in a SQLite connection supplied by the caller.

    Args:
        connection: SQLite connection opened in autocommit mode.
        clock: Source of timezone-aware timestamps for new rows.
    """

    def __init__(self, connection: sqlite3.Connection, clock: Clock) -> None:
        """Apply schema migrations on the supplied connection.

        Raises:
            StoreError: If the schema cannot be prepared.
        """
        self._connection = connection
        self._clock = clock
        try:
            ensure_sqlite_schema(self._connection)
        except sqlite3.Error as exc:
            raise StoreError("could not prepare the audit schema") from exc

    async def append(self, entry: AuditEntry) -> AuditRow:
        """Append one event in a write transaction and return its chain row.

        Raises:
            ValueError: If the clock returns a naive datetime.
            StoreError: If SQLite cannot append the row.
        """
        recorded_at = self._clock.now()
        stored_at = to_micros(recorded_at)
        detail = canonical_detail(entry.detail)
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            previous = self._connection.execute(
                "SELECT row_hash FROM audit_rows ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            previous_hash = GENESIS_HASH if previous is None else previous[0]
            row_hash = chain_hash(previous_hash, recorded_at, entry)
            cursor = self._connection.execute(
                "INSERT INTO audit_rows (recorded_at, action, outcome, principal_id, "
                "tenant, acting_for, target, trace_id, detail, previous_hash, "
                "row_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
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
                ),
            )
            self._connection.execute("COMMIT")
        except sqlite3.Error as exc:
            with suppress(sqlite3.Error):
                self._connection.execute("ROLLBACK")
            raise StoreError("could not append the audit row") from exc
        seq = cursor.lastrowid
        if seq is None:
            raise StoreError("SQLite did not return the inserted audit sequence")
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
            StoreError: If SQLite cannot read the audit rows.
        """
        try:
            rows = self._connection.execute(_SELECT_ROWS + " ORDER BY seq").fetchall()
        except sqlite3.Error as exc:
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
            StoreError: If SQLite cannot read the audit rows.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if tenant is not None:
            validate_tenant_id(tenant)
        query = _SELECT_ROWS
        parameters: tuple[str | int, ...]
        if tenant is None:
            parameters = (limit,)
            query += " ORDER BY seq DESC LIMIT ?"
        else:
            parameters = (tenant, limit)
            query += " WHERE tenant = ? ORDER BY seq DESC LIMIT ?"
        try:
            return [
                audit_row_from_row(row)
                for row in self._connection.execute(query, parameters)
            ]
        except sqlite3.Error as exc:
            raise StoreError("could not list audit rows") from exc
