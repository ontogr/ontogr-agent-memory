"""In-memory audit log implementation."""

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.relational_store.audit.chain import (
    GENESIS_HASH,
    chain_hash,
    first_broken_seq,
)
from ontogr_mem.relational_store.audit.models import AuditEntry, AuditRow


class MemoryAuditLog:
    """Keep an audit chain in process memory."""

    def __init__(self, clock: Clock) -> None:
        """Create an empty log using the supplied time source."""
        self._clock = clock
        self._rows: list[AuditRow] = []

    async def append(self, entry: AuditEntry) -> AuditRow:
        """Append an event to the chain and return its immutable row.

        Raises:
            ValueError: If the clock returns a naive datetime.
        """
        previous_hash = self._rows[-1].row_hash if self._rows else GENESIS_HASH
        recorded_at = self._clock.now()
        row = AuditRow(
            seq=len(self._rows) + 1,
            recorded_at=recorded_at,
            entry=entry,
            previous_hash=previous_hash,
            row_hash=chain_hash(previous_hash, recorded_at, entry),
        )
        self._rows.append(row)
        return row

    async def verify(self) -> int | None:
        """Return the first broken row, or None when the chain is intact.

        This detects edits and deletions in the middle. It cannot detect a
        database owner who rewrites later hashes or deletes the final row.
        """
        return first_broken_seq(self._rows)

    async def list_rows(
        self, *, tenant: str | None = None, limit: int = 100
    ) -> list[AuditRow]:
        """Return newest rows first, optionally filtered to one Tenant.

        Raises:
            ValueError: If ``limit`` is less than one.
            InvalidTenantError: If ``tenant`` is not a valid Tenant id.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if tenant is not None:
            validate_tenant_id(tenant)
        return [
            row
            for row in reversed(self._rows)
            if tenant is None or row.entry.tenant == tenant
        ][:limit]
