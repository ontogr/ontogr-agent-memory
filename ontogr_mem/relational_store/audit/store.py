"""The asynchronous port for append-only audit logs."""

from typing import Protocol

from ontogr_mem.relational_store.audit.models import AuditEntry, AuditRow


class AuditLog(Protocol):
    """Append and read audit rows linked by a verifiable hash chain.

    Verification detects edits and deletions in the middle of the chain. It
    cannot detect a database owner who rewrites later hashes or deletes the
    final row. Access events use a different model and need an adapter.
    """

    async def append(self, entry: AuditEntry) -> AuditRow:
        """Append one event and return its linked row."""
        ...

    async def verify(self) -> int | None:
        """Return the first broken sequence, or None when the chain verifies.

        This detects edits and deletions in the middle. It cannot detect a
        database owner who rewrites later hashes or deletes the final row.
        """
        ...

    async def list_rows(
        self, *, tenant: str | None = None, limit: int = 100
    ) -> list[AuditRow]:
        """Return newest rows first, optionally filtered to one Tenant."""
        ...
