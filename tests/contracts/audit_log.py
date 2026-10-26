"""Shared behavior required of in-memory and SQLite audit logs."""

import asyncio

import pytest

from ontogr_mem.data_models.tenant import InvalidTenantError
from ontogr_mem.relational_store.audit import (
    GENESIS_HASH,
    AuditAction,
    AuditEntry,
    AuditLog,
    AuditOutcome,
)


def make_entry(
    *,
    action: AuditAction = AuditAction.AUTHORIZE,
    outcome: AuditOutcome = AuditOutcome.ALLOWED,
    principal_id: str = "user-1",
    tenant: str | None = "acme",
    target: str = "search",
    detail: dict[str, str | int | bool | None] | None = None,
) -> AuditEntry:
    """Build a valid audit entry with values stable across adapters."""
    return AuditEntry(
        action=action,
        outcome=outcome,
        principal_id=principal_id,
        tenant=tenant,
        target=target,
        trace_id="trace-1",
        detail=detail if detail is not None else {},
    )


class AuditLogContract:
    """Audit behavior shared by every adapter."""

    @pytest.fixture
    def audit_log(self) -> AuditLog:
        """Return a fresh audit log for one contract case."""
        raise NotImplementedError

    @pytest.fixture
    def clock(self):
        """Return the fixed clock used by the audit log."""
        raise NotImplementedError

    async def test_empty_log_has_no_rows_and_verifies(self, audit_log: AuditLog):
        """A newly created log is an intact empty chain."""
        assert await audit_log.verify() is None
        assert await audit_log.list_rows() == []

    async def test_first_row_links_to_genesis_and_uses_the_clock(
        self, audit_log: AuditLog, clock
    ):
        """The first row starts at genesis and records the injected time."""
        row = await audit_log.append(make_entry())

        assert row.seq == 1
        assert row.previous_hash == GENESIS_HASH
        assert row.recorded_at == clock.now()

    async def test_rows_link_in_sequence_order(self, audit_log: AuditLog):
        """Each row points to the preceding row and increments its sequence."""
        appended = [
            await audit_log.append(make_entry(target=f"target-{index}"))
            for index in range(1, 4)
        ]

        assert [row.seq for row in appended] == [1, 2, 3]
        assert [row.previous_hash for row in appended[1:]] == [
            row.row_hash for row in appended[:-1]
        ]

    async def test_many_appends_keep_the_chain_intact(self, audit_log: AuditLog):
        """A chain with many rows verifies without a broken sequence."""
        for index in range(20):
            await audit_log.append(make_entry(target=f"target-{index}"))

        assert await audit_log.verify() is None

    async def test_detail_values_round_trip(self, audit_log: AuditLog):
        """String, integer, boolean, and null detail values are preserved."""
        detail = {"text": "value", "count": 3, "enabled": True, "missing": None}

        row = await audit_log.append(make_entry(detail=detail))

        assert row.entry.detail == detail

    async def test_list_rows_is_newest_first_and_respects_limit(
        self, audit_log: AuditLog
    ):
        """List order is newest first, and the limit must be positive."""
        for index in range(1, 4):
            await audit_log.append(make_entry(target=f"target-{index}"))

        assert [row.seq for row in await audit_log.list_rows()] == [3, 2, 1]
        assert [row.seq for row in await audit_log.list_rows(limit=2)] == [3, 2]
        with pytest.raises(ValueError):
            await audit_log.list_rows(limit=0)

    async def test_list_rows_filters_by_tenant(self, audit_log: AuditLog):
        """Tenant filters exclude other tenants and tenantless rows."""
        await audit_log.append(make_entry(tenant="acme"))
        await audit_log.append(make_entry(tenant="beta"))
        await audit_log.append(make_entry(tenant=None))

        assert [
            row.entry.tenant for row in await audit_log.list_rows(tenant="acme")
        ] == ["acme"]
        assert [row.entry.tenant for row in await audit_log.list_rows()] == [
            None,
            "beta",
            "acme",
        ]

    async def test_invalid_tenant_is_rejected_without_writing(
        self, audit_log: AuditLog
    ):
        """Invalid tenant ids are refused before a row reaches the log."""
        with pytest.raises(InvalidTenantError):
            make_entry(tenant="not a tenant")

        assert await audit_log.list_rows() == []

    async def test_empty_principal_is_rejected(self, audit_log: AuditLog):
        """An audit entry needs a principal id."""
        with pytest.raises(ValueError):
            make_entry(principal_id="")

        assert await audit_log.list_rows() == []

    async def test_concurrent_appends_have_no_gaps_or_forks(self, audit_log: AuditLog):
        """Concurrent callers append every row to one intact sequence."""
        await asyncio.gather(
            *(
                audit_log.append(make_entry(target=f"target-{index}"))
                for index in range(50)
            )
        )

        rows = await audit_log.list_rows(limit=50)
        assert await audit_log.verify() is None
        assert sorted(row.seq for row in rows) == list(range(1, 51))

    async def test_export_action_round_trips(self, audit_log: AuditLog):
        """The export event retains its action and outcome."""
        row = await audit_log.append(
            make_entry(action=AuditAction.EXPORT, outcome=AuditOutcome.OK)
        )

        assert row.entry.action is AuditAction.EXPORT
        assert row.entry.outcome is AuditOutcome.OK
