"""Check caller grants and persist authorization decisions in a SQLite chain."""

import sqlite3
from contextlib import closing

import pytest

from ontogr_mem.data_models.tenant import InvalidTenantError
from ontogr_mem.relational_store.audit import AuditAction, AuditOutcome
from ontogr_mem.relational_store.sqlite.audit import SqliteAuditLog
from ontogr_mem.service.access.audit_sink import ChainedAuditSink
from ontogr_mem.service.access.authz import Authorizer
from ontogr_mem.service.errors import ObjectNotFoundError
from ontogr_mem.service.principal import ALL_TENANTS, Grant, Principal
from tests.support.clock import FixedClock


class TestAccessFlow:
    """Keep grant enforcement and its durable audit evidence independent of keys."""

    @pytest.mark.parametrize("scope", ["acme", ALL_TENANTS])
    @pytest.mark.parametrize(
        ("granted", "decisions"),
        [
            (Grant.READ, "ADDDDD"),
            (Grant.WRITE, "DADDDD"),
            (Grant.FORGET, "DDADDD"),
            (Grant.SHARE, "DDDADD"),
            (Grant.EXPORT, "DDDDAD"),
            (Grant.ADMIN, "AAAAAA"),
        ],
    )
    async def test_principal_grants_control_access_and_are_audited(
        self, tmp_path, scope, granted, decisions
    ):
        """Scoped, wildcard, and admin grants retain their authorization rules."""
        database_path = tmp_path / "authorization.db"
        clock = FixedClock()
        principal = Principal(id="reader", grants={scope: frozenset({granted})})
        expected = []
        with closing(
            sqlite3.connect(database_path, isolation_level=None)
        ) as connection:
            audit = SqliteAuditLog(connection, clock)
            authorizer = Authorizer(ChainedAuditSink(audit), clock)
            errors = []
            for tenant in ("acme", "beta"):
                tenant_decisions = (
                    decisions if tenant == "acme" or scope == ALL_TENANTS else "DDDDDD"
                )
                for required, decision in zip(Grant, tenant_decisions, strict=True):
                    if decision == "A":
                        assert (
                            await authorizer.require(
                                principal, tenant, required, "search", target="memory-1"
                            )
                            == tenant
                        )
                    else:
                        with pytest.raises(ObjectNotFoundError) as denied:
                            await authorizer.require(
                                principal, tenant, required, "search", target="memory-1"
                            )
                        errors.append((str(denied.value), denied.value.hint))
                    expected.append(
                        (
                            tenant,
                            AuditOutcome.ALLOWED
                            if decision == "A"
                            else AuditOutcome.DENIED,
                        )
                    )
            assert len(set(errors)) <= 1
            if errors:
                assert errors[0] == (
                    "tenant or object not found",
                    "Check the tenant id and whether this principal has the required "
                    "grant (read, write, forget, share, export, or admin).",
                )
            for tenant in (ALL_TENANTS, "bad id"):
                with pytest.raises(InvalidTenantError):
                    await authorizer.require(principal, tenant, Grant.READ, "search")
            rows = await audit.list_rows()
            assert [(row.entry.tenant, row.entry.outcome) for row in rows] == list(
                reversed(expected)
            )
            assert all(
                row.entry.action == AuditAction.AUTHORIZE
                and row.entry.principal_id == "reader"
                and row.entry.target == "memory-1"
                and row.entry.detail == {"event_action": "search"}
                for row in rows
            )
            assert await audit.verify() is None
        with closing(
            sqlite3.connect(database_path, isolation_level=None)
        ) as connection:
            reopened = SqliteAuditLog(connection, clock)
            assert len(await reopened.list_rows()) == 12
            assert await reopened.verify() is None
