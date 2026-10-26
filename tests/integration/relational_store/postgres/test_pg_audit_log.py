"""The Postgres audit log passes the shared audit contract."""

import pytest

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.relational_store.audit import AuditLog
from ontogr_mem.relational_store.postgres.audit import PostgresAuditLog
from tests.contracts.audit_log import AuditLogContract
from tests.support.clock import FixedClock


class TestPostgresAuditLog(AuditLogContract):
    """Contract behavior over the pooled Postgres adapter."""

    @pytest.fixture
    def clock(self) -> Clock:
        """Return the fixed clock used by the audit log."""
        return FixedClock()

    @pytest.fixture
    def audit_log(self, pool, clean_tables, clock: Clock) -> AuditLog:
        """Return a fresh audit log on the shared database."""
        return PostgresAuditLog(pool, clock)
