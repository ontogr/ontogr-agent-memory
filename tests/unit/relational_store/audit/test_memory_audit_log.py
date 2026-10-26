"""Run the shared audit-log contract against the in-memory adapter."""

import pytest

from ontogr_mem.relational_store.audit import AuditLog
from ontogr_mem.relational_store.memory.audit import MemoryAuditLog
from tests.contracts.audit_log import AuditLogContract
from tests.support.clock import FixedClock


class TestMemoryAuditLog(AuditLogContract):
    """The in-memory adapter satisfies the audit-log contract."""

    @pytest.fixture
    def clock(self) -> FixedClock:
        """Return the fixed time source for one contract case."""
        return FixedClock()

    @pytest.fixture
    def audit_log(self, clock: FixedClock) -> AuditLog:
        """Return a fresh in-memory audit log."""
        return MemoryAuditLog(clock)
