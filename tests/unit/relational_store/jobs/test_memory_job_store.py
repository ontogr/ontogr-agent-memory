"""Run the shared storage contract without a database or external services."""

import pytest

from ontogr_mem.relational_store.memory import MemoryJobStore
from tests.contracts.erase_journal import EraseJournalContract
from tests.contracts.job_store import JobStoreContract


class TestMemoryJobStore(JobStoreContract):
    """The independent memory adapter satisfies the shared job contract."""

    @pytest.fixture
    def make_store(self):
        """Return a fresh in-memory store for each factory call."""

        async def make():
            return MemoryJobStore()

        return make


class TestMemoryEraseJournal(EraseJournalContract):
    """The memory state adapter persists erase lifecycle in its own process."""

    @pytest.fixture
    def make_store(self):
        """Return a fresh in-memory state store."""

        async def make():
            return MemoryJobStore()

        return make
