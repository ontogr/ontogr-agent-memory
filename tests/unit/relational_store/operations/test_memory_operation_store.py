"""The operation contract on the in-memory reference implementation."""

import pytest

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.relational_store import MemoryOperationStore, OperationStore
from tests.contracts.operation_store import OperationStoreContract
from tests.support.clock import FixedClock


class TestMemoryOperationStore(OperationStoreContract):
    """`MemoryOperationStore` satisfies the shared operation contract."""

    @pytest.fixture
    def clock(self) -> Clock:
        """Return the same fixed time for each operation in one contract case."""
        return FixedClock()

    @pytest.fixture
    def operation_store(self, clock: Clock) -> OperationStore:
        """Return an isolated in-memory operation store."""
        return MemoryOperationStore(clock)
