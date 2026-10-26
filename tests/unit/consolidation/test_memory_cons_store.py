"""The consolidation store contract on the in-memory twin."""

import pytest

from ontogr_mem.graphstore.memory import MemoryConsStore
from tests.contracts.cons_store import ConsStoreContract


class TestMemoryConsStore(ConsStoreContract):
    """`MemoryConsStore` satisfies the consolidation store contract."""

    @pytest.fixture
    def make_store(self):
        """Return a factory of stores on fresh in-memory tenants."""

        async def make() -> MemoryConsStore:
            return MemoryConsStore()

        return make
