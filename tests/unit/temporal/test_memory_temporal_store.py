"""The temporal store contract on the in-memory twin."""

import pytest

from tests.contracts.temporal_store import TemporalStoreContract
from tests.support.temporal import TemporalHarness, memory_temporal_harness


class TestMemoryTemporalStore(TemporalStoreContract):
    """`TemporalStore` contract on the in-memory twin."""

    @pytest.fixture
    def make_harness(self):
        """Return a factory of harnesses on fresh in-memory tenants."""

        async def make() -> TemporalHarness:
            return memory_temporal_harness()

        return make
