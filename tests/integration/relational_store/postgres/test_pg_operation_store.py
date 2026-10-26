"""The Postgres operation store passes the shared operation contract."""

import pytest

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.relational_store import OperationStore
from ontogr_mem.relational_store.postgres.operations import (
    PostgresOperationStore,
)
from tests.contracts.operation_store import OperationStoreContract
from tests.support.clock import FixedClock


class TestPostgresOperationStore(OperationStoreContract):
    """Contract behavior over the pooled Postgres adapter."""

    @pytest.fixture
    def clock(self) -> Clock:
        """Return a fixed clock for exact timestamp assertions."""
        return FixedClock()

    @pytest.fixture
    async def operation_store(self, pool, clean_tables, clock: Clock) -> OperationStore:
        """Return an isolated operation store with interrupted runs marked."""
        store = PostgresOperationStore(pool, clock)
        await store.mark_interrupted_runs()
        return store
