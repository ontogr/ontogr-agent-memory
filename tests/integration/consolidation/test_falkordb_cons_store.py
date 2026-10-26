"""The consolidation store contract on FalkorDB, and a locked run against it."""

import pytest

from ontogr_mem.consolidation.config import ConsolidationSettings
from ontogr_mem.consolidation.models import RunStatus
from ontogr_mem.consolidation.runner import ConsolidationRunner
from ontogr_mem.consolidation.stages import default_stages
from ontogr_mem.graphstore.cons_store import GraphConsStore
from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.graphstore.schemas import CONSOLIDATION
from tests.contracts.cons_store import ConsStoreContract
from tests.support.consolidation import ManualClock
from tests.support.tenants import new_tenant


class TestFalkorDBConsStore(ConsStoreContract):
    """`GraphConsStore` satisfies the contract on FalkorDB."""

    @pytest.fixture
    async def make_store(self, falkordb_store: FalkorDBGraphStore):
        """Return a factory of stores on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make() -> GraphConsStore:
            handle = await falkordb_store.open(new_tenant(), [CONSOLIDATION])
            tenants.append(handle.tenant)
            return GraphConsStore(falkordb_store, handle)

        yield make
        for tenant in tenants:
            await falkordb_store.erase_tenant(tenant)

    async def test_a_second_runner_on_the_tenant_is_locked(self, make_store) -> None:
        """A run that finds the lease held is locked and leaves a rerun request."""
        store = await make_store()
        clock = ManualClock()
        assert await store.acquire("other-worker", clock(), 600)
        runner = ConsolidationRunner(
            default_stages(), store, clock, ConsolidationSettings()
        )

        result = await runner.run()

        assert result.status is RunStatus.LOCKED
        assert await store.release("other-worker") is True
