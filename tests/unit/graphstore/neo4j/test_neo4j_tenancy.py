"""Neo4j tenant schema caching and recovery behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from ontogr_mem.graphstore.indices import DriftReport, SchemaPart
from ontogr_mem.graphstore.neo4j.tenancy import TenantGraphs


class TestTenantGraphs:
    """Behavior of database-wide Neo4j schema provisioning."""

    async def test_reuses_schema_shapes_and_reprovisions_live_drift(self):
        """Tenants share schema DDL while retaining their own requested schema."""
        runner = Mock(run=AsyncMock())
        manager = TenantGraphs(runner)
        manager.schema = SimpleNamespace(
            provision=AsyncMock(),
            verify=AsyncMock(
                side_effect=[DriftReport(), DriftReport(missing=("unique X.id",))]
            ),
        )
        first_part = SchemaPart("first", labels=("X",))
        second_part = SchemaPart("second", labels=("Y",))

        first = await manager.open("tenant-a", [first_part])
        second = await manager.open("tenant-b", [first_part])
        third = await manager.open("tenant-c", [second_part])
        repaired = await manager.open("tenant-d", [first_part])

        first_schema = manager.schema_of(first)
        assert first_schema == manager.schema_of(second)
        assert "Y" in manager.schema_of(third).labels
        assert "X" not in manager.schema_of(third).labels
        assert manager.schema_of(repaired) == first_schema
        assert manager.schema.provision.await_count == 3
        assert runner.run.await_count == 0
