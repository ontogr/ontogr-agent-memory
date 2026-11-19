"""Tenant discovery filtering at each backend boundary."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from ontogr_mem.graphstore.falkordb.tenancy import TenantGraphs as FalkorTenantGraphs
from ontogr_mem.graphstore.neo4j.tenancy import TenantGraphs as Neo4jTenantGraphs


async def test_falkordb_lists_only_valid_tenant_graphs(monkeypatch):
    """Ignore non-tenant and malformed Falkor graph names."""
    pool = Mock(
        list_graphs=AsyncMock(
            return_value=["m_tenant-b", "other", "m_bad!", "m_tenant-a", "m_tenant-a"]
        )
    )
    manager = FalkorTenantGraphs(pool)

    assert await manager.list_tenants() == ["tenant-a", "tenant-b"]
    pool.list_graphs.assert_awaited_once_with()


async def test_neo4j_skips_invalid_labels_and_tenant_values():
    """Ignore unsafe labels and malformed tenant values from Neo4j."""
    runner = Mock()
    runner.run = AsyncMock(
        side_effect=[
            SimpleNamespace(
                rows=[{"label": "Person"}, {"label": "`invalid"}, {"label": None}]
            ),
            SimpleNamespace(
                rows=[
                    {"tenant": "tenant-b"},
                    {"tenant": ""},
                    {"tenant": None},
                    {"tenant": 3},
                    {"tenant": "tenant-a"},
                ]
            ),
        ]
    )
    manager = Neo4jTenantGraphs(runner)

    assert await manager.list_tenants() == ["tenant-a", "tenant-b"]
    assert runner.run.await_count == 2
