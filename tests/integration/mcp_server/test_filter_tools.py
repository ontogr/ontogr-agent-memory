"""Exercise ontology-aware MCP tools through the in-process transport."""

from collections.abc import AsyncGenerator

import pytest

from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from tests.integration.mcp_server.test_server import (
    GROUNDED_FACT,
    _content,
    _write_and_wait,
    open_client,
)
from tests.support.falkordb import falkordb_test_settings
from tests.support.tenants import new_tenant


@pytest.fixture(autouse=True)
async def falkordb_environment(
    monkeypatch: pytest.MonkeyPatch, falkordb_store: FalkorDBGraphStore
) -> AsyncGenerator[None]:
    """Point the service settings at the compose FalkorDB for each test."""
    settings = falkordb_test_settings()
    monkeypatch.setenv("MEMORY_FALKORDB_HOST", settings.host)
    monkeypatch.setenv("MEMORY_FALKORDB_PORT", str(settings.port))
    yield


class TestFilterTools:
    """Expose search filters and ontology reads through MCP."""

    async def test_search_filter_list_entities_and_get_ontology(
        self, falkordb_store: FalkorDBGraphStore
    ) -> None:
        """MCP returns the filtered search, entity page, and ontology shapes."""
        tenant = new_tenant()
        try:
            async with open_client(tenant) as client:
                job = await _write_and_wait(
                    client,
                    "Ana works at Acme. Bob works at Globex.",
                    [
                        GROUNDED_FACT,
                        {
                            **GROUNDED_FACT,
                            "subject": "Bob",
                            "object": "Globex",
                            "evidence_quote": "Bob works at Globex",
                        },
                    ],
                )
                assert job["status"] == "done"

                search = _content(
                    await client.call_tool(
                        "search_memory",
                        {"query": "Where does Ana work?", "entity_types": ["Person"]},
                    )
                )
                assert search["state"] == "found"
                assert any("Acme" in fact["statement"] for fact in search["facts"])
                assert all(
                    "Globex" not in fact["statement"] for fact in search["facts"]
                )

                entities = _content(
                    await client.call_tool("list_entities", {"entity_type": "Person"})
                )
                assert [entity["name"] for entity in entities["entities"]] == ["Ana"]

                ontology = _content(await client.call_tool("get_ontology", {}))
                assert ontology["version"] >= 1
                assert "Person" in [item["label"] for item in ontology["entity_types"]]
        finally:
            await falkordb_store.erase_tenant(tenant)
