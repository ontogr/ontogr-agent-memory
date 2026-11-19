"""End-to-end checks of add_memory on a fresh tenant with the default ontology."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.client import CallToolResult

from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.llm import RoleModels
from ontogr_mem.mcp_server.server import build_server
from ontogr_mem.mcp_server.settings import McpSettings
from tests.support.falkordb import falkordb_test_url
from tests.support.tenants import new_tenant
from tests.support.text_writes import (
    ScriptedEntity,
    ScriptedFact,
    open_memory,
    text_models,
)


STORED = "Ana works at Acme."
UNPLACED = "Ana mentors Ben."
SCRIPT = {
    STORED: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Acme", "Organization")],
        [ScriptedFact("Ana works at Acme", 0, 1, "WORKS_AT", "Ana works at Acme")],
    ),
    UNPLACED: (
        [ScriptedEntity("Ana", "Person"), ScriptedEntity("Ben", "Person")],
        [ScriptedFact("Ana mentors Ben", 0, 1, "mentors", "Ana mentors Ben")],
    ),
}
TERMINAL_STATUSES = {"done", "failed"}


@pytest.fixture(params=["two_pass", "combined"], autouse=True)
def extraction_mode(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Select the extraction structure for each test through the environment."""
    monkeypatch.setenv("MEMORY_EXTRACTION_MODE", request.param)
    return str(request.param)


@pytest.fixture(autouse=True)
async def falkordb_environment(
    monkeypatch: pytest.MonkeyPatch, falkordb_store: FalkorDBGraphStore
) -> AsyncGenerator[None]:
    """Point the service settings at the compose FalkorDB for each test."""
    monkeypatch.setenv("MEMORY_GRAPH_URL", falkordb_test_url())
    yield


@asynccontextmanager
async def open_client(tenant: str, models: RoleModels) -> AsyncGenerator[Client]:
    """Open an in-process MCP client over a real memory service."""
    async with open_memory(tenant, models) as memory:
        settings = McpSettings(_env_file=None, api_key=None, skills_extension=False)
        async with Client(
            build_server(lambda request_tenant: memory, settings=settings)
        ) as client:
            yield client


def _content(result: CallToolResult) -> dict[str, Any]:
    """Return the structured content of a successful tool call."""
    assert result.structured_content is not None
    return result.structured_content


async def _wait_for_job(client: Client, job_id: str) -> dict[str, Any]:
    """Poll get_job until the job is terminal and return the job view."""
    for _ in range(300):
        job = _content(await client.call_tool("get_job", {"job_id": job_id}))
        if job["status"] in TERMINAL_STATUSES:
            return job
        await asyncio.sleep(0.1)
    raise AssertionError("job did not finish within 30 seconds")


class TestDefaultTenantMcpWrites:
    """MCP text writes on a fresh tenant use the default ontology."""

    async def test_add_memory_stores_a_default_relation_fact(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """add_memory stores and search_memory finds a default relation fact."""
        tenant = new_tenant()
        try:
            async with open_client(tenant, text_models(SCRIPT)) as client:
                receipt = await client.call_tool("add_memory", {"content": STORED})
                job = await _wait_for_job(client, _content(receipt)["job_id"])
                assert job["status"] == "done"
                assert len(job["result"]["fact_ids"]) == 1
                found = _content(
                    await client.call_tool(
                        "search_memory", {"query": "Where does Ana work?"}
                    )
                )
                assert [fact["statement"] for fact in found["facts"]] == [
                    "Ana works at Acme"
                ]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_add_memory_keeps_an_unknown_relation_as_an_unplaced_fact(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """add_memory stores an unknown relation with its text label."""
        tenant = new_tenant()
        try:
            async with open_client(tenant, text_models(SCRIPT)) as client:
                receipt = await client.call_tool("add_memory", {"content": UNPLACED})
                job = await _wait_for_job(client, _content(receipt)["job_id"])
                assert job["status"] == "done"
                assert job["result"]["unplaced"] == 1
                assert "ontology_refused" not in job["result"]
                assert len(job["result"]["fact_ids"]) == 1
                found = _content(
                    await client.call_tool(
                        "search_memory", {"query": "Who does Ana mentor?"}
                    )
                )
                assert len(found["facts"]) == 1
                assert found["facts"][0]["statement"] == "Ana mentors Ben"
                assert found["facts"][0]["relation"] == "mentors"
                assert found["facts"][0]["kind"] == "state"
                assert found["facts"][0]["modality"] == "actual"
        finally:
            await falkordb_store.erase_tenant(tenant)
