"""Integration tests for the MCP server over a real memory service on FalkorDB."""

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.client import CallToolResult

from ontogr_mem.client import Memory
from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.llm import RoleModels
from ontogr_mem.mcp_server.server import build_server
from ontogr_mem.mcp_server.settings import McpSettings
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.service.bootstrap import ServiceSettings
from tests.support.embedder import MockEmbedder
from tests.support.falkordb import falkordb_test_settings
from tests.support.graph_backend import GraphBackend
from tests.support.scripted_model import (
    ScriptedCall,
    scripted_model,
    scripted_role_models,
)
from tests.support.tenants import new_tenant


SOURCE = "Ana works at Acme."

WITHOUT_ADD_MEMORY = [
    "search_memory",
    "get_entity",
    "get_job",
    "write_facts",
    "forget_memory",
    "get_skill",
]

WITH_ADD_MEMORY = [
    "search_memory",
    "get_entity",
    "get_job",
    "add_memory",
    "write_facts",
    "forget_memory",
    "get_skill",
]

GROUNDED_FACT = {
    "subject": "Ana",
    "subject_type": "Person",
    "relation": "WORKS_AT",
    "object": "Acme",
    "object_type": "Organization",
    "evidence_quote": "Ana works at Acme",
}

UNGROUNDED_FACT = {**GROUNDED_FACT, "evidence_quote": "Ana runs Acme"}

TERMINAL_STATUSES = {"done", "failed"}


@pytest.fixture(autouse=True)
async def falkordb_environment(
    monkeypatch: pytest.MonkeyPatch, falkordb_store: FalkorDBGraphStore
) -> AsyncGenerator[None]:
    """Point the service settings at the compose FalkorDB for each test."""
    settings = falkordb_test_settings()
    monkeypatch.setenv("MEMORY_FALKORDB_HOST", settings.host)
    monkeypatch.setenv("MEMORY_FALKORDB_PORT", str(settings.port))
    yield


@asynccontextmanager
async def open_client(
    tenant: str, models: RoleModels | None = None
) -> AsyncGenerator[Client]:
    """Open an in-process MCP client over a real memory service.

    Args:
        tenant: Tenant the memory client writes to and reads from.
        models: Model pool. Defaults to a pool with no configured model.
    """
    async with Memory.open(
        tenant=tenant,
        service_settings=ServiceSettings(backend="falkordb", _env_file=None),
        state_settings=StateSettings(
            backend="memory", allow_ephemeral=True, _env_file=None
        ),
        models=scripted_role_models() if models is None else models,
        embedder=MockEmbedder(),
    ) as memory:
        settings = McpSettings(_env_file=None, api_key=None, skills_extension=False)
        server = build_server(memory, settings=settings)
        async with Client(server) as client:
            yield client


def _content(result: CallToolResult) -> dict[str, Any]:
    """Return the structured content of a successful tool call."""
    assert result.structured_content is not None
    return result.structured_content


async def _wait_for_job(client: Client, job_id: str) -> dict[str, Any]:
    """Poll get_job until the job is terminal and return the job view.

    Raises:
        AssertionError: If the job is not terminal within 30 seconds.
    """
    for _ in range(300):
        result = await client.call_tool("get_job", {"job_id": job_id})
        job = _content(result)
        if job["status"] in TERMINAL_STATUSES:
            return job
        await asyncio.sleep(0.1)
    raise AssertionError("job did not finish within 30 seconds")


async def _write_and_wait(
    client: Client, source_text: str, facts: list[dict[str, str]]
) -> dict[str, Any]:
    """Write facts through the server and return the finished job view."""
    receipt = await client.call_tool(
        "write_facts", {"source_text": source_text, "facts": facts}
    )
    return await _wait_for_job(client, _content(receipt)["job_id"])


class TestMcpServer:
    """The MCP server writes, recalls and corrects facts on a real service."""

    async def test_keyless_write_then_recall(self, falkordb_store: FalkorDBGraphStore):
        """A grounded fact is recalled, an ungrounded one is dropped and absent."""
        tenant = new_tenant()
        try:
            async with open_client(tenant) as client:
                job = await _write_and_wait(
                    client, SOURCE, [GROUNDED_FACT, UNGROUNDED_FACT]
                )

                assert job["status"] == "done"
                assert len(job["result"]["fact_ids"]) == 1
                assert job["result"]["dropped"] == [
                    {
                        "index": 1,
                        "reason": "ungrounded",
                        "detail": "quote not found in episode",
                    }
                ]

                found = _content(
                    await client.call_tool(
                        "search_memory", {"query": "Where does Ana work?"}
                    )
                )
                assert found["state"] == "found"
                statements = [fact["statement"] for fact in found["facts"]]
                assert any("Acme" in statement for statement in statements)
                assert not any("runs" in statement for statement in statements)

                stores = await GraphBackend(
                    store=falkordb_store, name="falkordb"
                ).stores_for(tenant)
                episode = await stores.episodes.get_episode(job["result"]["episode_id"])
                assert episode is not None
                assert episode.content == SOURCE
                entities = [
                    entity
                    for entity in await stores.resolver_store.active_entities()
                    if entity.canonical_name == "Ana"
                ]
                assert len(entities) == 1
                mentions = await stores.resolver_store.mentions_of(
                    entities[0].entity_id
                )
                assert len(mentions) == 1
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_get_entity_and_forget_memory(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """An entity lists its fact, and forgetting the fact closes it."""
        tenant = new_tenant()
        try:
            async with open_client(tenant) as client:
                job = await _write_and_wait(client, SOURCE, [GROUNDED_FACT])
                assert job["status"] == "done"
                [fact_id] = job["result"]["fact_ids"]

                view = _content(await client.call_tool("get_entity", {"name": "Ana"}))
                assert view["entity"]["name"] == "Ana"
                assert fact_id in [fact["fact_id"] for fact in view["facts"]]

                receipt = _content(
                    await client.call_tool(
                        "forget_memory",
                        {"fact_id": fact_id, "reason": "Left Acme."},
                    )
                )
                forgotten = await _wait_for_job(client, receipt["job_id"])
                assert forgotten["status"] == "done"

                current = _content(
                    await client.call_tool(
                        "search_memory", {"query": "Where does Ana work?"}
                    )
                )
                assert fact_id not in [fact["fact_id"] for fact in current["facts"]]

                historical = _content(
                    await client.call_tool(
                        "search_memory",
                        {"query": "Where does Ana work?", "include_history": True},
                    )
                )
                assert fact_id in [fact["fact_id"] for fact in historical["facts"]]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_all_invalid_and_empty_writes_keep_the_source_episode(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """Writes that store no fact still store their source text."""
        tenant = new_tenant()
        invalid_source = "Ana works at Acme."
        empty_source = "Ben lives in Paris."
        try:
            async with open_client(tenant) as client:
                invalid = await _write_and_wait(
                    client, invalid_source, [UNGROUNDED_FACT]
                )
                empty = await _write_and_wait(client, empty_source, [])

                stores = await GraphBackend(
                    store=falkordb_store, name="falkordb"
                ).stores_for(tenant)
                contents = []
                for job in (invalid, empty):
                    assert job["status"] == "done"
                    assert job["result"]["fact_ids"] == []
                    episode = await stores.episodes.get_episode(
                        job["result"]["episode_id"]
                    )
                    assert episode is not None
                    contents.append(episode.content)
                assert contents == [invalid_source, empty_source]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_keyless_write_never_calls_a_model_even_when_models_are_configured(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A fact write finishes without reaching any configured model."""
        tenant = new_tenant()

        def refuse(call: ScriptedCall) -> object:
            """Fail the write if it reaches a model."""
            raise AssertionError("keyless write called a model")

        model, recorded = scripted_model(answer=refuse)
        try:
            async with open_client(tenant, scripted_role_models(model)) as client:
                names = [tool.name for tool in await client.list_tools()]
                assert names == WITH_ADD_MEMORY

                job = await _write_and_wait(client, SOURCE, [GROUNDED_FACT])

                assert job["status"] == "done"
                assert len(job["result"]["fact_ids"]) == 1
                assert recorded.calls == []
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_add_memory_is_absent_without_extraction_models_and_refused_if_forced(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """Without extraction models add_memory is not listed and a call fails."""
        tenant = new_tenant()
        try:
            async with open_client(tenant) as client:
                names = [tool.name for tool in await client.list_tools()]
                assert names == WITHOUT_ADD_MEMORY

                result = await client.call_tool(
                    "add_memory", {"text": SOURCE}, raise_on_error=False
                )
                assert result.is_error
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_error_results_carry_only_code_message_and_hint(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """An unknown job id gives a typed error with exactly three keys."""
        tenant = new_tenant()
        try:
            async with open_client(tenant) as client:
                result = await client.call_tool(
                    "get_job", {"job_id": "no-such-job"}, raise_on_error=False
                )

                assert result.is_error
                error = json.loads("".join(block.text for block in result.content))
                assert sorted(error) == ["code", "hint", "message"]
                assert error["code"] == "not_found"
        finally:
            await falkordb_store.erase_tenant(tenant)
