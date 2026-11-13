"""The MCP server builder, exercised through an in-process client and a fake service."""

import pytest
from fastmcp import Client, FastMCP

from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.mcp_server.server import build_server
from ontogr_mem.mcp_server.settings import ALL_TOOLS, McpSettings
from ontogr_mem.service.contracts import HostFact, MemoryContext, MemoryKind
from tests.support.mcp_memory import FakeMemoryService, build_memory


NO_ADD_TOOLS = [
    "search_memory",
    "get_entity",
    "get_job",
    "write_facts",
    "forget_memory",
    "get_skill",
]

ALL_ENABLED_TOOLS = [
    "search_memory",
    "get_entity",
    "get_job",
    "add_memory",
    "write_facts",
    "forget_memory",
    "get_skill",
]

FORBIDDEN_PROPERTIES = {"self", "principal", "tenant", "grants"}


def _settings(
    tools: tuple[str, ...] = ALL_TOOLS, *, skills_extension: bool = False
) -> McpSettings:
    """Build settings that ignore the environment."""
    return McpSettings(
        _env_file=None,
        tools=tools,
        api_key=None,
        skills_extension=skills_extension,
    )


async def _list_tool_names(server: FastMCP) -> list[str]:
    """Return the tool names a client lists, in the order the server gives them."""
    async with Client(server) as client:
        return [tool.name for tool in await client.list_tools()]


class TestBuildServer:
    """Tool registration, tool schemas and calls through the built server."""

    @pytest.mark.parametrize(
        ("roles", "expected"),
        [
            pytest.param((), NO_ADD_TOOLS, id="no-extraction-model"),
            pytest.param(
                (ModelRole.ENTITY_EXTRACTION,), NO_ADD_TOOLS, id="entity-role-only"
            ),
            pytest.param(
                (ModelRole.ENTITY_EXTRACTION, ModelRole.FACT_EXTRACTION),
                ALL_ENABLED_TOOLS,
                id="both-extraction-roles",
            ),
        ],
    )
    @pytest.mark.parametrize("skills_extension", [False, True])
    async def test_add_memory_needs_both_extraction_models(
        self, roles, expected, skills_extension
    ):
        """add_memory is listed only when both extraction roles have a model."""
        memory = build_memory(FakeMemoryService(), *roles)
        settings = _settings(skills_extension=skills_extension)
        server = build_server(memory, settings=settings)

        names = await _list_tool_names(server)

        assert names == expected

    async def test_settings_cannot_enable_add_memory_without_models(self):
        """Listing add_memory in the settings does not enable it without models."""
        memory = build_memory(FakeMemoryService())
        server = build_server(memory, settings=_settings(("add_memory", "get_job")))

        names = await _list_tool_names(server)

        assert names == ["get_job"]

    async def test_settings_choose_the_listed_tools(self):
        """Only the tools named in the settings are registered."""
        memory = build_memory(FakeMemoryService())
        server = build_server(
            memory, settings=_settings(("search_memory", "get_skill"))
        )

        names = await _list_tool_names(server)

        assert names == ["search_memory", "get_skill"]

    async def test_empty_tools_setting_registers_no_tools(self):
        """An empty tool list leaves the server without tools."""
        memory = build_memory(FakeMemoryService())
        server = build_server(memory, settings=_settings(()))

        names = await _list_tool_names(server)

        assert names == []

    async def test_no_tool_schema_exposes_caller_identity_fields(self):
        """No tool lets a client choose the principal, tenant or grants."""
        memory = build_memory(
            FakeMemoryService(), ModelRole.ENTITY_EXTRACTION, ModelRole.FACT_EXTRACTION
        )
        server = build_server(memory, settings=_settings())

        async with Client(server) as client:
            tools = await client.list_tools()

        assert len(tools) == len(ALL_ENABLED_TOOLS)
        for tool in tools:
            properties = set(tool.input_schema.get("properties", {}))
            assert properties.isdisjoint(FORBIDDEN_PROPERTIES), tool.name

    async def test_add_memory_schema_exposes_content_and_options_only(self):
        """add_memory takes the content and its optional fields and nothing else."""
        memory = build_memory(
            FakeMemoryService(), ModelRole.ENTITY_EXTRACTION, ModelRole.FACT_EXTRACTION
        )
        server = build_server(memory, settings=_settings())

        async with Client(server) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}

        schema = tools["add_memory"].input_schema
        assert set(schema["properties"]) == {
            "content",
            "idempotency_key",
            "reference_time",
            "role",
            "session_id",
            "speaker",
            "thread",
            "wait_seconds",
        }
        assert schema.get("required", []) == ["content"]
        assert set(schema["properties"]).isdisjoint(FORBIDDEN_PROPERTIES)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            (
                "search_memory",
                {"query", "as_of", "include_history", "limit", "budget_tokens"},
            ),
            ("get_entity", {"name", "include_history"}),
            ("get_job", {"job_id"}),
            (
                "write_facts",
                {"source_text", "facts", "role", "idempotency_key", "wait_seconds"},
            ),
            ("forget_memory", {"fact_id", "reason"}),
        ],
    )
    async def test_tool_schema_lists_exactly_the_client_arguments(self, name, expected):
        """Each tool exposes the arguments of its client method and nothing more."""
        memory = build_memory(FakeMemoryService())
        server = build_server(memory, settings=_settings())

        async with Client(server) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}

        assert set(tools[name].input_schema["properties"]) == expected

    async def test_write_facts_sends_a_sourced_facts_request_and_returns_the_receipt(
        self,
    ):
        """write_facts stores the source text and facts and returns the receipt."""
        service = FakeMemoryService()
        server = build_server(build_memory(service), settings=_settings())
        fact = {
            "subject": "Ada",
            "subject_type": "Person",
            "relation": "works_at",
            "object": "Acme",
            "object_type": "Organization",
            "evidence_quote": "Ada works at Acme",
        }

        async with Client(server) as client:
            result = await client.call_tool(
                "write_facts",
                {
                    "source_text": "Ada works at Acme.",
                    "facts": [fact],
                    "idempotency_key": "key-1",
                },
            )

        assert result.structured_content is not None
        assert result.structured_content["job_id"] == "job-1"
        assert result.structured_content["status"] == "queued"
        assert result.structured_content["tenant"] == "acme-tenant"
        request = service.add_requests[0]
        assert request.kind == MemoryKind.SOURCED_FACTS
        assert request.content == "Ada works at Acme."
        assert request.idempotency_key == "key-1"
        assert request.facts == [HostFact(**fact)]

    async def test_search_memory_returns_the_context_state(self):
        """search_memory returns the state and text of the service's context."""
        service = FakeMemoryService()
        service.context = MemoryContext(state="found", rendered="Ada works at Acme.")
        server = build_server(build_memory(service), settings=_settings())

        async with Client(server) as client:
            result = await client.call_tool("search_memory", {"query": "x"})

        assert result.structured_content is not None
        assert result.structured_content["state"] == "found"
        assert result.structured_content["rendered"] == "Ada works at Acme."
        assert service.search_requests[0].query == "x"

    async def test_instructions_point_clients_at_both_skills(self):
        """The server instructions name the two skills a client should read."""
        server = build_server(build_memory(FakeMemoryService()), settings=_settings())

        instructions = server.instructions or ""
        assert "get_skill('memory-write')" in instructions
        assert "get_skill('memory-recall')" in instructions
