"""Build the MCP server that exposes memory tools for request-selected tenants."""

from collections.abc import Callable
from inspect import signature
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from fastmcp.skills.extension import SkillsExtension

from ontogr_mem.client import Memory
from ontogr_mem.mcp_server.auth import ApiKeyVerifier
from ontogr_mem.mcp_server.instructions import INSTRUCTIONS
from ontogr_mem.mcp_server.settings import McpSettings
from ontogr_mem.mcp_server.skills import SKILLS_ROOT, get_skill
from ontogr_mem.mcp_server.tools import guard_errors
from ontogr_mem.requirements import TEXT_WRITE_ROLES


def build_server(
    memory_for_request: Callable[[str | None], Memory],
    *,
    settings: McpSettings | None = None,
) -> FastMCP:
    """Build an MCP server whose tools resolve a Memory view per request."""
    server_settings = settings if settings is not None else McpSettings()
    auth = (
        ApiKeyVerifier(server_settings.api_key.get_secret_value())
        if server_settings.api_key is not None
        else None
    )
    server = FastMCP("ontogr-memory", instructions=INSTRUCTIONS,
                     mask_error_details=True, auth=auth)
    enabled = set(server_settings.tools)

    def register(name: str, method: str, tags: set[str]) -> None:
        memory = memory_for_request(None)

        async def call(**kwargs: Any) -> Any:
            from fastmcp.server.dependencies import get_http_headers  # noqa: PLC0415

            headers = get_http_headers()
            tenant = headers.get("x-memory-tenant")
            memory = memory_for_request(tenant)
            return await getattr(memory, method)(**kwargs)

        memory_method = getattr(memory, method)
        call.__signature__ = signature(memory_method)
        call.__annotations__ = memory_method.__annotations__.copy()
        server.tool(guard_errors(call), name=name, tags=tags)

    if "add_memory" in enabled:
        enabled.discard("add_memory")
        # This server accepts extraction only when the fixed process model pool
        # supports every required role; request-specific Memory views share it.
        memory = memory_for_request(None)
        if all(memory.models.configured(role) for role in TEXT_WRITE_ROLES):
            enabled.add("add_memory")
    methods = {
        "search_memory": ("search", {"read"}),
        "get_entity": ("get_entity", {"read"}),
        "list_entities": ("list_entities", {"read"}),
        "get_ontology": ("get_ontology", {"read"}),
        "stale_episodes": ("stale_episodes", {"read"}),
        "get_job": ("get_job", {"read"}),
        "add_memory": ("write_text", {"write"}),
        "write_facts": ("write_facts", {"write"}),
        "forget_memory": ("invalidate", {"forget"}),
    }
    for name, (method, tags) in methods.items():
        if name in enabled:
            register(name, method, tags)
    if "get_skill" in enabled:
        server.tool(guard_errors(get_skill), name="get_skill", tags={"skills"})
    if server_settings.skills_extension:
        provider = SkillsDirectoryProvider(SKILLS_ROOT, experimental_skills=True)
        server.add_provider(provider)
        server.add_extension(SkillsExtension(providers=[provider]))
    return server
