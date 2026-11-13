"""Build the MCP server that exposes one Memory client as tools and skills."""

from fastmcp import FastMCP
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from fastmcp.skills.extension import SkillsExtension

from ontogr_mem.client import Memory
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.mcp_server.auth import ApiKeyVerifier
from ontogr_mem.mcp_server.instructions import INSTRUCTIONS
from ontogr_mem.mcp_server.settings import McpSettings
from ontogr_mem.mcp_server.skills import SKILLS_ROOT, get_skill
from ontogr_mem.mcp_server.tools import guard_errors


def build_server(memory: Memory, *, settings: McpSettings | None = None) -> FastMCP:
    """Build a configured MCP door over an already-open memory client.

    Args:
        memory: Client whose caller and tenant are fixed for this process.
        settings: MCP settings, or settings loaded from the environment.

    Returns:
        An unstarted server. The caller owns the client's lifetime.
    """
    server_settings = settings if settings is not None else McpSettings()
    auth = (
        ApiKeyVerifier(server_settings.api_key.get_secret_value())
        if server_settings.api_key is not None
        else None
    )
    server = FastMCP(
        "ontogr-memory",
        instructions=INSTRUCTIONS,
        mask_error_details=True,
        auth=auth,
    )
    enabled = set(server_settings.tools)
    if not (
        memory.models.configured(ModelRole.ENTITY_EXTRACTION)
        and memory.models.configured(ModelRole.FACT_EXTRACTION)
    ):
        enabled.discard("add_memory")
    if "search_memory" in enabled:
        server.tool(guard_errors(memory.search), name="search_memory", tags={"read"})
    if "get_entity" in enabled:
        server.tool(guard_errors(memory.get_entity), name="get_entity", tags={"read"})
    if "get_job" in enabled:
        server.tool(guard_errors(memory.get_job), name="get_job", tags={"read"})
    if "add_memory" in enabled:
        server.tool(guard_errors(memory.write_text), name="add_memory", tags={"write"})
    if "write_facts" in enabled:
        server.tool(
            guard_errors(memory.write_facts), name="write_facts", tags={"write"}
        )
    if "forget_memory" in enabled:
        server.tool(
            guard_errors(memory.invalidate), name="forget_memory", tags={"forget"}
        )
    if "get_skill" in enabled:
        server.tool(guard_errors(get_skill), name="get_skill", tags={"skills"})
    if server_settings.skills_extension:
        provider = SkillsDirectoryProvider(SKILLS_ROOT, experimental_skills=True)
        server.add_provider(provider)
        server.add_extension(SkillsExtension(providers=[provider]))
    return server
