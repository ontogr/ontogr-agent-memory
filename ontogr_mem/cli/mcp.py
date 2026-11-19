"""Run the MCP server for the ontogr-mem command."""

import logging

from ontogr_mem._extras import import_extra
from ontogr_mem.client import Memory
from ontogr_mem.mcp_server.settings import McpSettings
from ontogr_mem.service.bootstrap import ServiceSettings
from ontogr_mem.store_urls import is_loopback_host, require_explicit_tenant


log = logging.getLogger(__name__)


async def run_mcp(settings: McpSettings) -> None:
    """Serve MCP until shutdown, then finish queued writes and close memory.

    Args:
        settings: Transport, bind address and fixed tenant for this process.
    """
    import_extra("fastmcp", "mcp")
    # FastMCP is an optional extra, so the server module loads only after the
    # check above has given a missing install its hint.
    from ontogr_mem.mcp_server.server import build_server  # noqa: PLC0415

    require_explicit_tenant(
        ServiceSettings().graph_url,
        tenant_given="tenant" in settings.model_fields_set,
        hint="set MEMORY_MCP_TENANT or pass --tenant",
    )

    if (
        settings.transport == "http"
        and not is_loopback_host(settings.host)
        and settings.api_key is None
    ):
        log.warning("MCP HTTP is binding outside loopback without an API key.")
    async with Memory.open(
        tenant=settings.tenant, ontology=settings.ontology
    ) as memory:
        server = build_server(memory, settings=settings)
        try:
            if settings.transport == "stdio":
                await server.run_async(transport="stdio", show_banner=False)
            else:
                await server.run_async(
                    transport="http",
                    host=settings.host,
                    port=settings.port,
                    show_banner=False,
                )
        finally:
            await memory.drain()
