"""Run the MCP server for the ontogr-mem command."""

import ipaddress
import logging

from ontogr_mem._extras import import_extra
from ontogr_mem.client import Memory
from ontogr_mem.mcp_server.settings import McpSettings


log = logging.getLogger(__name__)


def _loopback(host: str) -> bool:
    """Tell whether a bind host only accepts connections from this machine.

    Args:
        host: Host name or IP address the server binds to.

    Returns:
        True for "localhost" and loopback IP addresses, otherwise False. A name
        other than "localhost" is not resolved and counts as not loopback.
    """
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def run_mcp(settings: McpSettings) -> None:
    """Serve MCP until shutdown, then finish queued writes and close memory.

    Args:
        settings: Transport, bind address and fixed tenant for this process.
    """
    import_extra("fastmcp", "mcp")
    # FastMCP is an optional extra, so the server module loads only after the
    # check above has given a missing install its hint.
    from ontogr_mem.mcp_server.server import build_server  # noqa: PLC0415

    if (
        settings.transport == "http"
        and not _loopback(settings.host)
        and settings.api_key is None
    ):
        log.warning("MCP HTTP is binding outside loopback without an API key.")
    async with Memory.open(tenant=settings.tenant) as memory:
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
