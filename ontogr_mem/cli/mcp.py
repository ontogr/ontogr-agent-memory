"""Run the stdio MCP front for the shared memory server."""

from ontogr_mem.mcp_server.instructions import INSTRUCTIONS
from ontogr_mem.mcp_server.settings import McpSettings


async def run_mcp() -> None:
    """Forward the host's stdio MCP connection to the shared server."""
    from fastmcp import Client  # noqa: PLC0415
    from fastmcp.server.server import create_proxy  # noqa: PLC0415

    from ontogr_mem.relational_store.config import StateSettings  # noqa: PLC0415
    from ontogr_mem.server.entry import ensure_server  # noqa: PLC0415
    from ontogr_mem.service.bootstrap import ServiceSettings  # noqa: PLC0415

    settings = McpSettings()
    graph_url = ServiceSettings().graph_url.get_secret_value()
    state_url = StateSettings().url.get_secret_value()

    endpoint = await ensure_server(graph_url=graph_url, state_url=state_url)
    headers = {"Authorization": f"Bearer {endpoint.token}"} if endpoint.token else {}
    if settings.tenant:
        headers["X-Memory-Tenant"] = settings.tenant

    proxy = create_proxy(
        Client(f"{endpoint.url}/mcp", headers=headers), instructions=INSTRUCTIONS
    )
    await proxy.run_async(transport="stdio", show_banner=False)
