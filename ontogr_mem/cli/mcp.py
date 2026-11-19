"""Run the MCP server for the ontogr-mem command."""

import logging
import sys

from ontogr_mem.client import Memory
from ontogr_mem.llm import build_models
from ontogr_mem.mcp_server.settings import McpSettings
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.requirements import check_requirements
from ontogr_mem.service.bootstrap import ServiceSettings
from ontogr_mem.store_urls import (
    is_loopback_host,
    parse_graph_url,
    parse_state_url,
    require_explicit_tenant,
)
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings


log = logging.getLogger(__name__)


async def run_mcp(settings: McpSettings) -> None:
    """Serve MCP until shutdown, then finish queued writes and close memory.

    Args:
        settings: Transport, bind address and fixed tenant for this process.
    """
    require_explicit_tenant(
        ServiceSettings().graph_url,
        tenant_given="tenant" in settings.model_fields_set,
        hint="set MEMORY_MCP_TENANT or pass --tenant",
    )
    models = build_models()
    try:
        service = ServiceSettings()
        state = StateSettings()
        report = check_requirements(
            models=models,
            graph=parse_graph_url(service.graph_url.get_secret_value()),
            state=parse_state_url(state.url.get_secret_value()),
            embedding=EmbeddingSettings(),
            mcp=True,
        )
        report.raise_for_missing(subcommand="mcp")
        mode_message = "memory mode: model"
        if report.mode == "keyless":
            mode_message = (
                "memory mode: keyless, add_memory is not offered because no model "
                "is configured for entity_extraction and fact_extraction"
            )
        print(mode_message, file=sys.stderr)
        from ontogr_mem.mcp_server.server import build_server  # noqa: PLC0415

        async with Memory.open(
            tenant=settings.tenant, ontology=settings.ontology, models=models
        ) as memory:
            server = build_server(memory, settings=settings)
            try:
                if (
                    settings.transport == "http"
                    and not is_loopback_host(settings.host)
                    and settings.api_key is None
                ):
                    log.warning(
                        "MCP HTTP is binding outside loopback without an API key."
                    )
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
    finally:
        await models.aclose()
