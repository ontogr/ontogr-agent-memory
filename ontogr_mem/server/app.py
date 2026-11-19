"""Run the shared HTTP memory server."""

import asyncio
import hmac
import ipaddress
import logging
import os
import secrets
import socket
import time
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from ontogr_mem.client import Memory
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.llm import build_models
from ontogr_mem.mcp_server.server import build_server
from ontogr_mem.mcp_server.settings import McpSettings
from ontogr_mem.ontology.loader import load_ontology
from ontogr_mem.paths import data_directory, ensure_directory
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.requirements import TEXT_WRITE_ROLES
from ontogr_mem.server.files import (
    read_server_info,
    write_server_info,
    write_server_token,
)
from ontogr_mem.server.lock import ServerLock
from ontogr_mem.server.settings import ServerSettings
from ontogr_mem.service.access.settings import StdioSettings
from ontogr_mem.service.bootstrap import ServiceSettings, open_service
from ontogr_mem.store_urls import SqliteTarget, parse_state_url
from ontogr_mem.vectorstore.builtin import (
    EndpointEmbedder,
    LocalEmbedder,
    select_embedder,
)
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings
from ontogr_mem.vectorstore.embedder import Embedder
from ontogr_mem.vectorstore.source import EmbedderSource


logger = logging.getLogger(__name__)


def _health_ready(host: str, port: int) -> None:
    """Return when the server's HTTP listener serves its health route."""
    with urlopen(f"http://{host}:{port}/health", timeout=0.5):
        return


def _loopback(host: str) -> bool:
    """Return whether a bind host resolves exclusively to loopback addresses."""
    if host.lower() == "localhost":
        return True
    try:
        return all(
            ipaddress.ip_address(result[4][0]).is_loopback
            for result in socket.getaddrinfo(host, None)
        )
    except (OSError, ValueError):
        return False


async def serve(  # noqa: C901, PLR0912, PLR0915
    *, mcp_settings: McpSettings, server_settings: ServerSettings
) -> int:
    """Serve the shared MCP endpoint and own its state-store locks."""
    from fastmcp.server.dependencies import get_http_headers  # noqa: PLC0415
    from starlette.requests import Request  # noqa: PLC0415
    from starlette.responses import JSONResponse, Response  # noqa: PLC0415

    directory = data_directory()
    ensure_directory(directory)
    server_lock = ServerLock(directory / "server.lock")
    try:
        server_lock.acquire()
    except OSError:
        return 75
    store_lock: ServerLock | None = None
    info_path = directory / "server.json"
    process_id = os.getpid()
    try:
        service_settings = ServiceSettings()
        state_settings = StateSettings()
        state_target = parse_state_url(state_settings.url.get_secret_value())
        if isinstance(state_target, SqliteTarget):
            store_lock = ServerLock(Path(f"{state_target.path}.lock"))
            try:
                store_lock.acquire()
            except OSError:
                return 75
        models = build_models()
        ontology = (
            load_ontology(mcp_settings.ontology) if mcp_settings.ontology else None
        )
        embedding_settings = EmbeddingSettings()
        async with AsyncExitStack() as resources:
            resources.push_async_callback(models.aclose)

            async def load_embedder() -> Embedder:
                embedder = await select_embedder(embedding_settings)
                if isinstance(embedder, (EndpointEmbedder, LocalEmbedder)):
                    resources.push_async_callback(embedder.aclose)
                return embedder

            embedder_source = EmbedderSource(load_embedder)
            runtime = await resources.enter_async_context(
                open_service(
                    service_settings,
                    state_settings=state_settings,
                    models=models,
                    embedder=embedder_source,
                    ontology=ontology,
                    ontology_tenant=mcp_settings.tenant or "default",
                )
            )
            views: dict[str, Memory] = {}
            fixed_tenant = validate_tenant_id(mcp_settings.tenant or "default")
            loopback = _loopback(mcp_settings.host)

            def memory_for_request(requested: str | None) -> Memory:
                tenant = validate_tenant_id(requested or fixed_tenant)
                if not loopback and tenant != fixed_tenant:
                    from ontogr_mem.service.errors import (  # noqa: PLC0415
                        InvalidArgumentError,
                    )

                    raise InvalidArgumentError(
                        f"This server is fixed to tenant {fixed_tenant}",
                        hint=f"Use MEMORY_MCP_TENANT={fixed_tenant}",
                    )
                if tenant not in views:
                    views[tenant] = Memory(
                        runtime.service, tenant=tenant, models=models
                    )
                return views[tenant]

            token = (
                mcp_settings.api_key.get_secret_value()
                if mcp_settings.api_key
                else None
            )
            if token is None and loopback:
                token = secrets.token_urlsafe(32)
                write_server_token(directory / "server.token", token)
            if not loopback and token is None:
                logger.warning("MCP server is bound beyond loopback without an API key")
            settings = mcp_settings
            # A generated token must also protect the HTTP routes and MCP endpoint.
            if token and settings.api_key is None:
                from pydantic import SecretStr  # noqa: PLC0415

                settings = settings.model_copy(update={"api_key": SecretStr(token)})
            server = build_server(memory_for_request, settings=settings)
            stop_event = asyncio.Event()
            last_request = time.monotonic()
            port = mcp_settings.port
            if port is None:
                with socket.socket() as probe:
                    probe.bind((mcp_settings.host, 0))
                    port = probe.getsockname()[1]

            async def health(_: Request) -> Response:
                embedder = embedder_source.state.value
                loaded = getattr(embedder_source, "_embedder", None)
                return JSONResponse(
                    {
                        "version": __import__("ontogr_mem").__version__,
                        "pid": process_id,
                        "store": state_target.identity(),
                        "embedder": embedder,
                        "roles": {
                            role: models.configured(role) for role in TEXT_WRITE_ROLES
                        },
                        "embedding_model": embedding_settings.model,
                        "llm_model": os.getenv("MEMORY_LLM_MODEL"),
                        "dimension": loaded.dim
                        if embedder == "ready" and loaded
                        else None,
                        "ontology": (
                            mcp_settings.ontology.name
                            if mcp_settings.ontology
                            else "default"
                        ),
                    }
                )

            async def shutdown(_: Request) -> Response:
                headers = get_http_headers(include={"authorization"})
                if token and not hmac.compare_digest(
                    headers.get("authorization", ""), f"Bearer {token}"
                ):
                    return Response(status_code=401)
                stop_event.set()
                return Response(status_code=202)

            server.custom_route("/health", methods=["GET"])(health)
            server.custom_route("/shutdown", methods=["POST"])(shutdown)
            from starlette.types import Receive, Scope, Send  # noqa: PLC0415

            inner_app = server.http_app()

            async def app(scope: Scope, receive: Receive, send: Send) -> None:
                nonlocal last_request
                if scope["type"] == "http":
                    last_request = time.monotonic()
                await inner_app(scope, receive, send)

            task = asyncio.create_task(
                server.run_async(transport="http", host=mcp_settings.host, port=port)
            )
            preload = asyncio.create_task(embedder_source())
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    if task.done():
                        await task
                        return 76
                    try:
                        await asyncio.to_thread(_health_ready, mcp_settings.host, port)
                    except (OSError, URLError):
                        await asyncio.sleep(0.1)
                        continue
                    break
                else:
                    raise TimeoutError(
                        "MCP server did not become ready within 30 seconds"
                    )
                write_server_info(
                    info_path,
                    {
                        "pid": process_id,
                        "port": port,
                        "version": __import__("ontogr_mem").__version__,
                        "store": state_target.identity(),
                        "started_at": datetime.now(UTC).isoformat(),
                    },
                )
                while not stop_event.is_set():
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=60)
                    except TimeoutError:
                        if server_settings.idle_minutes == 0:
                            continue
                        if (
                            time.monotonic() - last_request
                            < server_settings.idle_minutes * 60
                        ):
                            continue
                        status = await runtime.service.get_status(
                            StdioSettings().to_principal()
                        )
                        if status.queued_jobs == 0:
                            break
                await asyncio.wait_for(runtime.service.drain(), timeout=30)
            finally:
                task.cancel()
                preload.cancel()
                await asyncio.gather(task, preload, return_exceptions=True)
                current = read_server_info(info_path)
                if current and current.get("pid") == process_id:
                    info_path.unlink(missing_ok=True)
        return 0
    finally:
        if store_lock:
            store_lock.release()
        server_lock.release()
