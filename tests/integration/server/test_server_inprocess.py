"""In-process integration tests for the shared HTTP memory server.

These tests drive ``serve`` directly on a real listener and talk to it over
HTTP. Unlike the subprocess lifecycle tests, the server runs under coverage in
this process, so its public lifecycle is measured.
"""

import asyncio
import json
import secrets
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import SecretStr

from ontogr_mem.mcp_server.settings import McpSettings
from ontogr_mem.paths import data_directory
from ontogr_mem.server.app import serve
from ontogr_mem.server.entry import ensure_server
from ontogr_mem.server.settings import ServerSettings
from tests.support.falkordb import falkordb_test_url


TEST_KEY = "inprocess-test-key"


@pytest.fixture(autouse=True)
def server_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point each test at isolated state and a reachable graph service."""
    monkeypatch.setenv("MEMORY_GRAPH_URL", falkordb_test_url())
    monkeypatch.setenv("MEMORY_STATE_URL", f"sqlite:///{tmp_path / 'state.sqlite3'}")
    monkeypatch.setenv("MEMORY_LLM_MODEL", "configured-model")
    monkeypatch.setenv("MEMORY_LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("MEMORY_SERVER_IDLE_MINUTES", "30")
    for key in (
        "MEMORY_MCP_API_KEY",
        "MEMORY_MCP_HOST",
        "MEMORY_MCP_PORT",
        "MEMORY_MCP_TENANT",
    ):
        monkeypatch.delenv(key, raising=False)


def _mcp_settings(*, host: str = "127.0.0.1", with_key: bool = True) -> McpSettings:
    """Return loopback MCP settings, optionally without a bearer token."""
    return McpSettings(
        _env_file=None,
        host=host,
        port=None,
        api_key=SecretStr(TEST_KEY) if with_key else None,
        tenant=f"proc_{secrets.token_hex(6)}",
        skills_extension=False,
    )


def _server_settings() -> ServerSettings:
    """Return server settings that keep the listener up until shutdown."""
    return ServerSettings(_env_file=None, idle_minutes=30)


async def _start_server(
    mcp_settings: McpSettings,
) -> tuple[asyncio.Task[int], dict[str, object]]:
    """Start ``serve`` in-process and wait for its published listener info."""
    task = asyncio.create_task(
        serve(mcp_settings=mcp_settings, server_settings=_server_settings())
    )
    info_path = data_directory() / "server.json"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if task.done():
            pytest.fail(f"server exited during startup with {task.result()}")
        try:
            info = json.loads(info_path.read_text())
        except (OSError, ValueError):
            info = {}
        if isinstance(info.get("port"), int):
            return task, info
        await asyncio.sleep(0.1)
    pytest.fail("server did not publish listener info within 30 seconds")


async def _http(request: Request) -> tuple[int, bytes]:
    """Issue one HTTP request off-loop and return its status and body."""

    def fetch() -> tuple[int, bytes]:
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    return await asyncio.to_thread(fetch)


async def _status(request: Request) -> int:
    """Return the HTTP status of one request."""
    return (await _http(request))[0]


def _authorized(path: str, base_url: str, token: str = TEST_KEY) -> Request:
    """Build a request carrying the server bearer token."""
    return Request(f"{base_url}{path}", headers={"Authorization": f"Bearer {token}"})


def _shutdown_request(base_url: str, token: str = TEST_KEY) -> Request:
    """Build an authenticated shutdown request."""
    return Request(
        f"{base_url}/shutdown",
        data=b"{}",
        headers={"Authorization": f"Bearer {token}"},
        method="POST",
    )


@asynccontextmanager
async def _serving() -> AsyncGenerator[
    tuple[asyncio.Task[int], dict[str, object], str]
]:
    """Run a server for one test and shut it down even when the test fails."""
    task, info = await _start_server(_mcp_settings())
    base_url = f"http://127.0.0.1:{info['port']}"
    try:
        yield task, info, base_url
    finally:
        if not task.done():
            await _http(_shutdown_request(base_url))
        await asyncio.wait_for(task, timeout=40)


class TestInProcessServer:
    """Exercise the served listener under in-process coverage."""

    async def test_health_auth_and_shutdown(self) -> None:
        """Health and shutdown routes enforce the configured bearer token."""
        async with _serving() as (task, _info, base_url):
            status, body = await _http(_authorized("/health", base_url))
            assert status == 200
            health = json.loads(body)
            assert health["version"]
            assert health["store"]
            assert health["llm_model"] == "configured-model"

            assert await _status(Request(f"{base_url}/health")) == 401
            assert (
                await _status(_authorized("/health", base_url, token="wrong-token"))
                == 401
            )
            assert await _status(Request(f"{base_url}/mcp")) == 401
            assert (
                await _status(_shutdown_request(base_url, token="wrong-token")) == 401
            )

            assert await _status(_shutdown_request(base_url)) == 202
            assert await asyncio.wait_for(task, timeout=40) == 0

        assert not (data_directory() / "server.json").exists()

    async def test_mcp_tool_call_over_http(self) -> None:
        """An authenticated MCP client calls a read tool on the served listener."""
        async with _serving() as (_task, _info, base_url):
            transport = StreamableHttpTransport(
                f"{base_url}/mcp",
                headers={"Authorization": f"Bearer {TEST_KEY}"},
            )
            async with Client(transport) as client:
                names = {tool.name for tool in await client.list_tools()}
                assert "get_ontology" in names

                result = await client.call_tool("get_ontology", {})
                assert not result.is_error
                ontology = result.structured_content
                assert isinstance(ontology, dict)
                assert {"tenant", "version", "entity_types", "relation_types"} <= set(
                    ontology
                )

    async def test_explicit_mcp_url_returns_server_endpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An explicit MCP URL resolves to the server base URL."""
        monkeypatch.chdir(tmp_path)
        async with _serving() as (_task, _info, base_url):
            monkeypatch.setenv("MEMORY_MCP_SERVER_URL", f"{base_url}/mcp")
            monkeypatch.setenv("MEMORY_MCP_API_KEY", TEST_KEY)

            endpoint = await ensure_server(
                graph_url=falkordb_test_url(),
                state_url=f"sqlite:///{tmp_path / 'state.sqlite3'}",
            )

            assert endpoint.url == base_url
            assert endpoint.token == TEST_KEY

    async def test_non_loopback_without_key_refuses_then_restarts(self) -> None:
        """A keyless non-loopback bind refuses and frees locks for a later start."""
        result = await serve(
            mcp_settings=_mcp_settings(host="0.0.0.0", with_key=False),
            server_settings=_server_settings(),
        )
        assert result == 78
        assert not (data_directory() / "server.json").exists()

        async with _serving() as (task, _info, base_url):
            assert await _status(_authorized("/health", base_url)) == 200
            assert await _status(_shutdown_request(base_url)) == 202
            assert await asyncio.wait_for(task, timeout=40) == 0
