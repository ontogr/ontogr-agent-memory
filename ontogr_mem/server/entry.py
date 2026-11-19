"""Start or attach to the per-user memory server."""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen

from ontogr_mem import __version__
from ontogr_mem.errors import AgentMemoryError
from ontogr_mem.mcp_server.settings import McpSettings
from ontogr_mem.paths import data_directory, ensure_directory
from ontogr_mem.server.files import read_server_info, read_server_token
from ontogr_mem.store_urls import parse_graph_url, parse_state_url


class ServerUnavailableError(AgentMemoryError):
    """The shared memory server cannot serve this client."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        """Create an error with an optional recovery hint."""
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True, slots=True)
class ServerEndpoint:
    """Connection details and capabilities reported by the server."""

    url: str
    token: str | None
    roles: dict[str, bool]
    embedding_model: str | None
    llm_model: str | None
    dimension: int | None
    ontology: str


def _store_identity(graph_url: str, state_url: str) -> str:
    parse_graph_url(graph_url)
    return parse_state_url(state_url).identity()


def _health(url: str) -> dict[str, object] | None:
    try:
        with urlopen(Request(f"{url}/health"), timeout=2) as response:
            payload = json.loads(response.read())
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, URLError, TimeoutError):
        return None


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    return tuple(map(int, match.group(1).split("."))) if match else (0,)


def _child_environment(graph_url: str, state_url: str) -> dict[str, str]:
    exact = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "SYSTEMROOT",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "FASTEMBED_CACHE_PATH",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "HF_ENDPOINT",
        "MEMORY_MCP_API_KEY",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in exact or key.startswith(("XDG_", "LC_"))
    }
    environment.update(
        {
            "MEMORY_GRAPH_URL": graph_url,
            "MEMORY_STATE_URL": state_url,
            "MEMORY_DATA_DIR": str(data_directory().resolve()),
        }
    )
    return environment


def _endpoint(url: str, health: dict[str, object], token: str | None) -> ServerEndpoint:
    roles = health.get("roles")
    embedding_model = health.get("embedding_model")
    llm_model = health.get("llm_model")
    dimension = health.get("dimension")
    return ServerEndpoint(
        url=url,
        token=token,
        roles=roles if isinstance(roles, dict) else {},
        embedding_model=embedding_model if isinstance(embedding_model, str) else None,
        llm_model=llm_model if isinstance(llm_model, str) else None,
        dimension=dimension if isinstance(dimension, int) else None,
        ontology=str(health.get("ontology", "default")),
    )


async def ensure_server(  # noqa: PLR0912, PLR0915
    *, graph_url: str, state_url: str, version: str = __version__
) -> ServerEndpoint:
    """Attach to a compatible server or start the per-user server process."""
    explicit_url = os.environ.get("MEMORY_MCP_SERVER_URL")
    directory = ensure_directory(data_directory())
    settings = McpSettings()
    token = (
        settings.api_key.get_secret_value() if settings.api_key is not None else None
    ) or read_server_token(directory / "server.token")
    if explicit_url:
        health = await asyncio.to_thread(_health, explicit_url.rstrip("/"))
        if health is None:
            raise ServerUnavailableError(
                f"Memory server at {explicit_url} is unavailable"
            )
        return _endpoint(explicit_url.rstrip("/"), health, token)

    expected_store = _store_identity(graph_url, state_url)
    info = read_server_info(directory / "server.json")
    current_url = (
        f"http://127.0.0.1:{info['port']}"
        if info and isinstance(info.get("port"), int)
        else ""
    )
    health = await asyncio.to_thread(_health, current_url) if current_url else None
    if health is not None:
        if health.get("store") != expected_store:
            raise ServerUnavailableError(
                f"Memory server is using a different store: {health.get('store')!r}",
                "Run ontogr-mem server stop before connecting to another store.",
            )
        running_version = str(health.get("version", "0"))
        if _version_tuple(running_version) > _version_tuple(version):
            raise ServerUnavailableError(
                "The memory server is newer; update the plugin or the package."
            )
        if _version_tuple(running_version) == _version_tuple(version):
            return _endpoint(current_url, health, token)
        try:
            request = Request(
                f"{current_url}/shutdown",
                method="POST",
                headers={"Authorization": f"Bearer {token}"} if token else {},
            )
            await asyncio.to_thread(urlopen, request, timeout=2)
        except (OSError, URLError):
            pass
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if await asyncio.to_thread(_health, current_url) is None:
                break
            await asyncio.sleep(0.2)

    log_path = directory / "server.log"
    attempts = 0
    deadline = time.monotonic() + 30
    process: subprocess.Popen[bytes] | None = None
    while time.monotonic() < deadline:
        if process is None or (process.poll() is not None and process.returncode != 75):
            if process is not None:
                if process.returncode == 76:
                    if attempts >= 1:
                        raise ServerUnavailableError(
                            f"Memory server failed to start. See {log_path}"
                        )
                    attempts += 1
                elif process.returncode != 0:
                    raise ServerUnavailableError(
                        f"Memory server failed to start. See {log_path}"
                    )
            arguments = [sys.executable, "-m", "ontogr_mem", "server", "run"]
            output = log_path.open("ab")
            try:
                if os.name == "nt":
                    process = subprocess.Popen(
                        arguments,
                        cwd=directory,
                        env=_child_environment(graph_url, state_url),
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        text=False,
                        creationflags=(
                            subprocess.DETACHED_PROCESS
                            | subprocess.CREATE_NEW_PROCESS_GROUP
                        ),
                    )
                else:
                    process = subprocess.Popen(
                        arguments,
                        cwd=directory,
                        env=_child_environment(graph_url, state_url),
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        text=False,
                        start_new_session=True,
                    )
            finally:
                output.close()
        info = read_server_info(directory / "server.json")
        if info and isinstance(info.get("port"), int):
            current_url = f"http://127.0.0.1:{info['port']}"
            health = await asyncio.to_thread(_health, current_url)
            if (
                health
                and health.get("version") == version
                and health.get("store") == expected_store
            ):
                refreshed = McpSettings()
                token = (
                    refreshed.api_key.get_secret_value()
                    if refreshed.api_key is not None
                    else None
                ) or read_server_token(directory / "server.token")
                return _endpoint(current_url, health, token)
        await asyncio.sleep(0.2)
    raise ServerUnavailableError(f"Memory server did not become ready. See {log_path}")
