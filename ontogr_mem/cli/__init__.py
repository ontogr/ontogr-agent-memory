"""The ontogr-mem console command."""

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from dotenv import dotenv_values
from pydantic import ValidationError

from ontogr_mem.agent import build_default_agent
from ontogr_mem.cli.chat import run_chat
from ontogr_mem.client import Memory
from ontogr_mem.data_models.tenant import InvalidTenantError, validate_tenant_id
from ontogr_mem.errors import AgentMemoryError, MissingExtraError, StoreInUseError
from ontogr_mem.llm import build_models
from ontogr_mem.llm.failures import ModelNotConfiguredError
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.mcp_server.settings import McpSettings
from ontogr_mem.ontology.errors import InvalidChange
from ontogr_mem.paths import data_directory
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.requirements import check_requirements
from ontogr_mem.server.entry import ServerUnavailableError
from ontogr_mem.server.files import read_server_info, read_server_token
from ontogr_mem.server.settings import ServerSettings
from ontogr_mem.service.bootstrap import ServiceSettings
from ontogr_mem.service.errors import InvalidArgumentError
from ontogr_mem.store_urls import (
    ExplicitTenantError,
    parse_graph_url,
    parse_state_url,
    require_explicit_tenant,
)
from ontogr_mem.vectorstore.builtin import EmbedderUnavailableError
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings


def _tenant_id_argument(value: str) -> str:
    """Validate a tenant id argument and report the validator's message on failure."""
    try:
        return validate_tenant_id(value)
    except InvalidTenantError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


async def _chat(tenant: str | None) -> None:
    """Keep the client open until chat and its drain have finished."""
    require_explicit_tenant(
        ServiceSettings().graph_url,
        tenant_given=tenant is not None,
        hint="pass --tenant",
    )
    models = build_models()
    try:
        service = ServiceSettings()
        state = StateSettings()
        check_requirements(
            models=models,
            graph=parse_graph_url(service.graph_url.get_secret_value()),
            state=parse_state_url(state.url.get_secret_value()),
            embedding=EmbeddingSettings(),
            required_roles=frozenset({ModelRole.ANSWER}),
        ).raise_for_missing(subcommand="chat")
        async with Memory.connect(tenant=tenant or "default", models=models) as memory:
            await run_chat(memory, build_default_agent(memory))
    finally:
        await models.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse a console command and return its process exit status.

    Args:
        argv: Arguments without the executable name, or the process arguments.

    Returns:
        Zero on normal exit, one on a runtime failure, or 130 on Ctrl-C.
        Argparse exits with its standard help and usage statuses.
    """
    parser = _command_parser()
    args = parser.parse_args(argv)
    os.environ["PYDANTIC_AI_NO_BANNER"] = "1"
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        return _report_error(error)


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "server":
        return _run_server_command(args)
    if args.command == "mcp":
        service = ServiceSettings()
        state = StateSettings()
        check_requirements(
            models=None,
            graph=parse_graph_url(service.graph_url.get_secret_value()),
            state=parse_state_url(state.url.get_secret_value()),
            embedding=EmbeddingSettings(),
            mcp=True,
        ).raise_for_missing(subcommand="mcp")
        from ontogr_mem.cli.mcp import run_mcp  # noqa: PLC0415

        asyncio.run(run_mcp())
        return 0
    asyncio.run(_chat(args.tenant))
    return 0


def _report_error(error: Exception) -> int:
    if isinstance(error, MissingExtraError):
        message = str(error)
    elif isinstance(error, EmbedderUnavailableError):
        message = f"{error}\n{error.hint}"
    elif isinstance(error, ServerUnavailableError):
        message = f"{error}\n{error.hint}" if error.hint else str(error)
    elif isinstance(error, ModelNotConfiguredError):
        message = (
            "Configure MEMORY_LLM_MODEL, MEMORY_LLM_BASE_URL and "
            "MEMORY_LLM_API_KEY for answers and text extraction."
        )
    elif isinstance(error, ValidationError):
        message = "Invalid memory settings; check the named MEMORY_* variables."
    elif isinstance(error, InvalidChange):
        message = f"Invalid ontology: {error}"
    elif isinstance(
        error,
        (
            InvalidArgumentError,
            ExplicitTenantError,
            StoreInUseError,
        ),
    ):
        message = str(error)
    elif isinstance(error, AgentMemoryError):
        message = "Memory failed; check database, embedding and model settings."
    else:
        message = "Command failed; check the model endpoint and memory services."
    print(message, file=sys.stderr)
    return 1


def _run_server_command(args: argparse.Namespace) -> int:
    _apply_server_env()
    if args.server_command == "run":
        from ontogr_mem.server.app import (  # noqa: PLC0415
            SERVER_ALREADY_RUNNING,
            SERVER_DRAIN_INCOMPLETE,
            serve,
        )

        settings_kwargs: dict[str, str | Path] = {}
        if args.tenant is not None:
            settings_kwargs["tenant"] = args.tenant
        if args.ontology is not None:
            settings_kwargs["ontology"] = args.ontology
        settings = McpSettings(**cast(dict[str, Any], settings_kwargs))
        result = asyncio.run(
            serve(mcp_settings=settings, server_settings=ServerSettings())
        )
        if result == SERVER_ALREADY_RUNNING:
            print(
                "server is already running (another instance holds the lock)",
                file=sys.stderr,
            )
        if result == SERVER_DRAIN_INCOMPLETE:
            print(
                "server stopped before all jobs drained; "
                "pending jobs remain and will be recovered on next start",
                file=sys.stderr,
            )
        return result
    if args.server_command == "status":
        return _server_status()
    return _server_stop()


def _command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ontogr-mem")
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="chat with persistent memory")
    chat.add_argument("--tenant", default=None, type=_tenant_id_argument)
    commands.add_parser("mcp", help="serve memory over MCP")
    server = commands.add_parser("server", help="manage the shared memory server")
    server_commands = server.add_subparsers(dest="server_command", required=True)
    run = server_commands.add_parser("run", help="run the memory server")
    run.add_argument("--tenant", default=None, type=_tenant_id_argument)
    run.add_argument("--ontology", type=Path)
    server_commands.add_parser("status", help="show server status")
    server_commands.add_parser("stop", help="stop the local server")
    return parser


def _apply_server_env() -> None:
    """Apply allowed values from the per-user server settings file."""
    forbidden = {"MEMORY_GRAPH_URL", "MEMORY_STATE_URL", "MEMORY_DATA_DIR"}
    values = dotenv_values(data_directory() / "server.env")
    present = forbidden.intersection(key for key, value in values.items() if value)
    if present:
        name = sorted(present)[0]
        msg = f"server.env cannot set {name}; remove it from server.env"
        raise InvalidArgumentError(msg)
    for key, value in values.items():
        if value is not None:
            os.environ.setdefault(key, value)


def _server_status() -> int:
    info = read_server_info(data_directory() / "server.json")
    if not info:
        print("not running")
        return 1
    api_key = McpSettings().api_key
    if api_key is not None:
        token = api_key.get_secret_value()
    else:
        token = read_server_token(data_directory() / "server.token")
    request = urllib.request.Request(
        f"http://127.0.0.1:{info['port']}/health",
        headers={"Authorization": f"Bearer {token}"} if token else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            health = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 401:
            if token:
                print(
                    "server authentication failed; stored token does not match "
                    "MEMORY_MCP_API_KEY",
                    file=sys.stderr,
                )
            else:
                print(
                    "server requires authentication but no token was sent; "
                    "set MEMORY_MCP_API_KEY or provide server.token",
                    file=sys.stderr,
                )
            return 1
        print("not running")
        return 1
    except (OSError, urllib.error.URLError):
        print("not running")
        return 1
    roles = health.get("roles", {})
    print(
        f"running pid={health.get('pid')} version={health.get('version')} "
        f"port={info.get('port')} store={health.get('store')} "
        f"embedder={health.get('embedder')} roles={roles} "
        f"embedding_model={health.get('embedding_model')}"
    )
    return 0


def _server_stop() -> int:
    info = read_server_info(data_directory() / "server.json")
    if not info:
        print("not running")
        return 1
    api_key = McpSettings().api_key
    if api_key is not None:
        token = api_key.get_secret_value()
    else:
        token = read_server_token(data_directory() / "server.token")
    request = urllib.request.Request(
        f"http://127.0.0.1:{info['port']}/shutdown",
        data=b"{}",
        headers={"Authorization": f"Bearer {token}"} if token else {},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return 0 if response.status == 202 else 1
    except urllib.error.HTTPError as error:
        if error.code == 401:
            if token:
                print(
                    "server authentication failed; the stored token does not match "
                    "MEMORY_MCP_API_KEY. Remove the stale server.token or set "
                    "MEMORY_MCP_API_KEY to the value used to start the server.",
                    file=sys.stderr,
                )
            else:
                print(
                    "server requires authentication but no token was sent; "
                    "set MEMORY_MCP_API_KEY or provide server.token",
                    file=sys.stderr,
                )
            return 1
        print("server is unreachable; no process was stopped", file=sys.stderr)
        return 1
    except (OSError, urllib.error.URLError):
        print("server is unreachable; no process was stopped", file=sys.stderr)
        return 1


__all__ = ["main"]
