"""Read-only diagnostics for an ontogr-mem installation."""

import asyncio
import json
import os
import urllib.request
from importlib.util import find_spec
from pathlib import Path

from dotenv import dotenv_values

from ontogr_mem.llm import build_models
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.paths import cache_directory, data_directory, model_cache_directory
from ontogr_mem.requirements import REQUIREMENTS, TEXT_WRITE_ROLES, check_requirements
from ontogr_mem.server.files import read_server_info, read_server_token
from ontogr_mem.service.bootstrap import open_graph_store
from ontogr_mem.store_urls import (
    DEFAULT_GRAPH_URL,
    is_loopback_host,
    parse_graph_url,
    parse_state_url,
)
from ontogr_mem.vectorstore.builtin.local import model_cached
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings


async def run_doctor() -> int:
    """Print setup checks without opening state or creating directories."""
    results = []
    graph, state = _check_urls(results)
    _check_preflight(results, graph, state)
    await _check_state(results, state)
    tenants = await _check_graph(results, graph)
    _check_tenant_rules(results, graph)
    _check_embedding(results)
    _check_models(results)
    _check_server_env(results)
    _check_server(results)
    _print_locations(graph, state, tenants)
    failed = False
    for status, label, detail, fix in results:
        print(f"[{status}] {label}: {detail}")
        if fix:
            print(f"       fix: {fix}")
        failed |= status == "fail"
    return int(failed)


def _check_urls(results):
    graph = state = None
    try:
        graph = parse_graph_url(os.environ.get("MEMORY_GRAPH_URL") or DEFAULT_GRAPH_URL)
        results.append(("pass", "graph URL", graph.identity(), None))
    except Exception:
        results.append(
            (
                "fail",
                "graph URL",
                "invalid URL",
                "export MEMORY_GRAPH_URL=redis://localhost:6379",
            )
        )
    try:
        state = parse_state_url(
            os.environ.get("MEMORY_STATE_URL", "sqlite:///memory.db")
        )
    except Exception:
        results.append(("fail", "state", "invalid URL", "check MEMORY_STATE_URL"))
    return graph, state


def _check_preflight(results, graph, state):
    if graph is None or state is None:
        results.append(("warn", "preflight", "not checked, invalid store URL", None))
        return
    check = check_requirements(
        models=None, graph=graph, state=state, embedding=EmbeddingSettings()
    )
    if check.missing:
        results.append(
            (
                "fail",
                "preflight",
                f"missing extras: {', '.join(check.missing)}",
                f"uv sync --extra {check.missing[0]}",
            )
        )
    else:
        results.append(
            ("pass", "preflight", f"available: {', '.join(check.required)}", None)
        )


async def _check_state(results, state):
    if state is None:
        return
    if state.backend == "sqlite":
        path = Path(state.path)
        if not path.exists():
            results.append(("pass", "state", f"{path} created on first use", None))
        elif os.access(path, os.R_OK | os.W_OK):
            results.append(("pass", "state", f"{path} readable and writable", None))
        else:
            results.append(
                ("fail", "state", f"{path} inaccessible", "check MEMORY_STATE_URL")
            )
    else:
        try:
            from ontogr_mem.relational_store.probe import probe_state  # noqa: PLC0415

            reachable = await probe_state(state)
            results.append(
                (
                    "pass" if reachable else "fail",
                    "state",
                    "PostgreSQL reachable" if reachable else "PostgreSQL unavailable",
                    None if reachable else "check MEMORY_STATE_URL",
                )
            )
        except Exception as error:
            results.append(
                (
                    "fail",
                    "state",
                    f"PostgreSQL probe failed ({type(error).__name__})",
                    "check MEMORY_STATE_URL",
                )
            )


async def _check_graph(results, graph):
    if graph is None:
        results.append(("warn", "graph reachable", "not checked, invalid URL", None))
        return None
    tenants = None
    try:
        async with open_graph_store() as store:
            await asyncio.wait_for(store.health(), timeout=3)
            tenants = await store.list_tenants()
        results.append(("pass", "graph reachable", graph.identity(), None))
    except Exception as error:
        fix = (
            "docker run -d -p 127.0.0.1:6379:6379 "
            "-v ontogr-mem-falkordb-data:/var/lib/falkordb/data "
            "falkordb/falkordb:v4.22.2"
            if graph.backend == "falkordb" and is_loopback_host(graph.host)
            else "check MEMORY_GRAPH_URL and the network path"
        )
        results.append(("fail", "graph reachable", type(error).__name__, fix))
    return tenants


def _check_tenant_rules(results, graph):
    if graph is not None and not is_loopback_host(graph.host):
        tenant = os.environ.get("MEMORY_MCP_TENANT")
        if not tenant or tenant == "default":
            results.append(
                (
                    "fail",
                    "tenant rule",
                    "remote graph requires an explicit tenant",
                    "export MEMORY_MCP_TENANT=<tenant>",
                )
            )
        else:
            results.append(("pass", "tenant rule", f"tenant {tenant} configured", None))
    else:
        results.append(("pass", "tenant rule", "loopback graph", None))
    if (
        graph is not None
        and not is_loopback_host(graph.host)
        and getattr(graph, "backend", None)
        and os.environ.get("MEMORY_STATE_URL", "sqlite:///memory.db").startswith(
            "sqlite:"
        )
    ):
        results.append(
            (
                "warn",
                "remote graph state",
                "SQLite state is local to this machine",
                "export MEMORY_STATE_URL=postgresql://<user>@<host>/<database>",
            )
        )


def _check_embedding(results):
    settings = EmbeddingSettings()
    if settings.provider == "endpoint" or settings.base_url:
        results.append(("pass", "embedder", "endpoint configured", None))
    elif _has_extra("fastembed"):
        try:
            cached = model_cached(settings)
        except Exception:
            cached = None
        if cached is True:
            results.append(
                ("pass", "embedder", f"model cached at {model_cache_directory()}", None)
            )
        elif cached is False:
            results.append(("warn", "embedder", "model not cached", "ontogr-mem warm"))
        else:
            results.append(("warn", "embedder", "cache not checked", None))
    else:
        results.append(("warn", "embedder", "not checked, missing extra", None))


def _check_models(results):
    try:
        models = build_models()
        roles = [
            role
            for role in (
                ModelRole.ANSWER,
                ModelRole.ENTITY_EXTRACTION,
                ModelRole.FACT_EXTRACTION,
            )
            if models.configured(role)
        ]
        mode = "keyless" if not models.any_configured() else "model"
        add_memory = all(models.configured(role) for role in TEXT_WRITE_ROLES)
        results.append(
            (
                "pass",
                "models and mode",
                f"{mode}; configured roles: "
                f"{', '.join(role.value for role in roles) or 'none'}; "
                f"add_memory {'listed' if add_memory else 'not listed'}",
                None,
            )
        )
    except Exception as error:
        results.append(
            (
                "warn",
                "models and mode",
                f"settings error ({type(error).__name__})",
                None,
            )
        )


def _check_server_env(results):
    path = data_directory() / "server.env"
    values = dotenv_values(path) if path.exists() else {}
    keys = sorted(key for key, value in values.items() if value is not None)
    forbidden = {
        "MEMORY_GRAPH_URL",
        "MEMORY_STATE_URL",
        "MEMORY_DATA_DIR",
    }.intersection(keys)
    status = "fail" if forbidden else "pass"
    results.append(
        (
            status,
            "settings file",
            f"{path} {'exists' if path.exists() else 'absent'}; "
            f"keys: {', '.join(keys) or 'none'}",
            f"remove {sorted(forbidden)[0]} from server.env" if forbidden else None,
        )
    )


def _check_server(results):
    data = data_directory()
    info = read_server_info(data / "server.json")
    if not info:
        results.append(("pass", "running server", "not running", None))
        return
    token = read_server_token(data / "server.token")
    request = urllib.request.Request(
        f"http://127.0.0.1:{info.get('port')}/health",
        headers={"Authorization": f"Bearer {token}"} if token else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            health = json.load(response)
        results.append(
            ("pass", "running server", f"http://127.0.0.1:{info.get('port')}", None)
        )
        models = build_models()
        expected = {role.value: models.configured(role) for role in TEXT_WRITE_ROLES}
        if health.get("roles") != expected:
            results.append(
                (
                    "warn",
                    "server settings",
                    "its extraction roles differ from these settings. The server reads "
                    "server.env in its data directory.",
                    "edit server.env, run ontogr-mem server stop, then run this "
                    "command again",
                )
            )
    except Exception as error:
        results.append(
            (
                "warn",
                "running server",
                f"status unavailable ({type(error).__name__})",
                None,
            )
        )


def _print_locations(graph, state, tenants):
    data = data_directory()
    cache = cache_directory()
    locations = [data, cache, model_cache_directory()]
    locations.extend(
        data / name
        for name in ("server.lock", "server.token", "server.json", "server.env")
    )
    if state is not None and state.backend == "sqlite":
        path = Path(state.path)
        locations.extend((path, Path(f"{path}-wal"), Path(f"{path}-shm")))
    for path in locations:
        print(f"[info] {path}: {'exists' if path.exists() else 'absent'}")
    print(
        "[info] Docker volumes: ontogr-mem-falkordb-data, ontogr-mem-state, "
        "ontogr-mem-models (Compose only)"
    )
    print(
        f"[info] graph names: m_<tenant>; tenants: "
        f"{len(tenants) if tenants is not None else 'not checked'}"
    )


def _has_extra(name):
    try:
        return all(
            find_spec(module) is not None for module in REQUIREMENTS[name].modules
        )
    except (ImportError, ValueError):
        return False
