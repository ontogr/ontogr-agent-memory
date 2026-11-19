"""The MCP runner and the ontogr-mem mcp command, with transport and storage faked."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import FastMCP
from pydantic import SecretStr

from ontogr_mem.cli import main
from ontogr_mem.cli.mcp import run_mcp
from ontogr_mem.errors import MissingExtraError
from ontogr_mem.mcp_server.settings import McpSettings
from ontogr_mem.store_urls import ExplicitTenantError
from tests.support.mcp_memory import FakeMemoryService, build_memory


SENTINEL_KEY = "sentinel-key"


def _fake_asyncio_run(outcome):
    """Stand in for asyncio.run: close the coroutine, then raise outcome if set."""

    def fake_run(coroutine):
        coroutine.close()
        if outcome is not None:
            raise outcome

    return fake_run


class RecordingService(FakeMemoryService):
    """A fake service that records the order of drain and context exit."""

    def __init__(self, events: list[str]) -> None:
        """Create a service that appends to the shared event list."""
        super().__init__()
        self.events = events

    async def drain(self) -> None:
        """Record that queued writes were finished."""
        self.events.append("drain")


class Harness:
    """Fakes for memory and the transport, plus what they saw.

    Attributes:
        events: Ordered names of what happened, such as open, serve, drain, close.
        transport_calls: Keyword arguments of each run_async call.
        failure: Exception the fake transport raises, or None to return.
        tenants: Tenant given to each Memory.open call.
    """

    def __init__(self) -> None:
        """Create a harness with no recorded events."""
        self.events: list[str] = []
        self.transport_calls: list[dict] = []
        self.failure: Exception | None = None
        self.tenants: list[str] = []
        self.ontologies: list[Path | None] = []

    def open_memory(self, *, tenant: str, ontology=None, models):
        """Stand in for Memory.open with a memory over a recording service."""
        self.tenants.append(tenant)
        self.ontologies.append(ontology)
        self.models = models

        @asynccontextmanager
        async def opened():
            self.events.append("open")
            try:
                yield build_memory(RecordingService(self.events))
            finally:
                self.events.append("close")

        return opened()

    async def run_async(self, **kwargs) -> None:
        """Stand in for FastMCP.run_async by recording its arguments."""
        self.events.append("serve")
        self.transport_calls.append(kwargs)
        if self.failure is not None:
            raise self.failure


@pytest.fixture
def harness(monkeypatch) -> Harness:
    """Replace memory opening and the FastMCP transport with fakes."""
    fakes = Harness()
    monkeypatch.setattr("ontogr_mem.cli.mcp.Memory.open", fakes.open_memory)
    # A function on the class receives the server first; the fake ignores it.
    monkeypatch.setattr(
        FastMCP, "run_async", lambda _server, **kwargs: fakes.run_async(**kwargs)
    )
    return fakes


def _settings(**values) -> McpSettings:
    """Build settings that ignore the environment."""
    return McpSettings(_env_file=None, **values)


class TestRunMcp:
    """Behavior of the runner that serves memory over MCP."""

    async def test_remote_graph_requires_explicit_tenant_before_open(
        self, harness, monkeypatch
    ):
        """A remote graph cannot use the implicit default tenant."""
        monkeypatch.setenv("MEMORY_GRAPH_URL", "redis://graph.example:6379")

        with pytest.raises(ExplicitTenantError, match="MEMORY_MCP_TENANT"):
            await run_mcp(_settings())

        assert harness.events == []

    async def test_loopback_graph_allows_implicit_tenant(self, harness, monkeypatch):
        """The implicit default tenant remains valid for a loopback graph."""
        monkeypatch.setenv("MEMORY_GRAPH_URL", "redis://localhost:6379")

        await run_mcp(_settings())

        assert harness.tenants == ["default"]

    async def test_explicit_tenant_allows_remote_graph(self, harness, monkeypatch):
        """An explicit tenant allows a remote graph connection."""
        monkeypatch.setenv("MEMORY_GRAPH_URL", "redis://graph.example:6379")

        await run_mcp(_settings(tenant="acme"))

        assert harness.tenants == ["acme"]

    async def test_stdio_serves_without_banner_and_drains_before_close(self, harness):
        """Stdio runs with no banner and writes finish before memory closes."""
        await run_mcp(_settings(transport="stdio"))

        assert harness.transport_calls == [{"transport": "stdio", "show_banner": False}]
        assert harness.events == ["open", "serve", "drain", "close"]

    async def test_http_serves_on_the_configured_host_and_port(self, harness):
        """The http transport receives the bind address from the settings."""
        await run_mcp(_settings(transport="http", host="127.0.0.1", port=9123))

        assert harness.transport_calls == [
            {
                "transport": "http",
                "host": "127.0.0.1",
                "port": 9123,
                "show_banner": False,
            }
        ]

    async def test_opens_memory_for_the_configured_tenant(self, harness):
        """The tenant setting selects the tenant the memory is opened for."""
        await run_mcp(_settings(tenant="acme"))

        assert harness.tenants == ["acme"]

    async def test_passes_shared_models_to_memory_open(self, harness, monkeypatch):
        """The model set built for startup is shared with the opened memory."""

        async def close_models():
            return None

        models = SimpleNamespace(aclose=close_models, configured=lambda _role: False)
        monkeypatch.setattr("ontogr_mem.cli.mcp.build_models", lambda: models)

        await run_mcp(_settings())

        assert harness.models is models

    async def test_the_ontology_setting_reaches_memory_open(self, harness):
        """The configured ontology path is passed to Memory.open."""
        await run_mcp(_settings(ontology=Path("o.yaml")))

        assert harness.ontologies == [Path("o.yaml")]

    async def test_transport_failure_propagates_after_drain(self, harness):
        """A failing transport raises out of the runner and still drains."""
        harness.failure = RuntimeError("transport broke")

        with pytest.raises(RuntimeError, match="transport broke"):
            await run_mcp(_settings())

        assert harness.events == ["open", "serve", "drain", "close"]

    async def test_runs_on_the_callers_event_loop(self, harness):
        """The runner is awaited here, so it starts no nested event loop."""
        # asyncio.run inside a running loop would raise instead of returning.
        await run_mcp(_settings())

        assert harness.events[:2] == ["open", "serve"]

    @pytest.mark.parametrize(
        ("host", "key", "warns"),
        [
            pytest.param("127.0.0.1", None, False, id="ipv4-loopback"),
            pytest.param("::1", None, False, id="ipv6-loopback"),
            pytest.param("localhost", None, False, id="localhost"),
            pytest.param("0.0.0.0", None, True, id="wildcard-without-key"),
            pytest.param("example.com", None, True, id="hostname-without-key"),
            pytest.param("0.0.0.0", SENTINEL_KEY, False, id="wildcard-with-key"),
            pytest.param("example.com", SENTINEL_KEY, False, id="hostname-with-key"),
        ],
    )
    async def test_warns_only_for_open_binding_without_a_key(
        self, harness, caplog, host, key, warns
    ):
        """Only a non-loopback http bind without a key logs a warning."""
        settings = _settings(
            transport="http",
            host=host,
            api_key=SecretStr(key) if key else None,
        )

        with caplog.at_level(logging.WARNING, logger="ontogr_mem.cli.mcp"):
            await run_mcp(settings)

        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert bool(warnings) is warns
        assert all(SENTINEL_KEY not in message for message in warnings)
        assert len(harness.transport_calls) == 1

    async def test_stdio_never_warns_about_the_bind_address(self, harness, caplog):
        """The host is unused by stdio, so a public host does not warn."""
        with caplog.at_level(logging.WARNING, logger="ontogr_mem.cli.mcp"):
            await run_mcp(_settings(transport="stdio", host="0.0.0.0"))

        assert caplog.records == []


class TestMcpCommand:
    """Behavior of the mcp subcommand: settings, overrides and failures."""

    @pytest.fixture(autouse=True)
    def isolated_environment(self, monkeypatch, tmp_path):
        """Clear MCP and graph URL variables and the banner variable after each test."""
        for name in list(os.environ):
            if name.startswith("MEMORY_MCP_") or name == "MEMORY_GRAPH_URL":
                monkeypatch.delenv(name)
        # An empty directory keeps a developer dotenv file out of the settings.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "0")
        monkeypatch.delenv("PYDANTIC_AI_NO_BANNER")

    @pytest.fixture
    def seen(self, monkeypatch) -> list[McpSettings]:
        """Record the settings the command hands to the runner."""
        recorded: list[McpSettings] = []

        async def idle() -> None:
            """Stand in for the runner coroutine."""

        def fake_runner(settings):
            recorded.append(settings)
            return idle()

        monkeypatch.setattr("ontogr_mem.cli.run_mcp", fake_runner)
        monkeypatch.setattr("ontogr_mem.cli.asyncio.run", _fake_asyncio_run(None))
        return recorded

    def test_flags_override_environment_values(self, seen, monkeypatch):
        """A given flag wins over the matching environment variable."""
        monkeypatch.setenv("MEMORY_MCP_PORT", "9000")

        status = main(["mcp", "--port", "9100"])

        assert status == 0
        assert seen[0].port == 9100

    def test_every_flag_reaches_the_settings(self, seen):
        """Transport, host, port and tenant flags all set their field."""
        main(
            [
                "mcp",
                "--transport",
                "http",
                "--host",
                "0.0.0.0",
                "--port",
                "9100",
                "--tenant",
                "acme",
            ]
        )

        settings = seen[0]
        assert (settings.transport, settings.host, settings.port, settings.tenant) == (
            "http",
            "0.0.0.0",
            9100,
            "acme",
        )

    def test_the_ontology_flag_reaches_the_settings(self, seen):
        """The ontology path flag sets the MCP ontology setting."""
        main(["mcp", "--ontology", "o.yaml"])

        assert seen[0].ontology == Path("o.yaml")

    def test_omitted_flag_keeps_the_environment_value(self, seen, monkeypatch):
        """A flag that is not given does not override the environment."""
        monkeypatch.setenv("MEMORY_MCP_TRANSPORT", "http")

        main(["mcp"])

        assert seen[0].transport == "http"

    def test_defaults_apply_when_nothing_is_set(self, seen):
        """With no flags and no variables the settings defaults are used."""
        main(["mcp"])

        settings = seen[0]
        assert (settings.transport, settings.host, settings.port, settings.tenant) == (
            "stdio",
            "127.0.0.1",
            8000,
            "default",
        )

    def test_unknown_transport_exits_with_status_two(self, seen, capsys):
        """A transport outside stdio and http is a usage error."""
        with pytest.raises(SystemExit) as excinfo:
            main(["mcp", "--transport", "smtp"])

        assert excinfo.value.code == 2
        assert "usage:" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["mcp", "--port", "70000"], id="port-too-high"),
            pytest.param(["mcp", "--port", "0"], id="port-zero"),
        ],
    )
    def test_invalid_settings_return_one_with_a_one_line_message(
        self, seen, capsys, argv
    ):
        """A rejected value gives status one and a message without the value."""
        status = main(argv)

        stderr = capsys.readouterr().err
        assert status == 1
        assert len(stderr.strip().splitlines()) == 1
        assert argv[-1] not in stderr
        assert seen == []

    def test_invalid_tenant_exits_with_the_validator_message(self, seen, capsys):
        """A malformed tenant is a usage error that shows the allowed format."""
        with pytest.raises(SystemExit) as excinfo:
            main(["mcp", "--tenant", "bad tenant!"])

        assert excinfo.value.code == 2
        assert (
            "1 to 60 letters, digits, dashes or underscores" in capsys.readouterr().err
        )
        assert seen == []

    def test_missing_extra_returns_one_with_the_install_hint(
        self, seen, monkeypatch, capsys
    ):
        """A missing FastMCP prints the install command and returns one."""
        monkeypatch.setattr(
            "ontogr_mem.cli.asyncio.run", _fake_asyncio_run(MissingExtraError(("mcp",)))
        )

        status = main(["mcp"])

        assert status == 1
        assert 'uv add "ontogr-mem[mcp]"' in capsys.readouterr().err
