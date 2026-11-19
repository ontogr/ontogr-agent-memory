"""The MCP proxy and the ontogr-mem mcp command."""

from ontogr_mem.cli import main
from ontogr_mem.errors import MissingExtraError


class TestMcpCommand:
    """Behavior of the mcp subcommand."""

    def test_starts_stdio_proxy_for_configured_endpoint(self, monkeypatch):
        """The CLI forwards stdio to the shared endpoint with auth headers."""
        calls = {}

        class Endpoint:
            url = "http://127.0.0.1:8000"
            token = "secret"

        class Proxy:
            async def run_async(self, **kwargs):
                calls["run"] = kwargs

        class Client:
            def __init__(self, url, *, headers):
                calls["client"] = (url, headers)

        # Imports happen inside run_mcp, so replace its dependency boundaries there.
        async def ensure_server(**kwargs):
            calls["ensure"] = kwargs
            return Endpoint()

        def create_proxy(client, *, instructions):
            calls["instructions"] = instructions
            return Proxy()

        monkeypatch.setattr("ontogr_mem.server.entry.ensure_server", ensure_server)
        monkeypatch.setattr("fastmcp.Client", Client)
        monkeypatch.setattr("fastmcp.server.server.create_proxy", create_proxy)
        monkeypatch.setenv("MEMORY_MCP_TENANT", "acme")

        assert main(["mcp"]) == 0
        assert calls["client"] == (
            "http://127.0.0.1:8000/mcp",
            {"Authorization": "Bearer secret", "X-Memory-Tenant": "acme"},
        )
        assert calls["run"] == {"transport": "stdio", "show_banner": False}

    def test_missing_extra_returns_one_with_install_hint(self, monkeypatch, capsys):
        """A missing MCP extra prints its install command and returns one."""

        async def fail():
            raise MissingExtraError(("mcp",))

        monkeypatch.setattr("ontogr_mem.cli.mcp.run_mcp", fail)
        monkeypatch.setattr("ontogr_mem.cli.asyncio.run", _raise_from_asyncio)

        assert main(["mcp"]) == 1
        assert 'uv add "ontogr-mem[mcp]"' in capsys.readouterr().err


def _raise_from_asyncio(coroutine):
    try:
        coroutine.send(None)
    except MissingExtraError:
        raise
    finally:
        coroutine.close()
