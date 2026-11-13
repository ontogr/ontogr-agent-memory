"""Settings for the MCP server, read from MEMORY_MCP_ variables."""

import pytest
from pydantic import SecretStr, ValidationError

from ontogr_mem.mcp_server.settings import McpSettings


MCP_ENV_NAMES = (
    "MEMORY_MCP_TRANSPORT",
    "MEMORY_MCP_HOST",
    "MEMORY_MCP_PORT",
    "MEMORY_MCP_TENANT",
    "MEMORY_MCP_API_KEY",
    "MEMORY_MCP_TOOLS",
    "MEMORY_MCP_SKILLS_EXTENSION",
)

DEFAULT_TOOLS = (
    "search_memory",
    "get_entity",
    "get_job",
    "add_memory",
    "write_facts",
    "forget_memory",
    "get_skill",
)


@pytest.fixture(autouse=True)
def clear_mcp_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the MEMORY_MCP_ variables before each test."""
    for name in MCP_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


class TestMcpSettings:
    """Values and validation rules for the MCP server settings."""

    def test_defaults_serve_all_tools_over_stdio(self):
        """Without variables, the server uses stdio with every tool enabled."""
        settings = McpSettings(_env_file=None)

        assert settings.transport == "stdio"
        assert settings.host == "127.0.0.1"
        assert settings.port == 8000
        assert settings.tenant == "default"
        assert settings.api_key is None
        assert settings.tools == DEFAULT_TOOLS
        assert settings.skills_extension is True

    @pytest.mark.parametrize("transport", ["stdio", "http"])
    def test_accepts_stdio_and_http_transports(self, monkeypatch, transport):
        """The two supported transports are accepted as written."""
        monkeypatch.setenv("MEMORY_MCP_TRANSPORT", transport)

        settings = McpSettings(_env_file=None)

        assert settings.transport == transport

    def test_rejects_unknown_transport(self, monkeypatch):
        """A transport outside stdio and http fails validation."""
        monkeypatch.setenv("MEMORY_MCP_TRANSPORT", "smtp")

        with pytest.raises(ValidationError):
            McpSettings(_env_file=None)

    @pytest.mark.parametrize("port", ["0", "70000"])
    def test_rejects_port_outside_one_to_65535(self, monkeypatch, port):
        """Ports must be between 1 and 65535 inclusive."""
        monkeypatch.setenv("MEMORY_MCP_PORT", port)

        with pytest.raises(ValidationError):
            McpSettings(_env_file=None)

    @pytest.mark.parametrize("tenant", ["a.b", "x" * 61])
    def test_rejects_tenant_ids_that_tenant_rules_refuse(self, monkeypatch, tenant):
        """Tenant ids use the same rules as validate_tenant_id."""
        monkeypatch.setenv("MEMORY_MCP_TENANT", tenant)

        with pytest.raises(ValidationError):
            McpSettings(_env_file=None)

    def test_parses_comma_separated_tools_and_trims_whitespace(self, monkeypatch):
        """Spaces around each comma separated name are ignored."""
        monkeypatch.setenv("MEMORY_MCP_TOOLS", "search_memory, get_entity")

        settings = McpSettings(_env_file=None)

        assert settings.tools == ("search_memory", "get_entity")

    def test_removes_duplicate_tool_names_keeping_first_order(self, monkeypatch):
        """A repeated tool name is listed once, at its first position."""
        monkeypatch.setenv("MEMORY_MCP_TOOLS", "get_entity,search_memory,get_entity")

        settings = McpSettings(_env_file=None)

        assert settings.tools == ("get_entity", "search_memory")

    def test_empty_tools_variable_enables_no_tools(self, monkeypatch):
        """An empty MEMORY_MCP_TOOLS value gives an empty tuple."""
        monkeypatch.setenv("MEMORY_MCP_TOOLS", "")

        settings = McpSettings(_env_file=None)

        assert settings.tools == ()

    def test_constructor_accepts_an_empty_tools_tuple(self):
        """Passing tools=() directly enables no tools."""
        settings = McpSettings(_env_file=None, tools=())

        assert settings.tools == ()

    def test_rejects_unknown_tool_and_names_it_in_the_error(self, monkeypatch):
        """The validation message includes the unknown tool name."""
        monkeypatch.setenv("MEMORY_MCP_TOOLS", "search_memory,drop_database")

        with pytest.raises(ValidationError) as excinfo:
            McpSettings(_env_file=None)

        assert "drop_database" in excinfo.value.errors()[0]["msg"]

    def test_blank_api_key_is_treated_as_unset(self, monkeypatch):
        """An empty MEMORY_MCP_API_KEY means no bearer token is required."""
        monkeypatch.setenv("MEMORY_MCP_API_KEY", "")

        settings = McpSettings(_env_file=None)

        assert settings.api_key is None

    def test_api_key_is_secret_and_absent_from_repr(self, monkeypatch):
        """The key is stored as SecretStr and repr does not reveal it."""
        monkeypatch.setenv("MEMORY_MCP_API_KEY", "sentinel-key")

        settings = McpSettings(_env_file=None)

        assert isinstance(settings.api_key, SecretStr)
        assert settings.api_key.get_secret_value() == "sentinel-key"
        assert "sentinel-key" not in repr(settings)

    def test_skills_extension_can_be_disabled(self, monkeypatch):
        """MEMORY_MCP_SKILLS_EXTENSION=false turns the skills extension off."""
        monkeypatch.setenv("MEMORY_MCP_SKILLS_EXTENSION", "false")

        settings = McpSettings(_env_file=None)

        assert settings.skills_extension is False
