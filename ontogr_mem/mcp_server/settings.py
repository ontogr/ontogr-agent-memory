"""Settings for one MCP server process."""

from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from ontogr_mem.data_models.tenant import validate_tenant_id


ALL_TOOLS = (
    "search_memory",
    "get_entity",
    "get_job",
    "add_memory",
    "write_facts",
    "forget_memory",
    "get_skill",
)


class McpSettings(BaseSettings):
    """Configure one MCP process from MEMORY_MCP_ variables.

    Attributes:
        transport: Transport the server serves, either stdio or http. Env:
            MEMORY_MCP_TRANSPORT.
        host: Interface the http transport binds to. Env: MEMORY_MCP_HOST.
        port: Port the http transport binds to, from 1 to 65535. Env:
            MEMORY_MCP_PORT.
        tenant: Tenant id the server reads and writes. Env: MEMORY_MCP_TENANT.
        api_key: Bearer token http clients must send. None disables the check.
            A blank value counts as None. Env: MEMORY_MCP_API_KEY.
        tools: Names of the tools to register. The registration order is fixed. Env:
            MEMORY_MCP_TOOLS as a comma separated list.
        skills_extension: Whether the server advertises the skills extension.
            Env: MEMORY_MCP_SKILLS_EXTENSION.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_MCP_", env_file=".env", extra="ignore", frozen=True
    )

    transport: Literal["stdio", "http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    tenant: str = "default"
    api_key: SecretStr | None = None
    tools: Annotated[tuple[str, ...], NoDecode] = ALL_TOOLS
    skills_extension: bool = True

    @field_validator("tenant")
    @classmethod
    def _tenant(cls, value: str) -> str:
        """Apply the tenant id rules."""
        return validate_tenant_id(value)

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_key(cls, value: object) -> object:
        """Map a blank key to None so an empty variable disables the check."""
        return None if value == "" else value

    @field_validator("tools", mode="before")
    @classmethod
    def _tools(cls, value: object) -> tuple[str, ...]:
        """Parse the tool list and reject names outside ALL_TOOLS."""
        if isinstance(value, str):
            names = tuple(part.strip() for part in value.split(",") if part.strip())
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, str) for item in value
        ):
            names = tuple(value)
        else:
            raise ValueError("tools must be a comma list or sequence of names")
        unknown = set(names) - set(ALL_TOOLS)
        if unknown:
            raise ValueError(f"unknown MCP tools: {', '.join(sorted(unknown))}")
        return tuple(dict.fromkeys(names))
