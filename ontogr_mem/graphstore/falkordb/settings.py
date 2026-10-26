"""Settings for the FalkorDB graph store."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from ontogr_mem.graphstore.config import ResilienceSettings


class FalkorDBSettings(BaseSettings):
    """Connection, pool and resilience settings for one FalkorDB server.

    Environment variables use the prefix `MEMORY_FALKORDB_`; nested resilience
    fields use a double underscore, for example
    `MEMORY_FALKORDB_RESILIENCE__MAX_ATTEMPTS`.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_FALKORDB_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    username: str | None = None
    password: SecretStr | None = None
    tls: bool = False
    max_connections: int = Field(default=16, ge=1)
    pool_timeout: float = Field(default=30.0, gt=0)
    socket_connect_timeout: float = Field(default=5.0, gt=0)
    socket_timeout: float = Field(default=30.0, gt=0)
    health_check_interval: int = Field(default=30, ge=0)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
