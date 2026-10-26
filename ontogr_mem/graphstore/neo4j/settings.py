"""Settings for the Neo4j graph store."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from ontogr_mem.graphstore.config import ResilienceSettings


class Neo4jSettings(BaseSettings):
    """Connection, pool and resilience settings for one Neo4j server.

    Environment variables use the prefix `MEMORY_NEO4J_`; nested resilience
    fields use a double underscore, for example
    `MEMORY_NEO4J_RESILIENCE__MAX_ATTEMPTS`. Without a password the driver
    connects without authentication.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_NEO4J_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: SecretStr | None = None
    database: str = "neo4j"
    max_connections: int = Field(default=16, ge=1)
    pool_timeout: float = Field(default=30.0, gt=0)
    connect_timeout: float = Field(default=5.0, gt=0)
    transaction_retry_s: float = Field(default=5.0, ge=0)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
