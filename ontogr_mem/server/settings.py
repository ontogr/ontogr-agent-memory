"""Settings for the memory server process."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Configure server lifecycle behavior from MEMORY_SERVER_ variables."""

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_SERVER_", env_file=".env", extra="ignore", frozen=True
    )

    idle_minutes: float = Field(default=30, ge=0)
