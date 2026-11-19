"""Durability configuration for service state."""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ontogr_mem.paths import data_directory
from ontogr_mem.store_urls import parse_state_url, sqlite_url


class StateSettings(BaseSettings):
    """Configure the durable state store URL."""

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_STATE_", env_file=".env", extra="ignore", frozen=True
    )

    url: SecretStr = Field(
        default_factory=lambda: SecretStr(
            sqlite_url(data_directory() / "agent_memory_state.sqlite3")
        )
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: SecretStr) -> SecretStr:
        """Reject unsupported state store URLs."""
        parse_state_url(value.get_secret_value())
        return value
