"""Durability configuration for service state."""

from math import isfinite

from pydantic import Field, SecretStr, field_validator, model_validator
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
    lease_seconds: float = 60.0
    heartbeat_seconds: float = 15.0

    @model_validator(mode="after")
    def validate_lease_settings(self) -> "StateSettings":
        """Require positive lease timings and a heartbeat within one third."""
        if (
            not isfinite(self.lease_seconds)
            or not isfinite(self.heartbeat_seconds)
            or self.lease_seconds <= 0
            or self.heartbeat_seconds <= 0
        ):
            raise ValueError("lease and heartbeat seconds must be positive")
        if self.heartbeat_seconds > self.lease_seconds / 3:
            raise ValueError(
                "heartbeat seconds must not exceed one third of lease seconds"
            )
        return self

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: SecretStr) -> SecretStr:
        """Reject unsupported state store URLs."""
        parse_state_url(value.get_secret_value())
        return value
