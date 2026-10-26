"""Durability configuration for service state."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class StateSettings(BaseSettings):
    """Select the service state backend and its durability policy.

    Attributes:
        backend: Storage backend. Env: `MEMORY_STATE_BACKEND`.
        sqlite_path: SQLite file path. Env: `MEMORY_STATE_SQLITE_PATH`.
        dsn: Postgres DSN when the backend is `postgres`.
            Env: `MEMORY_STATE_DSN`.
        allow_ephemeral: Permit volatile state. Env: `MEMORY_STATE_ALLOW_EPHEMERAL`.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_STATE_", env_file=".env", extra="ignore", frozen=True
    )

    backend: Literal["sqlite", "memory", "postgres"] = "sqlite"
    sqlite_path: str = "agent_memory_state.sqlite3"
    dsn: str = "postgresql://localhost/ontogr_mem"
    allow_ephemeral: bool = False
