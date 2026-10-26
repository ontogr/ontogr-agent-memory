"""Settings of the pgvector backend."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PgVectorSettings(BaseSettings):
    """Connection, search and outbox limits of the pgvector backend.

    Environment variables use the prefix `MEMORY_PGVECTOR_`.

    Attributes:
        dsn: Connection string of the Postgres database.
        min_size: Fewest connections kept open.
        max_size: Most connections kept open.
        idle_s: Close an idle connection after this many seconds.
        command_timeout_s: Time limit of a statement by default.
        search_timeout_s: Time limit of a search statement.
        bulk_timeout_s: Time limit of a write or a sweep statement.
        iterative_scan: pgvector iterative scan mode; `off` disables it.
        max_scan_tuples: Tuples an iterative scan may visit per query.
        exact_threshold: A narrowed search scans exactly below this row count.
        lease_s: How long a claim protects an outbox entry.
        max_attempts: Tries before an outbox entry becomes a dead letter.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_PGVECTOR_", env_file=".env", extra="ignore", frozen=True
    )

    dsn: str = "postgresql://memory:memorytest123@localhost:15432/memory"
    min_size: int = Field(default=1, ge=0)
    max_size: int = Field(default=10, ge=1)
    idle_s: float = Field(default=300.0, gt=0)
    command_timeout_s: float = Field(default=60.0, gt=0)
    search_timeout_s: float = Field(default=10.0, gt=0)
    bulk_timeout_s: float = Field(default=60.0, gt=0)
    iterative_scan: Literal["off", "strict_order", "relaxed_order"] = "strict_order"
    max_scan_tuples: int = Field(default=20000, ge=1)
    exact_threshold: int = Field(default=1000, ge=1)
    lease_s: float = Field(default=60.0, gt=0)
    max_attempts: int = Field(default=5, ge=1)
