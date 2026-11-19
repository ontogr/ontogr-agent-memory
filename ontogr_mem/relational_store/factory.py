"""Open state storage while enforcing the service durability policy."""

from pathlib import Path

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.paths import ensure_directory
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.relational_store.engine import pg_pool
from ontogr_mem.relational_store.jobs.ports import StateStore
from ontogr_mem.relational_store.postgres.jobs import PostgresJobStore
from ontogr_mem.relational_store.schema.runner import ensure_postgres_schema
from ontogr_mem.relational_store.sqlite import SqliteJobStore
from ontogr_mem.store_urls import (
    PostgresTarget,
    SqliteTarget,
    parse_state_url,
)


def _sqlite_store(target: SqliteTarget) -> StateStore:
    """Open durable job and erasure state at the configured path."""
    path = Path(target.path).expanduser()
    ensure_directory(path.parent)
    return SqliteJobStore(str(path))


async def _postgres_store(
    target: PostgresTarget, clock: Clock, *, owner_id: str, lease_seconds: float
) -> StateStore:
    """Open pooled job and erasure state with its schema ensured."""
    pool = await pg_pool(target.dsn.get_secret_value())
    await ensure_postgres_schema(pool)
    return PostgresJobStore(pool, clock, owner_id=owner_id, lease_seconds=lease_seconds)


async def open_state_store(
    settings: StateSettings, *, owner_id: str, clock: Clock
) -> StateStore:
    """Open the durable backend selected by the configured state URL.

    A SQLite URL opens a file at the configured path. A Postgres URL takes a
    process-wide pool and applies pending schema migrations before the store is
    returned, so a database that failed its migrations blocks the caller
    instead of serving writes.

    Args:
        settings: Configuration carrying the state store URL.
        owner_id: Identifier shared by stores owned by one service process.
        clock: Service clock used for lease checks.

    Returns:
        An open store that the caller must close.

    Raises:
        StoreUrlError: If the configured state URL is unsupported.
    """
    target = parse_state_url(settings.url.get_secret_value())
    if isinstance(target, PostgresTarget):
        return await _postgres_store(
            target,
            clock,
            owner_id=owner_id,
            lease_seconds=settings.lease_seconds,
        )
    return _sqlite_store(target)
