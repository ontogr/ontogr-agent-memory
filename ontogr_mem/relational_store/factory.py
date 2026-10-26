"""Open state storage while enforcing the service durability policy."""

from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.relational_store.engine import pg_pool
from ontogr_mem.relational_store.jobs.errors import EphemeralStateError
from ontogr_mem.relational_store.jobs.ports import StateStore
from ontogr_mem.relational_store.memory import MemoryJobStore
from ontogr_mem.relational_store.postgres.jobs import PostgresJobStore
from ontogr_mem.relational_store.schema.runner import ensure_postgres_schema
from ontogr_mem.relational_store.sqlite import SqliteJobStore


def _memory_store(settings: StateSettings) -> StateStore:
    """Refuse volatile state unless the caller explicitly permits it."""
    if not settings.allow_ephemeral:
        raise EphemeralStateError(
            "use a sqlite path with backend='sqlite', or set "
            "MEMORY_STATE_ALLOW_EPHEMERAL=true to allow in-memory state"
        )
    return MemoryJobStore()


def _sqlite_store(settings: StateSettings) -> StateStore:
    """Open durable job and erasure state at the configured path."""
    return SqliteJobStore(settings.sqlite_path)


async def _postgres_store(settings: StateSettings) -> StateStore:
    """Open pooled job and erasure state with its schema ensured."""
    pool = await pg_pool(settings.dsn)
    await ensure_postgres_schema(pool)
    return PostgresJobStore(pool)


_SYNC_OPENERS = {"memory": _memory_store, "sqlite": _sqlite_store}


async def open_state_store(settings: StateSettings) -> StateStore:
    """Open the configured backend, refusing unapproved ephemeral storage.

    The Postgres backend takes a process-wide pool and applies pending
    schema migrations before the store is returned, so a database that
    failed its migrations blocks the caller instead of serving writes.

    Args:
        settings: Backend configuration and explicit ephemeral permission.

    Returns:
        An open store that the caller must close.

    Raises:
        EphemeralStateError: If memory storage was requested without permission.
    """
    if settings.backend == "postgres":
        return await _postgres_store(settings)
    return _SYNC_OPENERS[settings.backend](settings)
