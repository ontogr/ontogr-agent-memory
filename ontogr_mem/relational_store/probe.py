"""Read-only probes for relational state backends."""

from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.relational_store.engine import pg_pool
from ontogr_mem.store_urls import PostgresTarget, parse_state_url


async def probe_state(settings: StateSettings) -> bool:
    """Check PostgreSQL connectivity with a read-only SELECT 1."""
    target = parse_state_url(settings.url.get_secret_value())
    if not isinstance(target, PostgresTarget):
        return True
    pool = await pg_pool(target.dsn.get_secret_value())
    async with pool.acquire() as connection:
        await connection.fetchval("SELECT 1")
    return True
