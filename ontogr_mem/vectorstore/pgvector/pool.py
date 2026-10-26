"""Connection pool behind a small Protocol. Connections carry no tenant state."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Protocol

from ontogr_mem._extras import import_extra
from ontogr_mem.vectorstore.pgvector.settings import PgVectorSettings


class Conn(Protocol):
    """The subset of an asyncpg connection this package uses."""

    async def execute(
        self, query: str, *args: Any, timeout: float | None = None
    ) -> str:
        """Run a statement and return its status string."""
        ...

    async def fetch(
        self, query: str, *args: Any, timeout: float | None = None
    ) -> list[Any]:
        """Run a query and return all rows."""
        ...

    async def fetchval(
        self, query: str, *args: Any, timeout: float | None = None
    ) -> Any:
        """Run a query and return the first column of the first row."""
        ...

    def transaction(
        self, *, readonly: bool = False
    ) -> AbstractAsyncContextManager[Any]:
        """Open a transaction."""
        ...

    def terminate(self) -> None:
        """Close the connection without waiting."""
        ...


class Pool(Protocol):
    """Acquire connections; tests can supply a fake."""

    def acquire(self) -> AbstractAsyncContextManager[Conn]:
        """Borrow a connection."""
        ...

    async def close(self) -> None:
        """Close the pool."""
        ...


class AsyncpgPool:
    """An `asyncpg` pool that terminates a cancelled borrower's connection."""

    def __init__(self, pool: Any, settings: PgVectorSettings) -> None:
        """Wrap an `asyncpg` pool and keep its settings."""
        self._pool = pool
        self.settings = settings

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Conn]:
        """Borrow a connection; terminate it if the borrower is cancelled."""
        conn = await self._pool.acquire()
        try:
            yield conn
        except asyncio.CancelledError:
            conn.terminate()
            raise
        finally:
            await self._pool.release(conn)

    async def close(self) -> None:
        """Close all connections."""
        await self._pool.close()


async def create_pool(settings: PgVectorSettings) -> AsyncpgPool:
    """Create an `asyncpg` pool from `settings`.

    The optional driver is imported here, so a base install never loads it.

    Raises:
        MissingExtraError: If `asyncpg` is not installed.
    """
    asyncpg = import_extra("asyncpg", "pgvector")
    pool = await asyncpg.create_pool(
        settings.dsn,
        min_size=settings.min_size,
        max_size=settings.max_size,
        max_inactive_connection_lifetime=settings.idle_s,
        command_timeout=settings.command_timeout_s,
    )
    return AsyncpgPool(pool, settings)
