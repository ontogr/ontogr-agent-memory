"""Connection pool and graph handles for one FalkorDB server."""

import asyncio
from typing import Any

from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool, SSLConnection
from redis.exceptions import ResponseError

from ontogr_mem.graphstore.falkordb.runner import GraphLike, QueryRunner
from ontogr_mem.graphstore.falkordb.settings import FalkorDBSettings


class FalkorClientPool:
    """One redis pool and one runner shared by every tenant graph."""

    def __init__(self, settings: FalkorDBSettings) -> None:
        """Store settings; the connection is created on first use."""
        self.settings = settings
        self._pool: BlockingConnectionPool | None = None
        self._db: FalkorDB | None = None
        self._create_lock = asyncio.Lock()
        self.runner = QueryRunner(self.graph, settings.resilience)

    async def _ensure_db(self) -> FalkorDB:
        """Create the client on first use, without blocking the event loop.

        The falkordb client sends a synchronous `INFO` command when it is built,
        so it is built in a worker thread. A connection failure here surfaces
        inside the caller's retry policy.
        """
        if self._db is not None:
            return self._db
        async with self._create_lock:
            if self._db is not None:
                return self._db
            pool = BlockingConnectionPool(**self._pool_kwargs())
            try:
                self._db = await asyncio.to_thread(FalkorDB, connection_pool=pool)
            except BaseException:
                await pool.aclose()
                raise
            self._pool = pool
            return self._db

    def _pool_kwargs(self) -> dict[str, Any]:
        s = self.settings
        kwargs: dict[str, Any] = {
            "host": s.host,
            "port": s.port,
            "username": s.username,
            "password": s.password.get_secret_value() if s.password else None,
            "max_connections": s.max_connections,
            "timeout": s.pool_timeout,
            "socket_connect_timeout": s.socket_connect_timeout,
            "socket_timeout": s.socket_timeout,
            "health_check_interval": s.health_check_interval,
            "decode_responses": True,
        }
        if s.tls:
            kwargs["connection_class"] = SSLConnection
        return kwargs

    async def graph(self, name: str) -> GraphLike:
        """Return a handle for graph `name`; handles are cheap."""
        graph: GraphLike = (await self._ensure_db()).select_graph(name)
        return graph

    async def ping(self) -> bool:
        """Send PING through the runner."""

        async def call() -> bool:
            db = await self._ensure_db()
            return bool(await db.connection.ping())

        return await self.runner.execute(call, idempotent=True, tag="ping")

    async def delete_graph(self, name: str) -> bool:
        """Delete graph `name`; a missing graph counts as deleted.

        Returns:
            True if a graph was deleted, False if it did not exist.
        """

        async def call() -> bool:
            graph: Any = await self.graph(name)
            try:
                await graph.delete()
            except ResponseError as exc:
                if "empty key" in str(exc).lower():
                    return False
                raise
            return True

        return await self.runner.execute(call, idempotent=True, tag="delete_graph")

    async def close(self) -> None:
        """Release the pool."""
        if self._pool is not None:
            await self._pool.aclose()
        self._db = None
        self._pool = None
