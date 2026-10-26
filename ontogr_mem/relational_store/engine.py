"""Cached connection access for the relational store."""

import sqlite3
from asyncio import CancelledError, Lock
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from typing import TYPE_CHECKING

from ontogr_mem.relational_store.schema.runner import ensure_sqlite_schema


if TYPE_CHECKING:
    import asyncpg


_PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA busy_timeout = 5000",
    "PRAGMA foreign_keys = ON",
)


def open_sqlite_connection(path: str, *, autocommit: bool) -> sqlite3.Connection:
    """Open one SQLite connection with the store's discipline applied.

    Every connection the package owns is opened here, so the pragmas hold on
    every path: WAL persists on the file, ``synchronous=NORMAL`` pairs with
    it, and a five-second busy window lets ``BEGIN IMMEDIATE`` writers queue
    instead of failing. The file must live on a local filesystem; WAL over a
    network share is undefined.

    Args:
        path: Database file path; its parent directory must already exist.
        autocommit: True for single-statement callers such as the shared
            operation and audit connection; False for the transactional
            connection the job store guards with its thread lock.

    Returns:
        An open connection with all pragmas applied.
    """
    connection = sqlite3.connect(
        path,
        check_same_thread=False,
        isolation_level=None if autocommit else "DEFERRED",
    )
    try:
        for pragma in _PRAGMAS:
            connection.execute(pragma)
    except BaseException:
        connection.close()
        raise
    return connection


@lru_cache(maxsize=None)
def sqlite_shared_connection(path: str) -> sqlite3.Connection:
    """Return the process-wide autocommit connection for one database file.

    The connection is a singleton for the process lifetime, so callers must
    not close it; schema state is ensured once per acquisition through the
    migration runner.
    """
    connection = open_sqlite_connection(path, autocommit=True)
    try:
        ensure_sqlite_schema(connection)
    except BaseException:
        connection.close()
        sqlite_shared_connection.cache_clear()
        raise
    return connection


_pg_pools: dict[str, "asyncpg.Pool"] = {}
_pg_pool_locks: dict[str, Lock] = {}


async def pg_pool(dsn: str) -> "asyncpg.Pool":
    """Return the process-wide connection pool for one Postgres DSN.

    The pool is a singleton for the process lifetime, so callers must not
    close it; ``asyncpg`` is imported lazily through the optional-extra seam.
    """
    from ontogr_mem._extras import import_extra

    asyncpg = import_extra("asyncpg", "postgres")
    pool = _pg_pools.get(dsn)
    if pool is not None:
        return pool
    # Creating a pool awaits, so the check-then-create needs a lock; it is per
    # DSN so unrelated databases do not wait on each other.
    async with _pg_pool_locks.setdefault(dsn, Lock()):
        pool = _pg_pools.get(dsn)
        if pool is None:
            pool = await asyncpg.create_pool(dsn)
            _pg_pools[dsn] = pool
    return pool


@asynccontextmanager
async def pg_connection(
    pool: "asyncpg.Pool",
) -> AsyncGenerator["asyncpg.Connection", None]:
    """Lend one pooled connection, discarding it when a call is cancelled.

    A statement cancelled mid-flight can leave the connection in an unknown
    state, so the connection is closed instead of returned to the pool.
    """
    connection = await pool.acquire()
    try:
        yield connection
    except CancelledError:
        with suppress(Exception):
            await connection.close()
        raise
    finally:
        if not connection.is_closed():
            await pool.release(connection)
