"""Lockstep SQL migrations with checksum drift detection."""

import hashlib
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ontogr_mem.data_models.timestamps import to_micros
from ontogr_mem.errors import AgentMemoryError


if TYPE_CHECKING:
    import asyncpg


SCHEMA_FILES = Path(__file__).parent

SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    revision INTEGER PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at BIGINT NOT NULL
);
"""

_apply_lock = threading.Lock()


class SchemaDriftError(AgentMemoryError):
    """An applied revision file no longer matches its recorded checksum."""


def revision_checksum(sql: str) -> str:
    """Return the stable checksum recorded for one revision file."""
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def pending_revisions(files: Path) -> list[tuple[int, str, str]]:
    """Return ordered revisions as (revision, sql, checksum) triples."""
    pending = []
    for path in sorted(files.glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        pending.append((int(path.name.split("_", 1)[0]), sql, revision_checksum(sql)))
    return pending


def ensure_sqlite_schema(connection: sqlite3.Connection) -> None:
    """Apply pending SQLite revisions and verify every applied checksum.

    Each call creates the migration table if needed, checks applied revisions
    against the shipped files, and applies pending revisions. Pending revisions
    run inside one serialized section, so concurrent opens in one process cannot
    apply the same revision twice. Existing database files that predate the
    migration table also run the baseline, which is safe because it is written
    as ``CREATE TABLE IF NOT EXISTS``.

    Raises:
        SchemaDriftError: When a recorded checksum differs from its file.
    """
    with _apply_lock, connection:
        connection.executescript(SCHEMA_MIGRATIONS_DDL)
        applied = dict(
            connection.execute("SELECT revision, checksum FROM schema_migrations")
        )
        for revision, sql, checksum in pending_revisions(SCHEMA_FILES / "files_sqlite"):
            recorded = applied.get(revision)
            if recorded is not None:
                if recorded != checksum:
                    raise SchemaDriftError(
                        f"schema revision {revision} no longer matches the "
                        "revision applied to this database"
                    )
                continue
            connection.executescript(sql)
            connection.execute(
                "INSERT INTO schema_migrations (revision, checksum, applied_at) "
                "VALUES (?, ?, ?)",
                (revision, checksum, to_micros(datetime.now(UTC))),
            )


_MIGRATION_LOCK_KEY = 90211


async def ensure_postgres_schema(pool: "asyncpg.Pool") -> None:
    """Apply pending Postgres revisions and verify every applied checksum.

    The runner is shared with SQLite: the same migration table, the same
    per-file checksums, and lockstep revision numbers, so a revision ships
    for both engines or neither. The whole check runs inside one transaction
    under a Postgres advisory lock, so concurrent processes cannot apply the
    same revision twice. Each statement runs separately because the driver
    executes one statement per call.

    Raises:
        SchemaDriftError: When a recorded checksum differs from its file.
    """
    from ontogr_mem.relational_store.engine import pg_connection

    async with pg_connection(pool) as connection, connection.transaction():
        await connection.execute(
            "SELECT pg_advisory_xact_lock($1)", _MIGRATION_LOCK_KEY
        )
        await connection.execute(SCHEMA_MIGRATIONS_DDL)
        applied = {
            row["revision"]: row["checksum"]
            for row in await connection.fetch(
                "SELECT revision, checksum FROM schema_migrations"
            )
        }
        for revision, sql, checksum in pending_revisions(SCHEMA_FILES / "files_pg"):
            recorded = applied.get(revision)
            if recorded is not None:
                if recorded != checksum:
                    raise SchemaDriftError(
                        f"schema revision {revision} no longer matches the "
                        "revision applied to this database"
                    )
                continue
            for statement in sql.split(";"):
                if statement.strip():
                    await connection.execute(statement)
            await connection.execute(
                "INSERT INTO schema_migrations (revision, checksum, "
                "applied_at) VALUES ($1, $2, $3)",
                revision,
                checksum,
                to_micros(datetime.now(UTC)),
            )
