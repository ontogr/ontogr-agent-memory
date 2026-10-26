"""Existing database files absorb the migration baseline unchanged."""

import sqlite3

import pytest

from ontogr_mem.data_models.jobs import JobState
from ontogr_mem.relational_store.engine import sqlite_shared_connection
from ontogr_mem.relational_store.schema.runner import SchemaDriftError
from ontogr_mem.relational_store.sqlite.audit import SqliteAuditLog
from ontogr_mem.relational_store.sqlite.jobs import SqliteJobStore
from ontogr_mem.relational_store.sqlite.operations import SqliteOperationStore
from tests.support.clock import FixedClock


LEGACY_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    tenant_id TEXT NOT NULL,
    op TEXT NOT NULL,
    payload TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    idempotency_key TEXT,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    result TEXT,
    error TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    trace_id TEXT NOT NULL,
    progress_stage TEXT,
    progress_done INTEGER,
    progress_total INTEGER,
    UNIQUE (tenant_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, seq);
CREATE TABLE IF NOT EXISTS erasures (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    erasure_id TEXT NOT NULL UNIQUE,
    tenant TEXT NOT NULL,
    mode TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    selector_kind TEXT NOT NULL,
    target TEXT NOT NULL,
    episode_ids TEXT NOT NULL,
    fact_ids TEXT NOT NULL,
    observation_ids TEXT NOT NULL,
    entity_ids TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    status TEXT NOT NULL,
    last_step TEXT NOT NULL,
    finished_at INTEGER,
    residue TEXT NOT NULL,
    counts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS erasures_status_tenant
    ON erasures (status, tenant, created_at, seq);
CREATE TABLE IF NOT EXISTS audit_rows (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at INTEGER NOT NULL,
    action TEXT NOT NULL,
    outcome TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    tenant TEXT,
    acting_for TEXT,
    target TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    detail TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_rows_tenant ON audit_rows (tenant, seq);
CREATE TABLE IF NOT EXISTS operation_rows (
  id TEXT PRIMARY KEY,
  parent_id TEXT,
  kind TEXT NOT NULL,
  principal_id TEXT NOT NULL,
  tenant TEXT,
  session_id TEXT,
  trace_id TEXT NOT NULL,
  started_at INTEGER NOT NULL,
  ended_at INTEGER NOT NULL,
  outcome TEXT NOT NULL,
  error_class TEXT,
  error TEXT,
  tokens_in INTEGER NOT NULL DEFAULT 0,
  tokens_out INTEGER NOT NULL DEFAULT 0,
  cost_usd REAL
);
CREATE INDEX IF NOT EXISTS operation_rows_tenant
ON operation_rows (tenant, started_at);
"""

NOW_MICROS = 1761350400000000


def _legacy_database(path):
    """Build one file exactly as the pre-migration adapters left it."""
    connection = sqlite3.connect(path)
    connection.executescript(LEGACY_SCHEMA)
    connection.execute(
        "INSERT INTO jobs (job_id, tenant_id, op, payload, request_hash, status, "
        "attempts, created_at, updated_at, trace_id) "
        "VALUES ('job-legacy', 'acme', 'add_memory', '{}', 'hash-1', 'done', 1, "
        f"{NOW_MICROS}, {NOW_MICROS}, 'trace-1')"
    )
    connection.execute(
        "INSERT INTO audit_rows (recorded_at, action, outcome, principal_id, tenant, "
        "target, trace_id, detail, previous_hash, row_hash) "
        "VALUES (?, 'erase', 'ok', 'owner', 'acme', 'facts', 'trace-1', '{}', "
        "'legacy-previous', 'legacy-hash')",
        (NOW_MICROS,),
    )
    connection.execute(
        "INSERT INTO operation_rows (id, kind, principal_id, tenant, trace_id, "
        "started_at, ended_at, outcome, tokens_in, tokens_out, cost_usd) "
        "VALUES ('op-legacy', 'write', 'owner', 'acme', 'trace-1', "
        f"{NOW_MICROS}, {NOW_MICROS}, 'ok', 3, 5, 0.25)"
    )
    connection.commit()
    connection.close()
    return path


class TestSchemaMigrationCompat:
    """A database written before the runner opens on it without changes."""

    async def test_legacy_file_absorbs_baseline_and_keeps_rows(self, tmp_path):
        """The baseline records itself and every legacy row stays readable."""
        path = str(tmp_path / "legacy.sqlite3")
        _legacy_database(path)

        store = SqliteJobStore(path)
        try:
            legacy = await store.get_job("job-legacy")
            assert legacy is not None
            assert legacy.tenant_id == "acme"
            assert legacy.status is JobState.DONE
        finally:
            await store.close()

        connection = sqlite3.connect(path)
        try:
            applied = connection.execute(
                "SELECT revision, checksum FROM schema_migrations"
            ).fetchall()
            assert [revision for revision, _ in applied] == [1]
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        finally:
            connection.close()
        assert "schema_migrations" in tables
        assert "jobs" in tables and "erasures" in tables

        shared = sqlite_shared_connection(path)
        operations = SqliteOperationStore(shared, FixedClock())
        rows = await operations.list_operations(tenant="acme")
        assert [row.id for row in rows] == ["op-legacy"]
        usage = await operations.usage("acme")
        assert (usage.operations, usage.tokens_in, usage.tokens_out) == (1, 3, 5)

        audit = SqliteAuditLog(shared, FixedClock())
        listed = await audit.list_rows(tenant="acme")
        assert len(listed) == 1
        assert listed[0].entry.principal_id == "owner"
        assert dict(listed[0].entry.detail) == {}
        assert await audit.verify() == listed[0].seq

    async def test_tampered_checksum_blocks_the_store(self, tmp_path):
        """A rewritten revision file is refused instead of silently applied."""
        path = str(tmp_path / "drift.sqlite3")
        _legacy_database(path)
        store = SqliteJobStore(path)
        await store.close()

        connection = sqlite3.connect(path)
        connection.execute("UPDATE schema_migrations SET checksum = 'tampered'")
        connection.commit()
        connection.close()

        with pytest.raises(SchemaDriftError):
            SqliteJobStore(path)
