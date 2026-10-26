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
