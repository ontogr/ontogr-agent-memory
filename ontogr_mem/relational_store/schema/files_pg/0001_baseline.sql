CREATE TABLE IF NOT EXISTS jobs (
    seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
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
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    trace_id TEXT NOT NULL,
    progress_stage TEXT,
    progress_done INTEGER,
    progress_total INTEGER,
    claimed_by TEXT,
    claimed_at BIGINT,
    run_after BIGINT NOT NULL DEFAULT 0,
    UNIQUE (tenant_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, seq);
CREATE INDEX IF NOT EXISTS jobs_claim ON jobs (status, run_after);
CREATE TABLE IF NOT EXISTS erasures (
    seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
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
    created_at BIGINT NOT NULL,
    status TEXT NOT NULL,
    last_step TEXT NOT NULL,
    finished_at BIGINT,
    residue TEXT NOT NULL,
    counts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS erasures_status_tenant
    ON erasures (status, tenant, created_at, seq);
CREATE TABLE IF NOT EXISTS audit_rows (
    seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    recorded_at BIGINT NOT NULL,
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
  seq BIGINT GENERATED ALWAYS AS IDENTITY,
  id TEXT PRIMARY KEY,
  parent_id TEXT,
  kind TEXT NOT NULL,
  principal_id TEXT NOT NULL,
  tenant TEXT,
  session_id TEXT,
  trace_id TEXT NOT NULL,
  started_at BIGINT NOT NULL,
  ended_at BIGINT NOT NULL,
  outcome TEXT NOT NULL,
  error_class TEXT,
  error TEXT,
  tokens_in INTEGER NOT NULL DEFAULT 0,
  tokens_out INTEGER NOT NULL DEFAULT 0,
  cost_usd DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS operation_rows_tenant
ON operation_rows (tenant, started_at);
