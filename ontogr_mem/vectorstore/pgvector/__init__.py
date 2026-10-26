"""The optional pgvector backend: Postgres tables, outbox and commit hook."""

from ontogr_mem.vectorstore.pgvector.commit_hook import (
    CommittedFact,
    enqueue_commit,
    entries_for_commit,
)
from ontogr_mem.vectorstore.pgvector.errors import (
    InvalidIdentifierError,
    RecallBelowThresholdError,
    TenantNotProvisionedError,
)
from ontogr_mem.vectorstore.pgvector.outbox import (
    InMemoryOutbox,
    Outbox,
    OutboxEntry,
    OutboxOp,
    PgOutbox,
    delete_entry,
    upsert_entry,
    validity_entry,
)
from ontogr_mem.vectorstore.pgvector.pool import AsyncpgPool, Pool, create_pool
from ontogr_mem.vectorstore.pgvector.provision import (
    drop_tenant,
    ensure_indexes,
    provision,
)
from ontogr_mem.vectorstore.pgvector.recall import (
    RecallReport,
    recall_selftest,
    require_recall,
)
from ontogr_mem.vectorstore.pgvector.reconcile import (
    DrainReport,
    ReconcileConfig,
    Reconciler,
    SweepReport,
    episode_ready,
)
from ontogr_mem.vectorstore.pgvector.schema import (
    HnswParams,
    build_manifest,
    schema_name,
)
from ontogr_mem.vectorstore.pgvector.settings import PgVectorSettings
from ontogr_mem.vectorstore.pgvector.store import PgBackend
from ontogr_mem.vectorstore.protocol import GraphView


__all__ = [
    "AsyncpgPool",
    "CommittedFact",
    "DrainReport",
    "GraphView",
    "HnswParams",
    "InMemoryOutbox",
    "InvalidIdentifierError",
    "Outbox",
    "OutboxEntry",
    "OutboxOp",
    "PgBackend",
    "PgOutbox",
    "PgVectorSettings",
    "Pool",
    "RecallBelowThresholdError",
    "RecallReport",
    "ReconcileConfig",
    "Reconciler",
    "SweepReport",
    "TenantNotProvisionedError",
    "build_manifest",
    "create_pool",
    "delete_entry",
    "drop_tenant",
    "ensure_indexes",
    "entries_for_commit",
    "enqueue_commit",
    "episode_ready",
    "provision",
    "recall_selftest",
    "require_recall",
    "schema_name",
    "upsert_entry",
    "validity_entry",
]
