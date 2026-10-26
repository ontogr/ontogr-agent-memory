"""Build atomic graph statements for temporal slot commits."""

from typing import Any
from uuid import uuid4

from ontogr_mem.data_models.timestamps import to_micros
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.reserved import LOCK
from ontogr_mem.temporal.ledger import GENESIS
from ontogr_mem.temporal.temporal_store import SlotCommit

from .rows import (
    _conflict_props,
    _fact_props,
    _ledger_props,
    _supersession_props,
)
from .scope import (
    CONFLICT,
    ENTITY,
    EXPIRED_MATCH,
    FACT,
    LEDGER_ENTRY,
    SLOT,
    SUPERSESSION,
)


def _slot_guard(scope: TenantScope, params: dict[str, Any]) -> list[str]:
    """Build the revision guard and add the writer token when locking is needed."""
    guard_lines = [
        "FOREACH (_ IN CASE WHEN $base = 0 THEN [1] ELSE [] END | "
        f"MERGE (s:{SLOT} {scope.pattern('id: $slot')}) "
        "ON CREATE SET s.revision = 0, s.ledger_seq = 0, "
        "s.ledger_head = $genesis)",
        "WITH 1 AS slot_ready",
        f"MATCH (s:{SLOT} {scope.pattern('id: $slot')})",
    ]
    if scope.dialect.keyed:
        params["writer"] = uuid4().hex
        guard_lines += [
            f"WITH s, s.{LOCK} AS previous",
            f"SET s.{LOCK} = $writer",
            "WITH s, previous, s.revision = $base AS ok",
            f"SET s.{LOCK} = previous",
            "WITH s WHERE ok",
        ]
    else:
        guard_lines.append("WITH s WHERE s.revision = $base")
    return guard_lines


def _preconditions(
    commit: SlotCommit, scope: TenantScope, params: dict[str, Any]
) -> list[str]:
    """Check endpoints and expired versions before the slot guard can write."""
    lines = ["WITH 1 AS ready"]
    entity_ids = list(commit.entity_ids)
    if entity_ids:
        params["entity_ids"] = entity_ids
        params["entity_count"] = len(entity_ids)
        lines += [
            "UNWIND $entity_ids AS eid",
            f"OPTIONAL MATCH (e:{ENTITY} {scope.pattern('id: eid')})",
            "WITH ready, count(e) AS found",
            "WHERE found = $entity_count",
        ]
    if commit.expired_facts:
        params["expired"] = [
            {
                "fact_id": fact.fact_id,
                "version": fact.version,
                "expired_at": to_micros(fact.expired_at),
            }
            for fact in commit.expired_facts
        ]
        params["expired_count"] = len(commit.expired_facts)
        lines += [
            "UNWIND $expired AS x",
            f"OPTIONAL MATCH ()-[f:{FACT} {scope.pattern(EXPIRED_MATCH)}]->()",
            "WITH ready, count(f) AS found",
            "WHERE found = $expired_count",
        ]
    return lines


def _fact_writes(
    commit: SlotCommit, scope: TenantScope, tenant: str, params: dict[str, Any]
) -> list[str]:
    """Build fact expiry and creation clauses and add the new fact properties."""
    lines: list[str] = []
    if commit.expired_facts:
        lines += [
            "WITH s",
            "UNWIND $expired AS x",
            f"MATCH ()-[f:{FACT} {scope.pattern(EXPIRED_MATCH)}]->()",
            "SET f.expired_at = x.expired_at",
        ]
    if commit.new_facts:
        params["new"] = [
            {
                "subject_id": fact.subject_id,
                "object_id": fact.object_id,
                "props": scope.stamp(_fact_props(fact), tenant),
            }
            for fact in commit.new_facts
        ]
        lines += [
            "WITH DISTINCT s",
            "UNWIND $new AS x",
            f"MATCH (a:{ENTITY} {scope.pattern('id: x.subject_id')})",
            f"MATCH (b:{ENTITY} {scope.pattern('id: x.object_id')})",
            f"CREATE (a)-[f:{FACT}]->(b)",
            "SET f += x.props",
        ]
    return lines


def _audit_writes(
    commit: SlotCommit, scope: TenantScope, tenant: str, params: dict[str, Any]
) -> list[str]:
    """Build conflict replacement and supersession clauses with their parameters."""
    lines: list[str] = []
    if commit.conflicts_changed:
        lines += [
            "WITH DISTINCT s",
            f"OPTIONAL MATCH (old:{CONFLICT} {scope.pattern('slot_key: $slot')})",
            "DETACH DELETE old",
        ]
        if commit.conflicts:
            params["conflicts"] = [
                scope.stamp(_conflict_props(conflict), tenant)
                for conflict in commit.conflicts
            ]
            lines += [
                "WITH DISTINCT s",
                "UNWIND $conflicts AS x",
                f"CREATE (c:{CONFLICT})",
                "SET c += x",
            ]
    if commit.supersessions:
        params["supersessions"] = [
            scope.stamp(_supersession_props(row), tenant)
            for row in commit.supersessions
        ]
        lines += [
            "WITH DISTINCT s",
            "UNWIND $supersessions AS x",
            f"CREATE (n:{SUPERSESSION})",
            "SET n += x",
        ]
    return lines


def _ledger_writes(
    commit: SlotCommit, scope: TenantScope, tenant: str, params: dict[str, Any]
) -> list[str]:
    """Append ledger rows and advance the slot revision and optional chain head."""
    lines: list[str] = []
    assignments = "SET s.revision = $revision"
    if commit.ledger:
        params["ledger"] = [
            scope.stamp(_ledger_props(entry), tenant) for entry in commit.ledger
        ]
        last = max(commit.ledger, key=lambda entry: entry.seq)
        params["ledger_seq"] = last.seq
        params["ledger_head"] = last.checksum
        assignments += ", s.ledger_seq = $ledger_seq, s.ledger_head = $ledger_head"
        lines += [
            "WITH DISTINCT s",
            "UNWIND $ledger AS x",
            f"CREATE (n:{LEDGER_ENTRY})",
            "SET n += x",
        ]
    lines += ["WITH DISTINCT s", assignments, "RETURN s.revision AS revision"]
    return lines


def build_commit_spec(commit: SlotCommit, scope: TenantScope, tenant: str) -> QuerySpec:
    """Build the single statement that guards and writes a slot commit."""
    params: dict[str, Any] = {
        "slot": commit.slot_key,
        "base": commit.base_revision,
        "genesis": GENESIS,
        "revision": commit.base_revision + 1,
    }
    guard_lines = _slot_guard(scope, params)
    lines = _preconditions(commit, scope, params)
    lines += guard_lines
    lines += _fact_writes(commit, scope, tenant, params)
    lines += _audit_writes(commit, scope, tenant, params)
    lines += _ledger_writes(commit, scope, tenant, params)
    return QuerySpec("\n".join(lines), params, tag="temporal_commit_slot")
