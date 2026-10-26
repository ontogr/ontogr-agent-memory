"""Crash-safe hand-off from the graph side: entry types, rules and stores."""

import itertools
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

from ontogr_mem.vectorstore.models import (
    EdgeExtras,
    EmbedItem,
    Extras,
    NodeExtras,
    Validity,
    VectorKind,
)
from ontogr_mem.vectorstore.pgvector.pool import Conn, Pool
from ontogr_mem.vectorstore.pgvector.schema import schema_name, table_name
from ontogr_mem.vectorstore.pgvector.settings import PgVectorSettings


class OutboxOp(StrEnum):
    """What the drain must do for one object."""

    UPSERT = "upsert"
    DELETE = "delete"
    VALIDITY = "validity"


_RANK = {OutboxOp.UPSERT: 0, OutboxOp.VALIDITY: 1, OutboxOp.DELETE: 2}


def op_rank(op: OutboxOp) -> int:
    """Return the drain order: upserts, then validity changes, then deletes."""
    return _RANK[op]


@dataclass(frozen=True, slots=True)
class OutboxEntry:
    """One pending piece of work, keyed by `(kind, object_id, op)`.

    Attributes:
        kind: Kind of vector the work affects.
        object_id: Id of the graph object.
        op: The work to do.
        text: Text to embed, for an upsert.
        extras: Kind-specific properties, for an upsert.
        validity: The fact's current version, for a validity change.
        episode_id: Episode that caused the work, for the completion gate.
        attempts: Claims so far.
        last_error: Error from the last failed attempt.
        token: Changes when the entry is re-enqueued; a stale ack keeps it.
        age_s: Age of the entry when it was claimed.
    """

    kind: VectorKind
    object_id: str
    op: OutboxOp
    text: str | None = None
    extras: Extras | None = None
    validity: Validity | None = None
    episode_id: str | None = None
    attempts: int = 0
    last_error: str | None = None
    token: str = ""
    age_s: float = 0.0

    @property
    def key(self) -> tuple[VectorKind, str, OutboxOp]:
        """Return the primary key."""
        return (self.kind, self.object_id, self.op)

    def to_item(self) -> EmbedItem:
        """Return the `EmbedItem` an upsert entry stands for.

        Raises:
            ValueError: If this entry is not an upsert with text.
        """
        if self.op is not OutboxOp.UPSERT or self.text is None:
            msg = f"{self.op.value} entry has no embed item"
            raise ValueError(msg)
        return EmbedItem(self.object_id, self.text, self.extras)


def upsert_entry(
    kind: VectorKind, item: EmbedItem, episode_id: str | None = None
) -> OutboxEntry:
    """Return an entry asking the drain to embed and store `item`."""
    return OutboxEntry(
        kind,
        item.id,
        OutboxOp.UPSERT,
        text=item.text,
        extras=item.extras,
        episode_id=episode_id,
    )


def delete_entry(
    kind: VectorKind, object_id: str, episode_id: str | None = None
) -> OutboxEntry:
    """Return an entry asking the drain to delete a vector row."""
    return OutboxEntry(kind, object_id, OutboxOp.DELETE, episode_id=episode_id)


def validity_entry(
    fact_id: str, validity: Validity, episode_id: str | None = None
) -> OutboxEntry:
    """Return an entry asking the drain to refresh an edge's validity mirror."""
    return OutboxEntry(
        VectorKind.EDGE,
        fact_id,
        OutboxOp.VALIDITY,
        validity=validity,
        episode_id=episode_id,
    )


@dataclass(frozen=True, slots=True)
class _CombineAction:
    """Mutation selected by the shared outbox combining rules."""

    remove: tuple[OutboxOp, ...] = ()
    put: OutboxEntry | None = None
    merge_validity: Validity | None = None


def _combine(current: Sequence[OutboxEntry], entry: OutboxEntry) -> _CombineAction:
    """Choose the queue mutation for an entry and its current object state."""
    if entry.op is OutboxOp.DELETE:
        return _CombineAction((OutboxOp.UPSERT, OutboxOp.VALIDITY), entry)
    if entry.op is OutboxOp.UPSERT:
        return _CombineAction((OutboxOp.DELETE, OutboxOp.VALIDITY), entry)

    if any(row.op is OutboxOp.DELETE for row in current):
        return _CombineAction()

    pending_upsert = next((row for row in current if row.op is OutboxOp.UPSERT), None)
    if (
        entry.validity is not None
        and pending_upsert is not None
        and isinstance(pending_upsert.extras, EdgeExtras)
    ):
        return _CombineAction(merge_validity=entry.validity)
    return _CombineAction(put=entry)


def validity_to_json(validity: Validity) -> dict[str, Any]:
    """Return a validity as a JSON object."""
    return {
        "status": validity.status,
        "valid_from_us": validity.valid_from_us,
        "valid_to_us": validity.valid_to_us,
        "recorded_at_us": validity.recorded_at_us,
        "expired_at_us": validity.expired_at_us,
    }


def validity_from_json(data: dict[str, Any]) -> Validity:
    """Return a validity from its JSON object."""
    return Validity(
        status=data["status"],
        valid_from_us=data["valid_from_us"],
        valid_to_us=data["valid_to_us"],
        recorded_at_us=data["recorded_at_us"],
        expired_at_us=data["expired_at_us"],
    )


def extras_to_json(extras: Extras | None) -> dict[str, Any] | None:
    """Return extras as a JSON object; `None` stays `None`."""
    if extras is None:
        return None
    if isinstance(extras, NodeExtras):
        return {"entity_type": extras.entity_type, "status": extras.status}
    return {
        "subject_id": extras.subject_id,
        "object_id": extras.object_id,
        "relation_type_id": extras.relation_type_id,
        "validity": validity_to_json(extras.validity),
    }


def extras_from_json(kind: VectorKind, data: dict[str, Any] | None) -> Extras | None:
    """Return extras from their JSON object for `kind`."""
    if data is None:
        return None
    if kind is VectorKind.NODE:
        return NodeExtras(data.get("entity_type"), data.get("status"))
    return EdgeExtras(
        subject_id=data["subject_id"],
        object_id=data["object_id"],
        relation_type_id=data.get("relation_type_id"),
        validity=validity_from_json(data["validity"]),
    )


class Outbox(Protocol):
    """Pending work queue. Operations are idempotent, so delivery is at least once."""

    async def enqueue(self, entries: Sequence[OutboxEntry]) -> None:
        """Add entries, applying the combining rules."""
        ...

    async def claim(self, limit: int) -> list[OutboxEntry]:
        """Lease up to `limit` entries; each claim counts as one attempt."""
        ...

    async def ack(self, entries: Sequence[OutboxEntry]) -> None:
        """Remove entries, unless they were re-enqueued since the claim."""
        ...

    async def fail(self, entries: Sequence[OutboxEntry], error: str) -> None:
        """Record an error and make the entries claimable again."""
        ...

    async def release(self, entries: Sequence[OutboxEntry]) -> None:
        """Give entries back without counting the attempt."""
        ...

    async def depth(self) -> int:
        """Return the number of entries, dead letters included."""
        ...

    async def dead_count(self) -> int:
        """Return the entries that exhausted their attempts."""
        ...

    async def oldest_age_s(self) -> float | None:
        """Return the age of the oldest entry, or `None` when empty."""
        ...

    async def pending_for_episode(self, episode_id: str) -> int:
        """Return the entries tied to an episode."""
        ...


class InMemoryOutbox:
    """Reference implementation of the outbox rules.

    Args:
        max_attempts: Tries before an entry becomes a dead letter.
        lease_s: How long a claim protects an entry.
        clock: Monotonic clock, injectable in tests.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 5,
        lease_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty outbox."""
        self.max_attempts = max_attempts
        self.lease_s = lease_s
        self._clock = clock
        self._rows: dict[tuple[VectorKind, str, OutboxOp], OutboxEntry] = {}
        self._born: dict[tuple[VectorKind, str, OutboxOp], float] = {}
        self._claimed: dict[tuple[VectorKind, str, OutboxOp], float] = {}
        self._tokens = itertools.count(1)

    def _put(self, entry: OutboxEntry) -> None:
        self._rows[entry.key] = replace(
            entry, attempts=0, last_error=None, token=str(next(self._tokens))
        )
        self._born[entry.key] = self._clock()
        self._claimed.pop(entry.key, None)

    def _drop(self, kind: VectorKind, object_id: str, *ops: OutboxOp) -> None:
        for op in ops:
            key = (kind, object_id, op)
            self._rows.pop(key, None)
            self._born.pop(key, None)
            self._claimed.pop(key, None)

    async def enqueue(self, entries: Sequence[OutboxEntry]) -> None:
        """Apply the combining rules, in order."""
        for entry in entries:
            current = [
                row
                for row in self._rows.values()
                if row.kind is entry.kind and row.object_id == entry.object_id
            ]
            action = _combine(current, entry)
            if action.remove:
                self._drop(entry.kind, entry.object_id, *action.remove)
            if action.merge_validity is not None:
                key = (entry.kind, entry.object_id, OutboxOp.UPSERT)
                pending = self._rows[key]
                if isinstance(pending.extras, EdgeExtras):
                    self._rows[key] = replace(
                        pending,
                        extras=replace(pending.extras, validity=action.merge_validity),
                        token=str(next(self._tokens)),
                    )
            elif action.put is not None:
                self._put(action.put)

    async def claim(self, limit: int) -> list[OutboxEntry]:
        """Lease the oldest claimable entries, upserts first."""
        now = self._clock()
        ready = [
            key
            for key, row in self._rows.items()
            if row.attempts < self.max_attempts
            and (key not in self._claimed or now - self._claimed[key] >= self.lease_s)
        ]
        ready.sort(key=lambda key: (op_rank(key[2]), self._born[key]))
        out: list[OutboxEntry] = []
        for key in ready[:limit]:
            claimed = replace(
                self._rows[key],
                attempts=self._rows[key].attempts + 1,
                age_s=now - self._born[key],
            )
            self._rows[key] = replace(claimed, age_s=0.0)
            self._claimed[key] = now
            out.append(claimed)
        return out

    def _same(self, entry: OutboxEntry) -> bool:
        current = self._rows.get(entry.key)
        return current is not None and current.token == entry.token

    async def ack(self, entries: Sequence[OutboxEntry]) -> None:
        """Remove entries whose token still matches."""
        for entry in entries:
            if self._same(entry):
                self._drop(entry.kind, entry.object_id, entry.op)

    async def fail(self, entries: Sequence[OutboxEntry], error: str) -> None:
        """Record the error and unclaim."""
        for entry in entries:
            if self._same(entry):
                self._rows[entry.key] = replace(self._rows[entry.key], last_error=error)
                self._claimed.pop(entry.key, None)

    async def release(self, entries: Sequence[OutboxEntry]) -> None:
        """Unclaim without counting the attempt."""
        for entry in entries:
            if self._same(entry):
                current = self._rows[entry.key]
                self._rows[entry.key] = replace(
                    current, attempts=max(current.attempts - 1, 0)
                )
                self._claimed.pop(entry.key, None)

    async def depth(self) -> int:
        """Return the entry count."""
        return len(self._rows)

    async def dead_count(self) -> int:
        """Return the entries out of attempts."""
        return sum(
            1 for row in self._rows.values() if row.attempts >= self.max_attempts
        )

    async def oldest_age_s(self) -> float | None:
        """Return the age of the oldest entry."""
        if not self._born:
            return None
        return self._clock() - min(self._born.values())

    async def pending_for_episode(self, episode_id: str) -> int:
        """Return the entries tied to `episode_id`."""
        return sum(1 for row in self._rows.values() if row.episode_id == episode_id)

    def rows(self) -> list[OutboxEntry]:
        """Return a snapshot of the entries."""
        return list(self._rows.values())


class PgOutbox:
    """Outbox in `"m_<tenant>".vec_outbox`.

    Args:
        pool: Connection source.
        tenant: Tenant id.
        settings: Lease and attempt limits.
    """

    def __init__(
        self, pool: Pool, tenant: str, settings: PgVectorSettings | None = None
    ) -> None:
        """Bind the outbox to one tenant schema."""
        self.pool = pool
        self.settings = settings or PgVectorSettings()
        self.max_attempts = self.settings.max_attempts
        self.lease_s = self.settings.lease_s
        self._t = table_name(schema_name(tenant), "vec_outbox")

    async def enqueue(self, entries: Sequence[OutboxEntry]) -> None:
        """Apply the combining rules for each entry, in order, in one transaction."""
        async with self.pool.acquire() as conn, conn.transaction():
            keys = sorted({(entry.kind.value, entry.object_id) for entry in entries})
            for kind, object_id in keys:
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1), "
                    "hashtext($2 || ':' || $3))",
                    self._t,
                    kind,
                    object_id,
                )
            for entry in entries:
                await self._enqueue_one(conn, entry)

    async def _drop(self, conn: Conn, entry: OutboxEntry, *ops: OutboxOp) -> None:
        await conn.execute(
            f"DELETE FROM {self._t} "
            "WHERE kind = $1 AND object_id = $2 AND op = ANY($3::text[])",
            entry.kind.value,
            entry.object_id,
            [op.value for op in ops],
        )

    async def _put(self, conn: Conn, entry: OutboxEntry) -> None:
        if entry.op is OutboxOp.VALIDITY:
            payload: dict[str, Any] | None = (
                {"validity": validity_to_json(entry.validity)}
                if entry.validity is not None
                else None
            )
        else:
            payload = extras_to_json(entry.extras)
        await conn.execute(
            f"""INSERT INTO {self._t}
            (kind, object_id, op, text, episode_id, extras)
            VALUES ($1, $2, $3, $4, $5, $6::text::jsonb)
            ON CONFLICT (kind, object_id, op) DO UPDATE SET text = EXCLUDED.text,
              episode_id = EXCLUDED.episode_id, extras = EXCLUDED.extras,
              attempts = 0, last_error = NULL, claimed_at = NULL,
              token = gen_random_uuid(), created_at = now()""",
            entry.kind.value,
            entry.object_id,
            entry.op.value,
            entry.text,
            entry.episode_id,
            json.dumps(payload) if payload is not None else None,
        )

    async def _current(self, conn: Conn, entry: OutboxEntry) -> list[OutboxEntry]:
        rows = await conn.fetch(
            f"SELECT kind, object_id, op, extras::text AS extras FROM {self._t} "
            "WHERE kind = $1 AND object_id = $2 FOR UPDATE",
            entry.kind.value,
            entry.object_id,
        )
        current: list[OutboxEntry] = []
        for row in rows:
            kind = VectorKind(row["kind"])
            op = OutboxOp(row["op"])
            raw = json.loads(row["extras"]) if row["extras"] else None
            has_edge_extras = (
                raw is not None
                and {
                    "subject_id",
                    "object_id",
                    "validity",
                }
                <= raw.keys()
            )
            current.append(
                OutboxEntry(
                    kind,
                    row["object_id"],
                    op,
                    extras=extras_from_json(kind, raw)
                    if op is OutboxOp.UPSERT
                    and raw is not None
                    and (kind is VectorKind.NODE or has_edge_extras)
                    else None,
                    validity=validity_from_json(raw["validity"])
                    if op is OutboxOp.VALIDITY and raw
                    else None,
                )
            )
        return current

    async def _enqueue_one(self, conn: Conn, entry: OutboxEntry) -> None:
        current = (
            await self._current(conn, entry) if entry.op is OutboxOp.VALIDITY else ()
        )
        action = _combine(current, entry)
        if action.remove:
            await self._drop(conn, entry, *action.remove)
        if action.merge_validity is not None:
            await conn.execute(
                f"UPDATE {self._t} SET extras = jsonb_set(extras, '{{validity}}', "
                "$3::text::jsonb), token = gen_random_uuid() "
                "WHERE kind = $1 AND object_id = $2 AND op = 'upsert'",
                entry.kind.value,
                entry.object_id,
                json.dumps(validity_to_json(action.merge_validity)),
            )
        elif action.put is not None:
            await self._put(conn, action.put)

    async def claim(self, limit: int) -> list[OutboxEntry]:
        """Lease the oldest claimable entries with `FOR UPDATE SKIP LOCKED`."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""UPDATE {self._t} o SET attempts = o.attempts + 1,
                  claimed_at = now()
                WHERE (o.kind, o.object_id, o.op) IN (
                  SELECT kind, object_id, op FROM {self._t}
                  WHERE attempts < $1
                    AND (claimed_at IS NULL
                      OR claimed_at < now() - ($2::float8 * interval '1 second'))
                  ORDER BY CASE op WHEN 'upsert' THEN 0
                    WHEN 'validity' THEN 1 ELSE 2 END, created_at
                  LIMIT $3 FOR UPDATE SKIP LOCKED)
                RETURNING o.kind, o.object_id, o.op, o.text, o.episode_id,
                  o.extras::text AS extras, o.attempts, o.last_error,
                  o.token::text AS token,
                  EXTRACT(EPOCH FROM now() - o.created_at)::float8 AS age_s""",
                self.max_attempts,
                self.lease_s,
                limit,
            )
        out: list[OutboxEntry] = []
        for row in rows:
            kind = VectorKind(row["kind"])
            op = OutboxOp(row["op"])
            raw = json.loads(row["extras"]) if row["extras"] else None
            out.append(
                OutboxEntry(
                    kind,
                    row["object_id"],
                    op,
                    text=row["text"],
                    extras=None
                    if op is OutboxOp.VALIDITY
                    else extras_from_json(kind, raw),
                    validity=validity_from_json(raw["validity"])
                    if op is OutboxOp.VALIDITY and raw
                    else None,
                    episode_id=row["episode_id"],
                    attempts=row["attempts"],
                    last_error=row["last_error"],
                    token=row["token"],
                    age_s=float(row["age_s"]),
                )
            )
        out.sort(key=lambda entry: (op_rank(entry.op), -entry.age_s))
        return out

    async def ack(self, entries: Sequence[OutboxEntry]) -> None:
        """Delete entries whose token still matches."""
        if not entries:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._t} o USING unnest($1::text[], $2::text[], "
                "$3::text[], $4::uuid[]) AS e(kind, object_id, op, token) "
                "WHERE o.kind = e.kind AND o.object_id = e.object_id "
                "AND o.op = e.op AND o.token = e.token",
                [entry.kind.value for entry in entries],
                [entry.object_id for entry in entries],
                [entry.op.value for entry in entries],
                [entry.token for entry in entries],
            )

    async def fail(self, entries: Sequence[OutboxEntry], error: str) -> None:
        """Record the error and unclaim, in one statement."""
        if not entries:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {self._t} o SET last_error = $5, claimed_at = NULL "
                "FROM unnest($1::text[], $2::text[], $3::text[], $4::uuid[]) "
                "AS e(kind, object_id, op, token) "
                "WHERE o.kind = e.kind AND o.object_id = e.object_id "
                "AND o.op = e.op AND o.token = e.token",
                [entry.kind.value for entry in entries],
                [entry.object_id for entry in entries],
                [entry.op.value for entry in entries],
                [entry.token for entry in entries],
                error,
            )

    async def release(self, entries: Sequence[OutboxEntry]) -> None:
        """Unclaim without counting the attempt."""
        if not entries:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {self._t} o "
                "SET attempts = GREATEST(o.attempts - 1, 0), claimed_at = NULL "
                "FROM unnest($1::text[], $2::text[], $3::text[], $4::uuid[]) "
                "AS e(kind, object_id, op, token) "
                "WHERE o.kind = e.kind AND o.object_id = e.object_id "
                "AND o.op = e.op AND o.token = e.token",
                [entry.kind.value for entry in entries],
                [entry.object_id for entry in entries],
                [entry.op.value for entry in entries],
                [entry.token for entry in entries],
            )

    async def depth(self) -> int:
        """Return the entry count."""
        async with self.pool.acquire() as conn:
            return int(await conn.fetchval(f"SELECT count(*) FROM {self._t}"))

    async def dead_count(self) -> int:
        """Return the entries out of attempts."""
        async with self.pool.acquire() as conn:
            return int(
                await conn.fetchval(
                    f"SELECT count(*) FROM {self._t} WHERE attempts >= $1",
                    self.max_attempts,
                )
            )

    async def oldest_age_s(self) -> float | None:
        """Return the age of the oldest entry."""
        async with self.pool.acquire() as conn:
            value = await conn.fetchval(
                f"SELECT EXTRACT(EPOCH FROM now() - min(created_at))::float8 "
                f"FROM {self._t}"
            )
        return None if value is None else float(value)

    async def pending_for_episode(self, episode_id: str) -> int:
        """Return the entries tied to `episode_id`."""
        async with self.pool.acquire() as conn:
            return int(
                await conn.fetchval(
                    f"SELECT count(*) FROM {self._t} WHERE episode_id = $1",
                    episode_id,
                )
            )
