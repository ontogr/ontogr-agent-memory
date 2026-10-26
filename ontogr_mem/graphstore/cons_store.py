"""Consolidation store on the graph backends, one typed label per document kind.

Each document is one node. Lookup fields are scalar properties, and the whole
document is kept as one JSON string under ``doc``. The lease and the tick counter
are single nodes. The lease is changed only through guarded writes on its revision,
and the counter is incremented under a lock-first write.
"""

from datetime import datetime, timedelta
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from ontogr_mem.consolidation.errors import StoreError
from ontogr_mem.consolidation.store import check_field, field_text, kind_of
from ontogr_mem.data_models.timestamps import to_micros
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.ident import quote
from ontogr_mem.graphstore.protocol import GraphStore, ScopedRunner
from ontogr_mem.graphstore.reserved import LOCK
from ontogr_mem.graphstore.types import GraphHandle, Guard, Node, NodeSelector


M = TypeVar("M", bound=BaseModel)

_LEASE_LABEL = "ConsLease"
_LEASE_ID = "lease"
_LEASE_INIT: dict[str, Any] = {"rev": 0, "holder": "", "expires": 0, "rerun": False}
_COUNTER_LABEL = "ConsCounter"
_COUNTER_ID = "tick"

# The counter node is locked by its first write, so two statements that read and
# add to `n` cannot interleave on Neo4j, where a plain read and write would lose
# an update.
_TICK = """
MERGE (c:{label} {key})
  ON CREATE SET c.n = 0
{lock_start}
SET c.n = c.n + 1
{lock_end}
RETURN c.n AS n
"""


def _load(model: type[M], raw: object) -> M:
    if not isinstance(raw, str):
        raise StoreError(f"stored {model.__name__} has no document")
    try:
        return model.model_validate_json(raw)
    except ValidationError as exc:
        raise StoreError(f"cannot load stored {model.__name__}") from exc


class GraphConsStore:
    """Consolidation store of one tenant graph.

    Args:
        store: The graph store that owns the tenant's graph.
        handle: The tenant's graph handle.
        tenant_keyed: Set explicitly only when a custom runner does not expose
            its backend scope.
        lease_attempts: Guarded lease writes tried before giving up.
    """

    def __init__(
        self,
        store: GraphStore,
        handle: GraphHandle,
        *,
        tenant_keyed: bool | None = None,
        lease_attempts: int = 8,
    ) -> None:
        """Bind the store to a tenant's graph."""
        self._store = store
        self._handle = handle
        self._runner: ScopedRunner = store.runner(handle)
        self._scope = (
            TenantScope.from_mode(tenant_keyed)
            if tenant_keyed is not None
            else self._runner.scope
        )
        self._lease_attempts = lease_attempts

    def _map(self, body: str = "") -> str:
        return self._scope.pattern(body)

    async def _rows(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        rows = await self._runner.run(
            QuerySpec(cypher, params, read_only=True, idempotent=True, tag="cons_read")
        )
        return rows.rows

    async def tick(self) -> int:
        """Return the next value of the tenant's counter: 1, 2, 3, and so on."""
        lock_start = lock_end = ""
        params: dict[str, Any] = {"id": _COUNTER_ID}
        if self._scope.dialect.keyed:
            lock_start = f"WITH c, c.{LOCK} AS previous\nSET c.{LOCK} = $writer"
            lock_end = f"SET c.{LOCK} = previous"
            params["writer"] = uuid4().hex
        cypher = _TICK.format(
            label=quote(_COUNTER_LABEL, "label"),
            key=self._map("id: $id"),
            lock_start=lock_start,
            lock_end=lock_end,
        )
        rows = await self._runner.run(QuerySpec(cypher, params, tag="cons_tick"))
        return int(rows.scalar("n", 0))

    async def get(self, model: type[M], doc_id: str) -> M | None:
        """Return one document, or None when it is not stored."""
        kind = kind_of(model)
        found = await self._store.get_nodes(self._handle, kind.label, [doc_id])
        return _load(model, found[0].props.get("doc")) if found else None

    async def put(self, obj: BaseModel) -> None:
        """Write one document, replacing any stored under the same id."""
        kind = kind_of(type(obj))
        props: dict[str, Any] = {"doc": obj.model_dump_json()}
        props.update({f: field_text(getattr(obj, f)) for f in kind.fields})
        result = await self._store.upsert_nodes(
            self._handle,
            [
                Node(
                    id=str(getattr(obj, kind.id_field)),
                    label=kind.label,
                    props=props,
                )
            ],
            replace=True,
        )
        if not result.complete:
            raise StoreError(f"could not write a {kind.label} document")

    async def delete(self, model: type[BaseModel], doc_id: str) -> None:
        """Remove one document. Removing a missing document is not an error."""
        kind = kind_of(model)
        await self._store.delete(self._handle, NodeSelector(kind.label, [doc_id]))

    async def find(self, model: type[M], field: str, value: object) -> list[M]:
        """Return the documents whose lookup ``field`` equals ``value``, by id."""
        check_field(model, field)
        label = quote(kind_of(model).label, "label")
        rows = await self._rows(
            f"MATCH (n:{label} {self._map()}) "
            f"WHERE n.{quote(field, 'property key')} = $value "
            "RETURN n.doc AS doc ORDER BY n.id",
            {"value": field_text(value)},
        )
        return [_load(model, row["doc"]) for row in rows]

    async def acquire(self, holder: str, now: datetime, ttl_s: float) -> bool:
        """Take the lease if it is free, expired, or already held by ``holder``."""
        now_us = to_micros(now)
        expires = to_micros(now + timedelta(seconds=ttl_s))
        for _ in range(self._lease_attempts):
            lease = await self._lease_row()
            owner = str(lease.get("holder", ""))
            if owner not in ("", holder) and int(lease.get("expires", 0)) > now_us:
                return False
            if await self._lease_write(
                int(lease.get("rev", 0)), {"holder": holder, "expires": expires}
            ):
                return True
        return False

    async def request_rerun(self) -> None:
        """Ask the current holder to run once more after it finishes.

        Raises:
            StoreError: If the lease kept changing and the request was not stored.
        """
        for _ in range(self._lease_attempts):
            lease = await self._lease_row()
            if await self._lease_write(int(lease.get("rev", 0)), {"rerun": True}):
                return
        raise StoreError("the lease kept changing; the rerun request was not stored")

    async def release(self, holder: str) -> bool:
        """Give the lease back if ``holder`` holds it; True if a rerun was asked.

        Raises:
            StoreError: If the lease kept changing and it was not released.
        """
        for _ in range(self._lease_attempts):
            lease = await self._lease_row()
            if lease.get("holder", "") != holder:
                return False
            released = await self._lease_write(
                int(lease.get("rev", 0)),
                {"holder": "", "expires": 0, "rerun": False},
            )
            if released:
                return bool(lease.get("rerun", False))
        raise StoreError("the lease kept changing; it was not released")

    async def _lease_row(self) -> dict[str, Any]:
        found = await self._store.get_nodes(self._handle, _LEASE_LABEL, [_LEASE_ID])
        return dict(found[0].props) if found else {}

    async def _lease_write(self, rev: int, values: dict[str, Any]) -> bool:
        return await self._store.guarded_write(
            self._handle,
            Guard(
                label=_LEASE_LABEL,
                id=_LEASE_ID,
                expect={"rev": rev},
                set={**values, "rev": rev + 1},
                create_with=_LEASE_INIT,
            ),
        )

    async def list(self, model: type[M]) -> list[M]:
        """Return every document of ``model``, ordered by id."""
        label = quote(kind_of(model).label, "label")
        rows = await self._rows(
            f"MATCH (n:{label} {self._map()}) RETURN n.doc AS doc ORDER BY n.id",
            {},
        )
        return [_load(model, row["doc"]) for row in rows]
