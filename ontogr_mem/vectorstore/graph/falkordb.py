"""Graph-native vectors on FalkorDB."""

import asyncio
import time
from collections.abc import Sequence
from typing import Any, Final

from ontogr_mem.graphstore.errors import StoreQueryError
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.ident import quote
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.types import GraphHandle
from ontogr_mem.vectorstore.errors import IndexDriftError, SpaceMismatchError
from ontogr_mem.vectorstore.filters import matches
from ontogr_mem.vectorstore.graph.base import LABELS, PROPERTY, GraphVectorBackend
from ontogr_mem.vectorstore.models import (
    SearchFilters,
    SpaceSpec,
    VectorKind,
)
from ontogr_mem.vectorstore.protocol import RawHit
from ontogr_mem.vectorstore.settings import VectorStoreSettings


NODE_ENTITY: Final = "NODE"
RELATIONSHIP_ENTITY: Final = "RELATIONSHIP"


class FalkorDBVectorBackend(GraphVectorBackend):
    """Vectors as `vecf32` properties with a native vector index per label.

    Each tenant has its own graph, so no property carries the tenant key. The
    index returns a cosine distance in [0, 2]; the score is `(2 - d) / 2`.
    """

    poll_s = 0.2
    wait_s = 15.0

    def __init__(
        self,
        store: GraphStore,
        handle: GraphHandle,
        settings: VectorStoreSettings | None = None,
    ) -> None:
        """Bind to a tenant's graph and serialize index searches."""
        super().__init__(store, handle, settings)
        # queryNodes repeats one node for a whole page under concurrent load
        # (FalkorDB #2848), so searches against one graph run one at a time.
        self._search_lock = asyncio.Lock()

    def _vector_expr(self, name: str) -> str:
        # A plain list is stored but never indexed; vecf32 is required.
        return f"vecf32({name})"

    async def ensure_ready(self, specs: Sequence[SpaceSpec]) -> None:
        """Create each missing index and wait until it is operational.

        Raises:
            SpaceMismatchError: If an index has another dimension.
            IndexDriftError: If an index is not operational after the wait.
        """
        for spec in specs:
            label = LABELS[spec.kind]
            entity = (
                RELATIONSHIP_ENTITY if spec.kind is VectorKind.EDGE else NODE_ENTITY
            )
            if await self._find(label, entity) is None:
                await self._create(spec)
            index = await self._await_operational(label, entity, spec)
            live = (index.get("options") or {}).get(PROPERTY) or {}
            dimension = live.get("dimension")
            if dimension != spec.dim:
                msg = (
                    f"vector index on {label} has dimension {dimension}, "
                    f"expected {spec.dim}"
                )
                raise SpaceMismatchError(msg)

    async def _find(self, label: str, entity: str) -> dict[str, Any] | None:
        rows = await self._runner.run(
            QuerySpec(
                "CALL db.indexes()",
                read_only=True,
                idempotent=True,
                tag="vector_index",
            )
        )
        for row in rows.rows:
            types = row.get("types") or {}
            if (
                row.get("label") == label
                and row.get("entitytype") == entity
                and "VECTOR" in (types.get(PROPERTY) or [])
            ):
                return row
        return None

    async def _create(self, spec: SpaceSpec) -> None:
        label = quote(LABELS[spec.kind], "label")
        if spec.kind is VectorKind.EDGE:
            pattern, alias = f"()-[e:{label}]->()", "e"
        else:
            pattern, alias = f"(n:{label})", "n"
        ddl = (
            f"CREATE VECTOR INDEX FOR {pattern} ON ({alias}.{PROPERTY}) OPTIONS "
            f"{{dimension: {int(spec.dim)}, similarityFunction: 'cosine', "
            f"efRuntime: {int(self._settings.ef_runtime)}}}"
        )
        try:
            await self._runner.run(QuerySpec(ddl, idempotent=True, tag="vector_ddl"))
        except StoreQueryError as exc:
            if "already" not in str(exc).lower():
                raise

    async def _await_operational(
        self, label: str, entity: str, spec: SpaceSpec
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.wait_s
        while True:
            index = await self._find(label, entity)
            if index is not None and index.get("status") == "OPERATIONAL":
                return index
            if time.monotonic() > deadline:
                msg = (
                    f"vector index on {label} for {spec.kind.value} is not operational"
                )
                raise IndexDriftError(msg)
            await asyncio.sleep(self.poll_s)

    async def _search(
        self, kind: VectorKind, vector: Sequence[float], k: int, filters: SearchFilters
    ) -> list[RawHit]:
        """Search the index, filtering and widening until `k` rows remain.

        A filter runs after the index has cut the result, so a narrow filter
        can leave fewer than `k` hits. The next attempt asks for twice as many
        until `k` distinct ids remain or the recheck budget is spent.

        `k` is the count the fetch sizer requested, which already includes the
        over-fetch factor. `max_rechecks` caps the wider follow-up queries.
        """
        async with self._search_lock:
            requested = k
            rechecks = 0
            while True:
                rows = await self._query(kind, vector, requested)
                hits: list[RawHit] = []
                for row in rows:
                    hit = self._hit(kind, row, filters)
                    if hit is not None:
                        hits.append(hit)
                distinct = len({hit_id for hit_id, _ in hits})
                if not rows or distinct >= k or rechecks >= self._settings.max_rechecks:
                    return hits
                requested *= 2
                rechecks += 1

    async def _query(
        self, kind: VectorKind, vector: Sequence[float], k: int
    ) -> list[dict[str, Any]]:
        if kind is VectorKind.EDGE:
            cypher = (
                f"CALL db.idx.vector.queryRelationships('FACT', '{PROPERTY}', $k, "
                "vecf32($q)) YIELD relationship, score "
                "RETURN relationship.fact_id AS id, score, "
                "properties(relationship) AS props"
            )
        else:
            cypher = (
                f"CALL db.idx.vector.queryNodes('{LABELS[kind]}', '{PROPERTY}', $k, "
                "vecf32($q)) YIELD node, score "
                "RETURN node.id AS id, score, properties(node) AS props"
            )
        result = await self._runner.run(
            QuerySpec(
                cypher,
                {"k": k, "q": list(vector)},
                read_only=True,
                idempotent=True,
                tag="vector_search",
            )
        )
        return result.rows

    @staticmethod
    def _hit(
        kind: VectorKind, row: dict[str, Any], filters: SearchFilters
    ) -> RawHit | None:
        props = row.get("props") or {}
        if not matches(kind, props, filters):
            return None
        distance = float(row["score"])
        score = min(1.0, max(0.0, (2.0 - distance) / 2.0))
        return str(row["id"]), score

    def _is_missing_index(self, error: StoreQueryError) -> bool:
        # The server gives the same answer for a missing and a building index.
        return "invalid arguments for procedure" in str(error).lower()
