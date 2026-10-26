"""Graph-native vectors on Neo4j, searched with the Cypher 25 `SEARCH` clause."""

import asyncio
import time
from collections.abc import Sequence
from typing import Any, Final

from ontogr_mem.data_models import OPEN_MICROS, FactStatus
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
    TemporalFilter,
    VectorKind,
)
from ontogr_mem.vectorstore.protocol import RawHit
from ontogr_mem.vectorstore.settings import VectorStoreSettings


FILTERS: Final[dict[VectorKind, tuple[str, ...]]] = {
    VectorKind.NODE: ("_tenant", "status"),
    VectorKind.EPISODE: ("_tenant",),
    VectorKind.EDGE: ("_tenant", "status", "expired_at"),
}
"""Properties each index can filter on inside the nearest-neighbour search."""


def index_name(label: str) -> str:
    """Return the name of the vector index on `label`."""
    return f"am_vec_{label.lower()}_{PROPERTY}"


class Neo4jVectorBackend(GraphVectorBackend):
    """Vectors as float-list properties with one vector index per label.

    The index covers every tenant and declares the properties a search filters
    on, so those filters run inside the nearest-neighbour search. Quantization
    is off, so the native score `(1 + cos) / 2` is exact.

    Args:
        store: The graph store that owns the shared database.
        handle: Handle of the tenant's graph.
        settings: Batch sizes, the over-fetch factor and recheck rounds.
    """

    poll_s: Final = 0.2
    wait_s: Final = 30.0

    def __init__(
        self,
        store: GraphStore,
        handle: GraphHandle,
        settings: VectorStoreSettings | None = None,
    ) -> None:
        """Bind to a tenant's graph; Neo4j graphs are always tenant keyed."""
        super().__init__(store, handle, settings, tenant_keyed=True)

    def _vector_expr(self, name: str) -> str:
        """Return the plain list parameter that Neo4j indexes directly."""
        return name

    async def ensure_ready(self, specs: Sequence[SpaceSpec]) -> None:
        """Create each missing index and wait until it is online.

        Raises:
            SpaceMismatchError: If an index has another dimension.
            IndexDriftError: If an index failed or is not online within the wait.
        """
        for spec in specs:
            label = LABELS[spec.kind]
            name = index_name(label)
            if name not in await self._indexes():
                await self._create(spec.kind, name, spec.dim)
            index = await self._await_online(name)
            dim = self._dimension(index)
            if dim != spec.dim:
                msg = f"vector index {name} has dimension {dim}, expected {spec.dim}"
                raise SpaceMismatchError(msg)

    async def _indexes(self) -> dict[str, dict[str, Any]]:
        rows = await self._runner.run(
            QuerySpec(
                "SHOW VECTOR INDEXES YIELD name, state, options",
                read_only=True,
                idempotent=True,
                auto_commit=True,
                tag="vector_index",
            )
        )
        return {str(row["name"]): row for row in rows.rows}

    async def _await_online(self, name: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.wait_s
        while True:
            index = (await self._indexes()).get(name)
            if index is not None and index.get("state") == "ONLINE":
                return index
            if index is not None and index.get("state") == "FAILED":
                msg = f"vector index {name} failed to build"
                raise IndexDriftError(msg)
            if time.monotonic() > deadline:
                msg = f"vector index {name} is not online"
                raise IndexDriftError(msg)
            await asyncio.sleep(self.poll_s)

    @staticmethod
    def _dimension(index: dict[str, Any]) -> int | None:
        config = (index.get("options") or {}).get("indexConfig") or {}
        dim = config.get("vector.dimensions")
        return int(dim) if dim is not None else None

    async def _create(self, kind: VectorKind, name: str, dim: int) -> None:
        try:
            await self._runner.run(
                QuerySpec(
                    self._ddl(kind, name, dim),
                    idempotent=True,
                    auto_commit=True,
                    tag="vector_ddl",
                )
            )
        except StoreQueryError as exc:
            # When two clients create the same index at once, the loser gets an
            # error saying an equivalent index exists.
            message = str(exc).lower()
            if "already exists" not in message and "equivalent" not in message:
                raise

    @staticmethod
    def _ddl(kind: VectorKind, name: str, dim: int) -> str:
        var = "r" if kind is VectorKind.EDGE else "n"
        if kind is VectorKind.EDGE:
            pattern = f"()-[r:{quote(LABELS[kind], 'relationship type')}]-()"
        else:
            pattern = f"(n:{quote(LABELS[kind], 'label')})"
        filters = ", ".join(f"{var}.{prop}" for prop in FILTERS[kind])
        return (
            f"CREATE VECTOR INDEX {quote(name)} IF NOT EXISTS FOR {pattern} "
            f"ON {var}.{quote(PROPERTY, 'property key')} WITH [{filters}] "
            "OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {int(dim)}, "
            "`vector.similarity_function`: 'cosine', "
            "`vector.quantization.type`: 'none'}}"
        )

    async def _search(
        self, kind: VectorKind, vector: Sequence[float], k: int, filters: SearchFilters
    ) -> list[RawHit]:
        """Return the nearest neighbours that pass the filters.

        Filters the index cannot evaluate are applied here. When they leave
        fewer than `k` hits and the index returned everything it was asked for,
        the fetch doubles for up to `max_rechecks` additional rounds.
        """
        fetch = k
        hits: list[RawHit] = []
        for _ in range(self._settings.max_rechecks + 1):
            rows = await self._query(kind, vector, fetch, filters)
            hits = [
                (str(row["id"]), min(1.0, max(0.0, float(row["score"]))))
                for row in rows
                if matches(kind, row, filters)
            ]
            if len(hits) >= k or len(rows) < fetch:
                break
            fetch *= 2
        return hits

    async def _query(
        self,
        kind: VectorKind,
        vector: Sequence[float],
        fetch: int,
        filters: SearchFilters,
    ) -> list[dict[str, Any]]:
        cypher, params = self._statement(kind, fetch, filters)
        params["q"] = list(vector)
        rows = await self._runner.run(
            QuerySpec(
                cypher,
                params,
                read_only=True,
                idempotent=True,
                tag="vector_search",
            )
        )
        return rows.rows

    def _statement(
        self, kind: VectorKind, fetch: int, filters: SearchFilters
    ) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {**self._params, "k": fetch}
        if kind is VectorKind.EDGE:
            var = "r"
            match = f"MATCH ()-[r:{quote(LABELS[kind], 'relationship type')}]->()"
            conditions = ["r._tenant = $tenant"]
            if not (filters.temporal or TemporalFilter()).history:
                conditions.append(f"r.status = '{FactStatus.ACTIVE.value}'")
                conditions.append("r.expired_at = $open")
                params["open"] = OPEN_MICROS
            returns = (
                "RETURN r.fact_id AS id, score, r.subject_id AS subject_id, "
                "r.object_id AS object_id, "
                "r.relation_type_id AS relation_type_id, r.status AS status, "
                "r.valid_from AS valid_from, r.valid_to AS valid_to, "
                "r.expired_at AS expired_at"
            )
        else:
            var = "n"
            match = f"MATCH (n:{quote(LABELS[kind], 'label')})"
            conditions = ["n._tenant = $tenant"]
            if kind is VectorKind.NODE and filters.statuses is not None:
                conditions.append("n.status IN $statuses")
                params["statuses"] = sorted(filters.statuses)
            returns = (
                "RETURN n.id AS id, score, n.entity_type AS entity_type, "
                "n.status AS status"
            )
        where = " AND ".join(conditions)
        name = quote(index_name(LABELS[kind]), "index name")
        cypher = (
            f"CYPHER 25 {match} SEARCH {var} IN (VECTOR INDEX {name} FOR $q "
            f"WHERE {where} LIMIT $k) SCORE AS score {returns}"
        )
        return cypher, params

    @staticmethod
    def _is_missing_index(error: StoreQueryError) -> bool:
        text = str(error).lower()
        return "indexnotfound" in text or "does not exist" in text
