"""Vector logic shared by the FalkorDB and Neo4j backends, stored on graph objects."""

from collections.abc import Sequence
from typing import Any, Final

from ontogr_mem.graphstore.errors import (
    QueryTimeoutError as StoreQueryTimeout,
)
from ontogr_mem.graphstore.errors import StoreQueryError
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.types import GraphHandle, Guard
from ontogr_mem.vectorstore.errors import (
    CapabilityError,
    IndexDriftError,
    QueryTimeoutError,
)
from ontogr_mem.vectorstore.filters import reject_past_belief
from ontogr_mem.vectorstore.models import (
    BackendWrite,
    Capabilities,
    Row,
    SearchFilters,
    SpaceSpec,
    ValidityUpdate,
    VectorKind,
)
from ontogr_mem.vectorstore.protocol import RawHit
from ontogr_mem.vectorstore.settings import VectorStoreSettings
from ontogr_mem.vectorstore.vecmath import dot, to_score


LABELS: Final[dict[VectorKind, str]] = {
    VectorKind.NODE: "Entity",
    VectorKind.EDGE: "FACT",
    VectorKind.EPISODE: "Episode",
}
PROPERTY: Final = "embedding"
META_LABEL: Final = "__Meta"
EDGE_KEY: Final = "fact_id"


class GraphVectorBackend:
    """Shared statements; subclasses supply the index DDL and the search query.

    Args:
        store: The graph store that owns the tenant's graph.
        handle: Handle of the tenant's graph.
        settings: Batch sizes and the over-fetch factor.
        tenant_keyed: True when tenants share one database and every node
            carries a `_tenant` property, as on Neo4j.
    """

    capabilities = Capabilities(
        validity_mirror=False,
        exact_scan=False,
        iterative_scan=False,
        ann_index=True,
        filters_rows=True,
    )

    def __init__(
        self,
        store: GraphStore,
        handle: GraphHandle,
        settings: VectorStoreSettings | None = None,
        *,
        tenant_keyed: bool = False,
    ) -> None:
        """Bind to a tenant's graph."""
        self._store = store
        self._handle = handle
        self._runner = store.runner(handle)
        self._settings = settings or VectorStoreSettings()
        self._tenant_keyed = tenant_keyed
        self._params: dict[str, Any] = {"tenant": handle.tenant} if tenant_keyed else {}

    def _vector_expr(self, name: str) -> str:
        """Return the expression that stores the list parameter `name`."""
        raise NotImplementedError

    async def ensure_ready(self, specs: Sequence[SpaceSpec]) -> None:
        """Create the missing indexes and check the existing ones."""
        raise NotImplementedError

    async def _search(
        self, kind: VectorKind, vector: Sequence[float], k: int, filters: SearchFilters
    ) -> list[RawHit]:
        """Return `(id, score)` pairs, scores already in [0, 1]."""
        raise NotImplementedError

    def _is_missing_index(self, error: StoreQueryError) -> bool:
        raise NotImplementedError

    def _scope(self, node: str) -> list[str]:
        """Return the extra conditions that isolate this tenant, when keyed."""
        return [f"{node}._tenant = $tenant"] if self._tenant_keyed else []

    @staticmethod
    def _where(conditions: Sequence[str]) -> str:
        return f" WHERE {' AND '.join(conditions)}" if conditions else ""

    def _node_match(self, kind: VectorKind, id_expr: str | None = None) -> str:
        props: list[str] = []
        if id_expr is not None:
            props.append(f"id: {id_expr}")
        if self._tenant_keyed:
            props.append("_tenant: $tenant")
        body = f" {{{', '.join(props)}}}" if props else ""
        return f"MATCH (x:{LABELS[kind]}{body})"

    @staticmethod
    def _edge_match() -> str:
        return f"MATCH (a)-[e:{LABELS[VectorKind.EDGE]}]->(b)"

    async def read_space(self, kind: VectorKind) -> SpaceSpec | None:
        """Return the space stored on the tenant's meta node."""
        rows = await self._store.get_nodes(
            self._handle, META_LABEL, [f"space:{kind.value}"]
        )
        if not rows or "model" not in rows[0].props:
            return None
        props = rows[0].props
        return SpaceSpec(
            kind,
            props["model"],
            int(props["dim"]),
            props["text_template"],
            int(props["template_version"]),
        )

    async def write_space(self, spec: SpaceSpec) -> None:
        """Store the space unless one is already stored.

        The guard makes the first of several concurrent writers the only one.
        """
        await self._store.guarded_write(
            self._handle,
            Guard(
                META_LABEL,
                f"space:{spec.kind.value}",
                expect={"model": None},
                set={
                    "model": spec.model,
                    "dim": spec.dim,
                    "text_template": spec.text_template,
                    "template_version": spec.template_version,
                },
                create_with={},
            ),
        )

    async def existing_hashes(
        self, kind: VectorKind, ids: Sequence[str]
    ) -> dict[str, str]:
        """Return the stored text hash of each id that has a vector."""
        if kind is VectorKind.EDGE:
            cypher = (
                f"{self._edge_match()}"
                f"{self._where([f'e.{EDGE_KEY} IN $ids', self._has_vector(kind)])} "
                f"RETURN e.{EDGE_KEY} AS id, e.text_hash AS hash"
            )
        else:
            cypher = (
                f"{self._node_match(kind)}"
                f"{self._where(['x.id IN $ids', self._has_vector(kind)])} "
                "RETURN x.id AS id, x.text_hash AS hash"
            )
        rows = await self._runner.run(
            QuerySpec(
                cypher,
                {"ids": list(ids), **self._params},
                read_only=True,
                idempotent=True,
                tag="vector_hashes",
            )
        )
        return {str(row["id"]): row["hash"] for row in rows.rows}

    def _has_vector(self, kind: VectorKind) -> str:
        if kind is VectorKind.EDGE:
            scope = " AND a._tenant = $tenant" if self._tenant_keyed else ""
            return f"e.{PROPERTY} IS NOT NULL{scope}"
        scope = " AND x._tenant = $tenant" if self._tenant_keyed else ""
        return f"x.{PROPERTY} IS NOT NULL{scope}"

    async def write(self, kind: VectorKind, rows: Sequence[Row]) -> BackendWrite:
        """Set the vector on each existing object; report the missing ones."""
        payload = [
            {"id": row.id, "vector": list(row.vector), "hash": row.text_hash}
            for row in rows
        ]
        if kind is VectorKind.EDGE:
            stamp = ", e._tenant = $tenant" if self._tenant_keyed else ""
            cypher = (
                f"UNWIND $rows AS r {self._edge_match()}"
                f"{self._where([f'e.{EDGE_KEY} = r.id', *self._scope('a')])} "
                f"SET e.{PROPERTY} = {self._vector_expr('r.vector')}, "
                f"e.text_hash = r.hash{stamp} "
                f"RETURN e.{EDGE_KEY} AS id"
            )
        else:
            cypher = (
                "UNWIND $rows AS r "
                f"{self._node_match(kind, 'r.id')} "
                f"SET x.{PROPERTY} = {self._vector_expr('r.vector')}, "
                "x.text_hash = r.hash RETURN x.id AS id"
            )
        result = await self._runner.run(
            QuerySpec(
                cypher,
                {"rows": payload, **self._params},
                idempotent=True,
                tag="vector_write",
            )
        )
        written = {str(row["id"]) for row in result.rows}
        return BackendWrite(
            len(written), tuple(row.id for row in rows if row.id not in written)
        )

    async def update_validity(self, updates: Sequence[ValidityUpdate]) -> int:
        """Refuse, because the graph itself holds the validity of a fact.

        Raises:
            CapabilityError: Always. The graph-native backends have no mirror.
        """
        msg = "graph-native backends store no validity mirror"
        raise CapabilityError(msg)

    async def delete(self, kind: VectorKind, ids: Sequence[str]) -> int:
        """Remove the vector and hash from each id; the object stays."""
        if kind is VectorKind.EDGE:
            cypher = (
                f"{self._edge_match()}"
                f"{self._where([f'e.{EDGE_KEY} IN $ids', self._has_vector(kind)])} "
                f"REMOVE e.{PROPERTY}, e.text_hash RETURN count(e) AS removed"
            )
        else:
            cypher = (
                f"{self._node_match(kind)}"
                f"{self._where(['x.id IN $ids', self._has_vector(kind)])} "
                f"REMOVE x.{PROPERTY}, x.text_hash RETURN count(x) AS removed"
            )
        result = await self._runner.run(
            QuerySpec(
                cypher,
                {"ids": list(ids), **self._params},
                idempotent=True,
                tag="vector_delete",
            )
        )
        return int(result.scalar("removed", 0))

    async def search(
        self,
        kind: VectorKind,
        vector: Sequence[float],
        k: int,
        filters: SearchFilters,
        min_score: float | None,
    ) -> list[RawHit]:
        """Return the top `k` by score, then drop scores below `min_score`.

        Rows that a concurrent query returns twice are removed by id.

        Raises:
            CapabilityError: If the filters ask for past belief. The graph
                holds the validity of a fact, but a vector index cannot say
                what was believed at an instant in the past.
            IndexDriftError: If the index is missing or not usable.
            QueryTimeoutError: If the query exceeded its time limit.
        """
        reject_past_belief(filters)
        try:
            hits = await self._search(kind, vector, k, filters)
        except StoreQueryTimeout as exc:
            raise QueryTimeoutError(str(exc)) from exc
        except StoreQueryError as exc:
            if self._is_missing_index(exc):
                msg = f"vector index for {LABELS[kind]} is missing or not usable"
                raise IndexDriftError(msg) from exc
            raise
        best: dict[str, float] = {}
        for hit_id, score in hits:
            best[hit_id] = max(score, best.get(hit_id, 0.0))
        top = sorted(best.items(), key=lambda hit: (-hit[1], hit[0]))[:k]
        if min_score is None:
            return top
        return [hit for hit in top if hit[1] >= min_score]

    async def score_by_ids(
        self, kind: VectorKind, vector: Sequence[float], ids: Sequence[str]
    ) -> dict[str, float]:
        """Return the score of each id that has a vector, from the stored one."""
        vectors = await self.fetch_vectors(kind, ids)
        return {i: to_score(dot(vector, vec)) for i, vec in vectors.items()}

    async def exists(self, kind: VectorKind, ids: Sequence[str]) -> set[str]:
        """Return the ids that have a vector."""
        if kind is VectorKind.EDGE:
            cypher = (
                f"{self._edge_match()}"
                f"{self._where([f'e.{EDGE_KEY} IN $ids', self._has_vector(kind)])} "
                f"RETURN e.{EDGE_KEY} AS id"
            )
        else:
            cypher = (
                f"{self._node_match(kind)}"
                f"{self._where(['x.id IN $ids', self._has_vector(kind)])} "
                "RETURN x.id AS id"
            )
        rows = await self._runner.run(
            QuerySpec(
                cypher,
                {"ids": list(ids), **self._params},
                read_only=True,
                idempotent=True,
                tag="vector_exists",
            )
        )
        return {str(row["id"]) for row in rows.rows}

    async def iter_ids(
        self, kind: VectorKind, after: str | None, limit: int
    ) -> list[str]:
        """Return a keyset page of stored ids."""
        conditions = [self._has_vector(kind)]
        params: dict[str, Any] = {**self._params, "limit": limit}
        if after is not None:
            conditions.append(f"{self._id_expr(kind)} > $after")
            params["after"] = after
        if kind is VectorKind.EDGE:
            cypher = (
                f"{self._edge_match()}{self._where(conditions)} "
                f"RETURN e.{EDGE_KEY} AS id ORDER BY id LIMIT $limit"
            )
        else:
            cypher = (
                f"{self._node_match(kind)}{self._where(conditions)} "
                "RETURN x.id AS id ORDER BY id LIMIT $limit"
            )
        rows = await self._runner.run(
            QuerySpec(cypher, params, read_only=True, idempotent=True, tag="vector_ids")
        )
        return [str(row["id"]) for row in rows.rows]

    @staticmethod
    def _id_expr(kind: VectorKind) -> str:
        return f"e.{EDGE_KEY}" if kind is VectorKind.EDGE else "x.id"

    async def fetch_vectors(
        self, kind: VectorKind, ids: Sequence[str]
    ) -> dict[str, tuple[float, ...]]:
        """Return the stored unit vectors of `ids`."""
        if kind is VectorKind.EDGE:
            cypher = (
                f"{self._edge_match()}"
                f"{self._where([f'e.{EDGE_KEY} IN $ids', self._has_vector(kind)])} "
                f"RETURN e.{EDGE_KEY} AS id, e.{PROPERTY} AS vector"
            )
        else:
            cypher = (
                f"{self._node_match(kind)}"
                f"{self._where(['x.id IN $ids', self._has_vector(kind)])} "
                f"RETURN x.id AS id, x.{PROPERTY} AS vector"
            )
        rows = await self._runner.run(
            QuerySpec(
                cypher,
                {"ids": list(ids), **self._params},
                read_only=True,
                idempotent=True,
                tag="vector_fetch",
            )
        )
        return {
            str(row["id"]): tuple(float(x) for x in row["vector"]) for row in rows.rows
        }
