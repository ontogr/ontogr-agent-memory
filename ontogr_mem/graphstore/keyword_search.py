"""Fulltext keyword adapter for tenant-scoped FalkorDB and Neo4j runners."""

from ontogr_mem.graphstore.errors import GraphStoreError, QueryTimeoutError
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.protocol import ScopedRunner
from ontogr_mem.retrieval.errors import (
    KeywordIndexError,
    KeywordSearchError,
    KeywordTimeoutError,
)
from ontogr_mem.retrieval.keyword import KeywordHit, KeywordKind
from ontogr_mem.retrieval.settings import KeywordSettings
from ontogr_mem.retrieval.text import terms


_LABELS = {
    KeywordKind.ENTITY: ("Alias", ("text",)),
    KeywordKind.OBSERVATION: ("Observation", ("statement", "quote")),
    KeywordKind.EPISODE: ("Episode", ("content",)),
}


class GraphKeywordSearch:
    """Fulltext search within one runner's tenant.

    Neo4j searches over-fetched shared-index candidates before tenant filtering.
    Other tenants can therefore reduce the number of hits, but cannot appear
    in the result. Scores depend on the shared corpus.
    """

    def __init__(
        self, runner: ScopedRunner, settings: KeywordSettings | None = None
    ) -> None:
        """Bind a tenant runner and optional query limits."""
        self._runner = runner
        self._settings = settings or KeywordSettings()

    async def search(
        self, kind: KeywordKind, query: str, *, limit: int
    ) -> list[KeywordHit]:
        """Return ranked hits, mapping storage failures to keyword errors."""
        tokens = terms(query)[: self._settings.max_terms]
        if not tokens or limit < 1:
            return []
        label, properties = _LABELS[kind]
        keyed = self._runner.scope.dialect.keyed
        try:
            if not keyed:
                indexes = await self._runner.run(
                    QuerySpec(
                        "CALL db.indexes()",
                        read_only=True,
                        idempotent=True,
                        timeout_ms=self._settings.timeout_ms,
                        tag="keyword_indexes",
                    )
                )
                ready = any(
                    row["label"] == label
                    and row.get("status") == "OPERATIONAL"
                    and all(
                        "FULLTEXT" in (row.get("types", {}).get(prop) or [])
                        for prop in properties
                    )
                    for row in indexes.rows
                )
                if not ready:
                    raise KeywordIndexError(f"fulltext index on {label} is unavailable")
            prefix = (
                f"MATCH (node) WHERE node._tenant = $tenant "
                f"SEARCH node IN (FULLTEXT INDEX am_ft_{label} "
                "FOR $query LIMIT $candidates) SCORE AS score WITH node, score "
                if keyed
                else (
                    f"CALL db.idx.fulltext.queryNodes('{label}', $query) "
                    "YIELD node, score "
                )
            )
            if kind is KeywordKind.ENTITY:
                scope = self._runner.scope.pattern("id: node.entity_id")
                suffix = (
                    f"MATCH (e:Entity {scope}) WHERE e.status = 'active' "
                    "RETURN e.id AS id, max(score) AS score "
                )
            else:
                suffix = "RETURN node.id AS id, max(score) AS score "
            rows = await self._runner.run(
                QuerySpec(
                    prefix + suffix + "ORDER BY score DESC, id LIMIT $limit",
                    {
                        "query": (" OR " if keyed else "|").join(tokens),
                        "candidates": limit * self._settings.tenant_overfetch,
                        "limit": limit,
                    },
                    read_only=True,
                    idempotent=True,
                    timeout_ms=self._settings.timeout_ms,
                    tag="keyword_search",
                )
            )
        except QueryTimeoutError as exc:
            raise KeywordTimeoutError(str(exc)) from exc
        except GraphStoreError as exc:
            message = str(exc)
            if any(
                part in message
                for part in (
                    "IndexNotFound",
                    "no such fulltext",
                    "not ready yet",
                )
            ):
                raise KeywordIndexError(message) from exc
            raise KeywordSearchError(message) from exc
        return [
            KeywordHit(str(row["id"]), float(row["score"]), rank)
            for rank, row in enumerate(rows.rows)
        ]
