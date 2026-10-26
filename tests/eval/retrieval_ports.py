"""Selective read suppression for retrieval lane ablations."""

from collections.abc import Sequence

from ontogr_mem.retrieval import KeywordHit, KeywordKind, KeywordSearch
from ontogr_mem.vectorstore import Hit, SearchFilters, VectorKind, VectorStore


class SilentVectors(VectorStore):
    """Delegate vector reads except for the explicitly silenced kinds."""

    def __init__(self, inner: VectorStore, *, kinds: frozenset[VectorKind]) -> None:
        """Keep the original gateway and select suppressed search kinds."""
        super().__init__(inner.backend, inner.gateway, inner.settings)
        self.inner = inner
        self.kinds = kinds

    async def search(
        self,
        kind: VectorKind,
        vector: Sequence[float],
        *,
        k: int,
        filters: SearchFilters | None = None,
        min_score: float | None = None,
    ) -> list[Hit]:
        """Return no hits for silenced kinds and delegate other searches."""
        if kind in self.kinds:
            return []
        return await self.inner.search(
            kind, vector, k=k, filters=filters, min_score=min_score
        )


class SilentKeywords:
    """Delegate keyword reads except for the explicitly silenced kinds."""

    def __init__(self, inner: KeywordSearch, *, kinds: frozenset[KeywordKind]) -> None:
        """Select suppressed document kinds on the original keyword port."""
        self.inner = inner
        self.kinds = kinds

    async def search(
        self, kind: KeywordKind, query: str, *, limit: int
    ) -> list[KeywordHit]:
        """Return no hits for silenced kinds and delegate other searches."""
        if kind in self.kinds:
            return []
        return await self.inner.search(kind, query, limit=limit)
