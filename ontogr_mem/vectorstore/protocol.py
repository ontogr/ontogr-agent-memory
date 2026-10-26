"""The row-level port a vector backend implements."""

from collections.abc import Sequence
from typing import Protocol

from ontogr_mem.vectorstore.models import (
    BackendWrite,
    Capabilities,
    EmbedItem,
    Row,
    SearchFilters,
    SpaceSpec,
    TemporalFilter,
    ValidityUpdate,
    VectorKind,
)


RawHit = tuple[str, float]


class GraphView(Protocol):
    """Graph facts used to check existence, past belief and reconciliation."""

    async def exists(self, kind: VectorKind, ids: Sequence[str]) -> set[str]:
        """Return the subset of `ids` that exists in the graph."""
        ...

    async def iter_items(
        self, kind: VectorKind, after: str | None, limit: int
    ) -> list[EmbedItem]:
        """Return a keyset page of graph objects with their text and extras."""
        ...

    async def still_valid(
        self, ids: Sequence[str], temporal: TemporalFilter
    ) -> set[str]:
        """Return the ids whose fact satisfies `temporal` in the graph."""
        ...


class Backend(Protocol):
    """Rows only. Embedding, hashing and the space checks live in `VectorStore`."""

    capabilities: Capabilities

    async def ensure_ready(self, specs: Sequence[SpaceSpec]) -> None:
        """Make the store usable for `specs` or raise a typed error.

        Raises:
            SpaceMismatchError: If an existing index has another dimension.
            IndexDriftError: If an index cannot be created or does not become usable.
        """
        ...

    async def read_space(self, kind: VectorKind) -> SpaceSpec | None:
        """Return the stored space of `kind`, if any."""
        ...

    async def write_space(self, spec: SpaceSpec) -> None:
        """Store the space of `spec.kind` unless one is already stored."""
        ...

    async def existing_hashes(
        self, kind: VectorKind, ids: Sequence[str]
    ) -> dict[str, str]:
        """Return the stored text hash of each id that has a vector."""
        ...

    async def write(self, kind: VectorKind, rows: Sequence[Row]) -> BackendWrite:
        """Store the vectors; rows whose graph object is missing are reported."""
        ...

    async def update_validity(self, updates: Sequence[ValidityUpdate]) -> int:
        """Set the validity mirror of existing edge rows; return rows changed."""
        ...

    async def delete(self, kind: VectorKind, ids: Sequence[str]) -> int:
        """Remove the vectors of `ids`; missing ids are ignored. Returns the count."""
        ...

    async def search(
        self,
        kind: VectorKind,
        vector: Sequence[float],
        k: int,
        filters: SearchFilters,
        min_score: float | None,
    ) -> list[RawHit]:
        """Return up to `k` distinct `(id, score)` pairs, best first.

        `min_score` is inclusive and applies after the cut to `k`.

        Raises:
            IndexDriftError: If the index is missing or not usable.
            QueryTimeoutError: If the query exceeded its time limit.
        """
        ...

    async def score_by_ids(
        self, kind: VectorKind, vector: Sequence[float], ids: Sequence[str]
    ) -> dict[str, float]:
        """Return the score of each id that has a vector; no cutoff."""
        ...

    async def exists(self, kind: VectorKind, ids: Sequence[str]) -> set[str]:
        """Return the subset of `ids` that has a vector."""
        ...

    async def iter_ids(
        self, kind: VectorKind, after: str | None, limit: int
    ) -> list[str]:
        """Return a keyset page of ids strictly greater than `after`, ascending."""
        ...

    async def fetch_vectors(
        self, kind: VectorKind, ids: Sequence[str]
    ) -> dict[str, tuple[float, ...]]:
        """Return the stored unit vectors of the ids that have one."""
        ...
