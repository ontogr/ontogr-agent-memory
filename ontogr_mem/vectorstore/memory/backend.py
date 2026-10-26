"""Exact in-memory backend used as the reference for the other backends."""

from collections.abc import Sequence
from dataclasses import replace

from ontogr_mem.vectorstore.filters import matches, reject_past_belief
from ontogr_mem.vectorstore.models import (
    BackendWrite,
    Capabilities,
    EdgeExtras,
    Extras,
    NodeExtras,
    Row,
    SearchFilters,
    SpaceSpec,
    ValidityUpdate,
    VectorKind,
)
from ontogr_mem.vectorstore.protocol import RawHit
from ontogr_mem.vectorstore.vecmath import dot, to_score


def _matches_row(
    kind: VectorKind, extras: Extras | None, filters: SearchFilters
) -> bool:
    if kind is VectorKind.NODE:
        node = extras if isinstance(extras, NodeExtras) else NodeExtras()
        properties = {"entity_type": node.entity_type, "status": node.status}
    elif kind is VectorKind.EDGE:
        if not isinstance(extras, EdgeExtras):
            return False
        validity = extras.validity
        properties = {
            "relation_type_id": extras.relation_type_id,
            "subject_id": extras.subject_id,
            "object_id": extras.object_id,
            "status": validity.status,
            "valid_from": validity.valid_from_us,
            "valid_to": validity.valid_to_us,
            "expired_at": validity.expired_at_us,
        }
    else:
        properties = {}
    return matches(kind, properties, filters)


class InMemoryBackend:
    """Brute-force cosine over dicts, for one tenant."""

    capabilities = Capabilities(
        validity_mirror=True,
        exact_scan=True,
        iterative_scan=False,
        ann_index=False,
        filters_rows=True,
    )

    def __init__(self) -> None:
        """Create an empty backend."""
        self._rows: dict[VectorKind, dict[str, Row]] = {k: {} for k in VectorKind}
        self._spaces: dict[VectorKind, SpaceSpec] = {}

    async def ensure_ready(self, specs: Sequence[SpaceSpec]) -> None:
        """Nothing to provision."""

    async def read_space(self, kind: VectorKind) -> SpaceSpec | None:
        """Return the stored space."""
        return self._spaces.get(kind)

    async def write_space(self, spec: SpaceSpec) -> None:
        """Store the space unless one exists."""
        self._spaces.setdefault(spec.kind, spec)

    async def existing_hashes(
        self, kind: VectorKind, ids: Sequence[str]
    ) -> dict[str, str]:
        """Return the stored hash of each id that has a vector."""
        rows = self._rows[kind]
        return {i: rows[i].text_hash for i in ids if i in rows}

    async def write(self, kind: VectorKind, rows: Sequence[Row]) -> BackendWrite:
        """Store the rows, skipping one whose text hash is already stored."""
        written = 0
        store = self._rows[kind]
        for row in rows:
            current = store.get(row.id)
            if current is not None and current.text_hash == row.text_hash:
                continue
            store[row.id] = row
            written += 1
        return BackendWrite(written=written)

    async def update_validity(self, updates: Sequence[ValidityUpdate]) -> int:
        """Replace the validity mirror of existing edge rows."""
        changed = 0
        store = self._rows[VectorKind.EDGE]
        for update in updates:
            current = store.get(update.fact_id)
            if current is None or not isinstance(current.extras, EdgeExtras):
                continue
            store[update.fact_id] = replace(
                current, extras=replace(current.extras, validity=update.validity)
            )
            changed += 1
        return changed

    async def delete(self, kind: VectorKind, ids: Sequence[str]) -> int:
        """Remove the vectors of `ids`."""
        return sum(1 for i in ids if self._rows[kind].pop(i, None) is not None)

    async def search(
        self,
        kind: VectorKind,
        vector: Sequence[float],
        k: int,
        filters: SearchFilters,
        min_score: float | None,
    ) -> list[RawHit]:
        """Return the exact top `k`, then drop scores below `min_score`."""
        reject_past_belief(filters)
        scored = [
            (row.id, to_score(dot(vector, row.vector)))
            for row in self._rows[kind].values()
            if _matches_row(kind, row.extras, filters)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        top = scored[:k]
        if min_score is None:
            return top
        return [pair for pair in top if pair[1] >= min_score]

    async def score_by_ids(
        self, kind: VectorKind, vector: Sequence[float], ids: Sequence[str]
    ) -> dict[str, float]:
        """Return the score of each existing id."""
        store = self._rows[kind]
        return {i: to_score(dot(vector, store[i].vector)) for i in ids if i in store}

    async def exists(self, kind: VectorKind, ids: Sequence[str]) -> set[str]:
        """Return the ids that have a vector."""
        store = self._rows[kind]
        return {i for i in ids if i in store}

    async def iter_ids(
        self, kind: VectorKind, after: str | None, limit: int
    ) -> list[str]:
        """Return a keyset page of ids."""
        ids = sorted(self._rows[kind])
        if after is not None:
            ids = [i for i in ids if i > after]
        return ids[:limit]

    async def fetch_vectors(
        self, kind: VectorKind, ids: Sequence[str]
    ) -> dict[str, tuple[float, ...]]:
        """Return the stored unit vectors."""
        store = self._rows[kind]
        return {i: store[i].vector for i in ids if i in store}
