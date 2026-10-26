"""A graph view that the pgvector tests can seed and inspect in memory."""

from collections.abc import Callable, Sequence

from ontogr_mem.vectorstore.models import EmbedItem, TemporalFilter, VectorKind


class FakeGraph:
    """The three graph reads the pgvector store and reconciler use.

    Args:
        valid: Predicate deciding whether a fact was believed at the filter's
            instant. Defaults to every fact.
    """

    def __init__(
        self, valid: Callable[[str, TemporalFilter], bool] | None = None
    ) -> None:
        """Create an empty graph."""
        self.items: dict[VectorKind, dict[str, EmbedItem]] = {
            kind: {} for kind in VectorKind
        }
        self.valid: Callable[[str, TemporalFilter], bool] = valid or (
            lambda _object_id, _temporal: True
        )
        self.valid_calls: list[list[str]] = []

    def add(self, kind: VectorKind, *items: EmbedItem) -> None:
        """Add or replace graph objects."""
        for item in items:
            self.items[kind][item.id] = item

    def remove(self, kind: VectorKind, object_id: str) -> None:
        """Remove a graph object."""
        self.items[kind].pop(object_id, None)

    def ids(self, kind: VectorKind) -> set[str]:
        """Return the ids the graph holds for `kind`."""
        return set(self.items[kind])

    async def exists(self, kind: VectorKind, ids: Sequence[str]) -> set[str]:
        """Return the subset of `ids` present in the graph."""
        have = self.items[kind]
        return {object_id for object_id in ids if object_id in have}

    async def iter_items(
        self, kind: VectorKind, after: str | None, limit: int
    ) -> list[EmbedItem]:
        """Return a keyset page of graph objects, ascending by id."""
        have = self.items[kind]
        keys = sorted(
            object_id for object_id in have if after is None or object_id > after
        )
        return [have[key] for key in keys[:limit]]

    async def still_valid(
        self, ids: Sequence[str], temporal: TemporalFilter
    ) -> set[str]:
        """Return the ids whose fact the graph believed at the asked instant."""
        self.valid_calls.append(list(ids))
        return {object_id for object_id in ids if self.valid(object_id, temporal)}
