"""`VectorStore`: embeds once per text, stores through a backend, searches by cosine."""

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from typing import TypeVar

from ontogr_mem.observability.metrics import Metrics
from ontogr_mem.vectorstore.embedder import EmbeddingGateway
from ontogr_mem.vectorstore.errors import (
    CapabilityError,
    SpaceMismatchError,
    StoreNotReadyError,
)
from ontogr_mem.vectorstore.fetch_size import fetch_size
from ontogr_mem.vectorstore.models import (
    Capabilities,
    EdgeExtras,
    EmbedItem,
    FailedItem,
    FailReason,
    Hit,
    NodeExtras,
    Row,
    SearchFilters,
    SpaceSpec,
    TemporalFilter,
    ValidityUpdate,
    VectorKind,
    WriteResult,
    as_ids,
)
from ontogr_mem.vectorstore.protocol import Backend, GraphView, RawHit
from ontogr_mem.vectorstore.settings import VectorStoreSettings
from ontogr_mem.vectorstore.spaces import SpaceRegistry
from ontogr_mem.vectorstore.text import TEMPLATES, text_hash
from ontogr_mem.vectorstore.vecmath import to_unit


T = TypeVar("T")
Rechecker = Callable[[Sequence[str], TemporalFilter], Awaitable[set[str]]]


def _chunks(seq: Sequence[T], size: int) -> list[Sequence[T]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _check_extras(kind: VectorKind, item: EmbedItem) -> None:
    fits = (
        (kind is VectorKind.EDGE and isinstance(item.extras, EdgeExtras))
        or (kind is VectorKind.NODE and isinstance(item.extras, NodeExtras | None))
        or (kind is VectorKind.EPISODE and item.extras is None)
    )
    if not fits:
        msg = (
            f"item {item.id!r}: extras {type(item.extras).__name__} "
            f"do not fit kind {kind.value}"
        )
        raise ValueError(msg)


class VectorStore:
    """Stores one vector per object and finds the nearest ones by cosine.

    A vector is replaced only when the text it was made from changes. Failures
    that concern one item are listed in `WriteResult.failed`; other failures
    raise a typed error. An empty search result means no hits.

    Args:
        backend: Where the vectors live.
        gateway: The embedder, wrapped with retries and limits.
        settings: Batch sizes and the over-fetch factor.
        kinds: The kinds of vector this store serves.
        graph: The graph the vectors mirror. It re-checks past belief and
            confirms that a hit still has an object.
        rechecker: Decides which facts were believed at an instant in the
            past. Defaults to the graph's own answer.
    """

    def __init__(
        self,
        backend: Backend,
        gateway: EmbeddingGateway,
        settings: VectorStoreSettings | None = None,
        kinds: Sequence[VectorKind] = tuple(VectorKind),
        *,
        graph: GraphView | None = None,
        rechecker: Rechecker | None = None,
    ) -> None:
        """Bind a backend and an embedder; call `ensure_ready` before use."""
        self.backend = backend
        self.gateway = gateway
        self.graph = graph
        self.rechecker = rechecker
        self.settings = settings or gateway.settings
        self.metrics = Metrics()
        self.specs = {
            kind: SpaceSpec(
                kind,
                gateway.model_id,
                gateway.dim,
                TEMPLATES[kind].id,
                TEMPLATES[kind].version,
            )
            for kind in kinds
        }
        self._ready = False

    @property
    def capabilities(self) -> Capabilities:
        """Return what the backend can do."""
        return self.backend.capabilities

    async def ensure_ready(self) -> None:
        """Verify the backend and the embedding spaces.

        The store refuses reads and writes until this succeeds.

        Raises:
            SpaceMismatchError: If the embedder, the stored space or an index
                disagree with the configuration.
            IndexDriftError: If an index is missing or unusable.
        """
        self._ready = False
        await self.backend.ensure_ready(list(self.specs.values()))
        await SpaceRegistry(self.backend, self.gateway).ensure(self.specs)
        self._ready = True

    def _spec(self, kind: VectorKind) -> SpaceSpec:
        if not self._ready:
            msg = "call ensure_ready() first"
            raise StoreNotReadyError(msg)
        try:
            return self.specs[kind]
        except KeyError:
            msg = f"kind {kind.value} is not configured"
            raise ValueError(msg) from None

    async def upsert(self, kind: VectorKind, items: Sequence[EmbedItem]) -> WriteResult:
        """Embed and store vectors; a vector changes only if its text changed.

        When an id appears twice, the last item wins. Items that cannot be
        embedded, whose graph object does not exist, or that the database
        refuses are reported in `failed`.

        Raises:
            ValueError: If an item's extras do not fit `kind`.
        """
        self._spec(kind)
        template = TEMPLATES[kind]
        latest: dict[str, EmbedItem] = {}
        for item in items:
            _check_extras(kind, item)
            latest[item.id] = item
        failed: list[FailedItem] = []
        todo: list[tuple[EmbedItem, str, str]] = []
        for item in latest.values():
            rendered = template.render(item.text)
            if not rendered.strip():
                failed.append(FailedItem(item.id, FailReason.EMPTY_TEXT, "empty text"))
                continue
            todo.append((item, rendered, text_hash(template, rendered)))

        stored = await self._stored_hashes(kind, [t[0].id for t in todo])
        pending = [t for t in todo if stored.get(t[0].id) != t[2]]
        outcomes = await self.gateway.embed([t[1] for t in pending])
        rows: list[Row] = []
        for (item, _, digest), outcome in zip(pending, outcomes, strict=True):
            if outcome.vector is None:
                reason = outcome.reason or FailReason.EMBED_FAILED
                failed.append(FailedItem(item.id, reason, outcome.detail))
                continue
            rows.append(Row(item.id, digest, outcome.vector, item.extras))

        written = 0
        for batch in _chunks(rows, self.settings.write_batch):
            if self.graph is not None:
                present = await self.graph.exists(kind, [row.id for row in batch])
                missing = [row for row in batch if row.id not in present]
                failed.extend(
                    FailedItem(
                        row.id,
                        FailReason.MISSING_NODE,
                        "graph object does not exist",
                    )
                    for row in missing
                )
                write_batch = [row for row in batch if row.id in present]
            else:
                write_batch = batch
            if not write_batch:
                continue
            result = await self.backend.write(kind, write_batch)
            written += result.written
            failed.extend(
                FailedItem(i, FailReason.MISSING_NODE, "graph object does not exist")
                for i in result.missing
            )
            failed.extend(result.rejected)
        self.metrics.inc(f"{kind.value}.embedded", len(rows))
        self.metrics.inc(f"{kind.value}.unchanged", len(todo) - len(pending))
        self.metrics.inc(f"{kind.value}.failed", len(failed))
        return WriteResult(
            attempted=len(latest),
            embedded=len(rows),
            unchanged=len(todo) - len(pending),
            written=written,
            failed=tuple(failed),
        )

    async def needs_embedding(
        self, kind: VectorKind, items: Sequence[EmbedItem]
    ) -> list[EmbedItem]:
        """Return the items with no vector or a vector made from other text."""
        self._spec(kind)
        template = TEMPLATES[kind]
        stored = await self._stored_hashes(kind, [item.id for item in items])
        out: list[EmbedItem] = []
        for item in items:
            rendered = template.render(item.text)
            if not rendered.strip():
                continue
            if stored.get(item.id) != text_hash(template, rendered):
                out.append(item)
        return out

    async def update_validity(self, updates: Sequence[ValidityUpdate]) -> int:
        """Change only the validity mirror of existing edge rows.

        Raises:
            CapabilityError: If the backend has no validity mirror.
        """
        self._spec(VectorKind.EDGE)
        if not self.capabilities.validity_mirror:
            msg = "this backend does not store the validity of a fact"
            raise CapabilityError(msg)
        latest = {update.fact_id: update for update in updates}
        return await self.backend.update_validity(list(latest.values()))

    async def delete(self, kind: VectorKind, ids: Sequence[str]) -> int:
        """Remove the vectors of `ids`; ids without a vector are ignored."""
        self._spec(kind)
        removed = 0
        for part in _chunks(as_ids(ids), self.settings.id_batch):
            removed += await self.backend.delete(kind, part)
        return removed

    async def search(
        self,
        kind: VectorKind,
        vector: Sequence[float],
        *,
        k: int,
        filters: SearchFilters | None = None,
        min_score: float | None = None,
    ) -> list[Hit]:
        """Return up to `k` nearest objects, best first.

        Args:
            kind: Which vectors to search.
            vector: The query vector; it is scaled to unit length.
            k: Most hits to return.
            filters: Typed property filters, refused when they do not fit `kind`.
            min_score: Lowest score to keep, inclusive, applied after the cut
                to `k`.

        Returns:
            Hits sorted by score, then id. Empty means no hits.

        Raises:
            ValueError: If a filter does not apply to `kind`.
            SpaceMismatchError: If the query has the wrong length.
            InvalidVectorError: If the query is not finite or has zero norm.
            CapabilityError: If the filters need a validity mirror this backend
                does not have.
            IndexDriftError: If the index is missing or unusable.
            QueryTimeoutError: If the query exceeded its time limit.
        """
        spec = self._spec(kind)
        if k <= 0:
            return []
        if len(vector) != spec.dim:
            msg = f"query has {len(vector)} values, {kind.value} space has {spec.dim}"
            raise SpaceMismatchError(msg)
        chosen = (filters or SearchFilters()).for_kind(kind)
        k_fetch = fetch_size(
            k,
            chosen,
            min_score,
            self.settings.overfetch,
            backend_filters_rows=self.capabilities.filters_rows,
        )
        started = time.monotonic()
        try:
            temporal = chosen.temporal
            if temporal is not None and temporal.needs_graph:
                if not self.capabilities.validity_mirror or (
                    self.rechecker is None and self.graph is None
                ):
                    msg = "known_at search needs a validity mirror and graph re-check"
                    raise CapabilityError(msg)
                raw = await self._search_rechecked(
                    kind,
                    to_unit(vector, spec.dim),
                    k,
                    chosen,
                    min_score,
                )
            else:
                raw = await self.backend.search(
                    kind, to_unit(vector, spec.dim), k_fetch, chosen, min_score
                )
        finally:
            self.metrics.observe(f"search.{kind.value}", time.monotonic() - started)
        ordered = sorted(raw, key=lambda hit: (-hit[1], hit[0]))[:k]
        return [Hit(i, s, rank) for rank, (i, s) in enumerate(ordered)]

    async def _search_rechecked(
        self,
        kind: VectorKind,
        unit: tuple[float, ...],
        k: int,
        filters: SearchFilters,
        min_score: float | None,
    ) -> list[RawHit]:
        temporal = filters.temporal
        rechecker = self.rechecker
        if rechecker is None and self.graph is not None:
            rechecker = self.graph.still_valid
        if (
            temporal is None
            or not self.capabilities.validity_mirror
            or rechecker is None
        ):
            msg = "known_at search needs a validity mirror and graph re-check"
            raise CapabilityError(msg)
        history = replace(filters, temporal=TemporalFilter(history=True))
        k_fetch = fetch_size(
            k,
            filters,
            min_score,
            self.settings.overfetch,
            backend_filters_rows=self.capabilities.filters_rows,
        )
        kept: list[RawHit] = []
        for _ in range(self.settings.max_rechecks + 1):
            raw = await self.backend.search(kind, unit, k_fetch, history, min_score)
            valid = await rechecker([hit_id for hit_id, _ in raw], temporal)
            kept = [hit for hit in raw if hit[0] in valid]
            self.metrics.inc("search.recheck_rounds")
            if len(kept) >= k or len(raw) < k_fetch:
                break
            k_fetch *= 2
        return kept

    async def score_by_ids(
        self, kind: VectorKind, vector: Sequence[float], ids: Sequence[str]
    ) -> dict[str, float]:
        """Return the score of each id that has a vector; no cutoff.

        Raises:
            SpaceMismatchError: If the query has the wrong length.
            InvalidVectorError: If the query is not finite or has zero norm.
        """
        spec = self._spec(kind)
        if len(vector) != spec.dim:
            msg = f"query has {len(vector)} values, {kind.value} space has {spec.dim}"
            raise SpaceMismatchError(msg)
        unit = to_unit(vector, spec.dim)
        out: dict[str, float] = {}
        for part in _chunks(as_ids(ids), self.settings.id_batch):
            out.update(await self.backend.score_by_ids(kind, unit, part))
        return out

    async def exists(self, kind: VectorKind, ids: Sequence[str]) -> set[str]:
        """Return the ids that have a vector."""
        self._spec(kind)
        found: set[str] = set()
        for part in _chunks(as_ids(ids), self.settings.id_batch):
            found |= await self.backend.exists(kind, part)
        return found

    async def iter_ids(
        self, kind: VectorKind, after: str | None, limit: int
    ) -> list[str]:
        """Return a keyset page of stored ids.

        Args:
            kind: Which vectors to list.
            after: Return ids greater than this one; `None` starts at the first.
            limit: Most ids to return.

        Raises:
            ValueError: If `limit` is not positive.
        """
        self._spec(kind)
        if limit < 1:
            msg = "limit must be positive"
            raise ValueError(msg)
        return await self.backend.iter_ids(kind, after, limit)

    async def fetch_vectors(
        self, kind: VectorKind, ids: Sequence[str]
    ) -> dict[str, tuple[float, ...]]:
        """Return the stored unit vectors of `ids`, for dedupe and reranking."""
        self._spec(kind)
        out: dict[str, tuple[float, ...]] = {}
        for part in _chunks(as_ids(ids), self.settings.id_batch):
            out.update(await self.backend.fetch_vectors(kind, part))
        return out

    async def hydrate(
        self, kind: VectorKind, hits: Sequence[Hit]
    ) -> tuple[list[Hit], list[str]]:
        """Keep the hits that still have a vector; return `(kept, missing)`.

        The caller can enqueue `missing` for an orphan sweep.
        """
        present = (
            await self.graph.exists(kind, [hit.id for hit in hits])
            if self.graph is not None
            else await self.exists(kind, [hit.id for hit in hits])
        )
        kept = [hit for hit in hits if hit.id in present]
        missing = [hit.id for hit in hits if hit.id not in present]
        self.metrics.inc(f"{kind.value}.hydrate_missing", len(missing))
        return kept, missing

    async def _stored_hashes(
        self, kind: VectorKind, ids: Sequence[str]
    ) -> dict[str, str]:
        stored: dict[str, str] = {}
        for part in _chunks(ids, self.settings.id_batch):
            stored.update(await self.backend.existing_hashes(kind, part))
        return stored
