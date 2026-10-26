"""Run the fixed vector, keyword, entity and graph retrieval lanes."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial

from ontogr_mem.data_models import Entity, EntityStatus
from ontogr_mem.errors import StoreError
from ontogr_mem.ingestion.store import EpisodeStore
from ontogr_mem.llm import estimate_tokens
from ontogr_mem.resolution.errors import ResolutionError
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.retrieval.context import assemble
from ontogr_mem.retrieval.errors import (
    KeywordSearchError,
    RerankerError,
    RetrievalError,
)
from ontogr_mem.retrieval.fusion import fuse
from ontogr_mem.retrieval.keyword import KeywordSearch
from ontogr_mem.retrieval.lanes import LaneOutput
from ontogr_mem.retrieval.lanes.entity import run_entity_lane
from ontogr_mem.retrieval.lanes.graph import run_graph_lane
from ontogr_mem.retrieval.lanes.keyword import run_keyword_lane
from ontogr_mem.retrieval.lanes.vector import run_vector_lane
from ontogr_mem.retrieval.models import (
    EntityCard,
    EpisodeResult,
    FactResult,
    LaneStatus,
    RetrievalRequest,
    RetrievalResult,
)
from ontogr_mem.retrieval.ranking import apply_penalties
from ontogr_mem.retrieval.rerank import Passage, Reranker, blend
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.temporal.errors import TemporalError
from ontogr_mem.temporal.query import FactView, TemporalQuery
from ontogr_mem.temporal.reader import FactReader, ObservationText
from ontogr_mem.vectorstore.errors import VectorStoreError
from ontogr_mem.vectorstore.vector_store import VectorStore


_PORT_ERRORS = (
    VectorStoreError,
    KeywordSearchError,
    TemporalError,
    ResolutionError,
    StoreError,
)

FactKey = tuple[str, int]


def _fact_key(view: FactView) -> FactKey:
    """Return the identity of one visible version."""
    return view.fact_id, view.version


def _lane_names(
    key: FactKey, lane_lists: Sequence[tuple[str, Sequence[FactKey]]]
) -> tuple[str, ...]:
    """Return the lanes whose ranked lists contain this fact version."""
    return tuple(name for name, keys in lane_lists if key in keys)


def _port_error(exc: Exception) -> bool:
    """Return whether `exc` is an expected store or lane error."""
    return isinstance(exc, _PORT_ERRORS) and not isinstance(exc, ValueError)


class Retriever:
    """Retrieve ranked facts, episode passages and entity cards from one tenant."""

    def __init__(
        self,
        *,
        vectors: VectorStore,
        keywords: KeywordSearch,
        facts: FactReader,
        entities: ResolverStore,
        episodes: EpisodeStore,
        settings: RetrievalSettings | None = None,
        reranker: Reranker | None = None,
        count_tokens: Callable[[str], int] = estimate_tokens,
        relation_label: Callable[[int], str] | None = None,
    ) -> None:
        """Bind tenant-scoped stores and optional ranking and token counting.

        Args:
            vectors: Ready vector store whose gateway embeds the query once.
            keywords: Search port for observations, episodes, and entity aliases.
            facts: Reader that applies temporal visibility and conflict status.
            entities: Resolver store used for entity lookup and alias cards.
            episodes: Store used to hydrate episode passages.
            settings: Candidate, ranking, and context limits. Defaults to settings
                loaded from the environment.
            reranker: Optional passage scorer. Typed failures retain fused ranks.
            count_tokens: Counter used for context admission and final usage.
            relation_label: Returns a human-readable label for a relation type id.
                Omit it to show facts without a relation name.
        """
        self._vectors = vectors
        self._keywords = keywords
        self._facts = facts
        self._entities = entities
        self._episodes = episodes
        self._settings = settings or RetrievalSettings()
        self._reranker = reranker
        self._count_tokens = count_tokens
        self._relation_label = relation_label

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Run retrieval lanes and return ranked items with budgeted context.

        A failed search degrades its lane without discarding successful searches.
        The context budget does not truncate the typed result lists.

        Args:
            request: Query, temporal filters, result limits, and context budget.

        Returns:
            Ranked facts, episodes, entity cards, lane records, and context text.

        Raises:
            RetrievalError: If every initial lane fails or required fact or
                episode hydration fails.
        """
        temporal_query = request.to_temporal_query()
        vector_lane, keyword_lane, entity_lane = await self._initial_lanes(
            request, temporal_query
        )
        graph_lane = await run_graph_lane(
            self._facts,
            seeds=self._graph_seeds(
                vector_lane, keyword_lane, entity_lane, request.limit
            ),
            query=temporal_query,
            graph_depth=request.graph_depth,
            settings=self._settings,
            query_text=request.query,
            disputed=request.disputed,
        )
        fact_results = self._fused_facts(
            (vector_lane, keyword_lane, graph_lane), request.limit
        )
        notes: list[str] = []
        shortlist_limit = (
            self._settings.shortlist_factor * request.limit
            if self._reranker is not None
            else request.limit
        )
        episode_results = await self._episodes_result(
            vector_lane, keyword_lane, request.limit, shortlist_limit, notes
        )
        fact_results, episode_results = await self._rerank_results(
            request, fact_results, episode_results, shortlist_limit, notes
        )
        fact_results = apply_penalties(
            fact_results, limit=request.limit, settings=self._settings
        )
        fact_results = await self._add_evidence(fact_results, notes)
        entity_cards = await self._entity_cards(entity_lane, fact_results, notes)
        result = RetrievalResult(
            facts=tuple(fact_results),
            episodes=tuple(episode_results),
            entities=tuple(entity_cards),
            lanes=tuple(
                lane.record
                for lane in (vector_lane, keyword_lane, entity_lane, graph_lane)
            ),
            notes=tuple(notes),
        )
        return replace(
            result,
            context=assemble(
                result,
                request.now,
                request.budget_tokens,
                self._settings,
                self._count_tokens,
                relation_label=self._relation_label,
            ),
        )

    async def _initial_lanes(
        self, request: RetrievalRequest, temporal_query: TemporalQuery
    ) -> tuple[LaneOutput, LaneOutput, LaneOutput]:
        """Embed once and run the independent lanes with their own filters."""
        outcome = (await self._vectors.gateway.embed([request.query]))[0]
        embedding = outcome.vector
        embedding_error = outcome.detail or str(outcome.reason or "embedding failed")

        lanes = [
            partial(
                run_vector_lane,
                self._vectors,
                embedding,
                limit=request.limit,
                settings=self._settings,
                facts=self._facts,
                query=temporal_query,
                disputed=request.disputed,
                embedding_error=embedding_error,
            ),
            partial(
                run_keyword_lane,
                self._keywords,
                self._facts,
                query=request.query,
                temporal_query=temporal_query,
                limit=request.limit,
                settings=self._settings,
                disputed=request.disputed,
            ),
            partial(
                run_entity_lane,
                self._vectors,
                self._keywords,
                self._entities,
                embedding,
                query=request.query,
                mentions=request.mentions,
                settings=self._settings,
                embedding_error=embedding_error,
            ),
        ]
        try:
            vector_lane, keyword_lane, entity_lane = await asyncio.gather(
                *(lane() for lane in lanes)
            )
        except _PORT_ERRORS as exc:
            if isinstance(exc, ValueError):
                raise
            raise RetrievalError("fact hydration failed") from exc

        all_lane_errors = [
            lane.error
            for lane in (vector_lane, keyword_lane, entity_lane)
            if lane.error is not None
        ]
        failed_lanes = [
            lane
            for lane in (vector_lane, keyword_lane, entity_lane)
            if lane.record.status is LaneStatus.FAILED
        ]
        if len(failed_lanes) == 3:
            error = all_lane_errors[0] if all_lane_errors else None
            raise RetrievalError("all retrieval lanes failed") from error
        return vector_lane, keyword_lane, entity_lane

    def _graph_seeds(
        self,
        vector_lane: LaneOutput,
        keyword_lane: LaneOutput,
        entity_lane: LaneOutput,
        limit: int,
    ) -> list[str]:
        """Combine entity hits with endpoints from the leading fused facts."""
        vector_keys = [_fact_key(view) for view in vector_lane.facts]
        keyword_keys = [_fact_key(view) for view in keyword_lane.facts]
        provisional = fuse(
            [vector_keys, keyword_keys], limit=limit, settings=self._settings
        )
        seeds = list(entity_lane.entities)
        for key, _ in provisional[: self._settings.seed_fact_count]:
            view = _find_view(key, (vector_lane, keyword_lane))
            if view is not None:
                seeds.extend((view.subject_id, view.object_id))
        return list(dict.fromkeys(seeds))[: self._settings.seed_cap]

    def _fused_facts(self, lanes: Sequence[LaneOutput], limit: int) -> list[FactResult]:
        """Fuse fact versions and retain their lane provenance."""
        lane_lists = [
            (lane.record.name, [_fact_key(view) for view in lane.facts])
            for lane in lanes
        ]
        views = {_fact_key(view): view for lane in lanes for view in lane.facts}
        fused = fuse(
            [keys for _, keys in lane_lists],
            limit=limit,
            settings=self._settings,
        )
        return [
            FactResult(
                view=views[key],
                score=score,
                lanes=_lane_names(key, lane_lists),
            )
            for key, score in fused
        ]

    async def _rerank_results(
        self,
        request: RetrievalRequest,
        fact_results: list[FactResult],
        episode_results: list[EpisodeResult],
        shortlist_limit: int,
        notes: list[str],
    ) -> tuple[list[FactResult], list[EpisodeResult]]:
        """Apply the optional passage scorer to facts and episodes concurrently."""
        if self._reranker is not None:
            fact_scores = [
                (f"{fact.view.fact_id}@{fact.view.version}", fact.score)
                for fact in fact_results
            ]
            fact_passages = [
                Passage(
                    key,
                    fact.view.statement
                    or (
                        f"{fact.view.subject_id} {fact.view.relation_type_id} "
                        f"{fact.view.object_id}"
                    ),
                )
                for (key, _), fact in zip(
                    fact_scores[:shortlist_limit],
                    fact_results[:shortlist_limit],
                    strict=True,
                )
            ]
            episode_scores = [
                (episode.episode_id, episode.score) for episode in episode_results
            ]
            episode_passages = [
                Passage(
                    episode.episode_id, episode.text[: self._settings.passage_chars]
                )
                for episode in episode_results
            ]
            fact_rerank, episode_rerank = await asyncio.gather(
                self._rerank(request.query, fact_scores, fact_passages),
                self._rerank(request.query, episode_scores, episode_passages),
            )
            fact_values, fact_note = fact_rerank
            episode_values, episode_note = episode_rerank
            notes.extend(note for note in (fact_note, episode_note) if note is not None)
            fact_map = dict(fact_values)
            fact_results = [
                replace(fact, score=fact_map[key])
                for (key, _), fact in zip(fact_scores, fact_results, strict=True)
            ]
            episode_map = {episode.episode_id: episode for episode in episode_results}
            best = max((score for _, score in episode_values), default=0.0)
            episode_results = [
                replace(episode_map[key], score=score / best if best > 0 else score)
                for key, score in episode_values[: request.limit]
            ]

        return fact_results, episode_results

    async def _rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, float]],
        passages: Sequence[Passage],
    ) -> tuple[list[tuple[str, float]], str | None]:
        """Blend one result list, preserving fused scores on a typed failure."""
        if self._reranker is None or not passages:
            return list(candidates), None
        try:
            scores = await self._reranker.score(query, passages)
        except RerankerError as exc:
            return list(candidates), f"reranker failed: {exc}"
        shortlist = {passage.id for passage in passages}
        return blend(
            candidates,
            {key: value for key, value in scores.items() if key in shortlist},
            weight=self._settings.rerank_weight,
        ), None

    async def _add_evidence(
        self, facts: Sequence[FactResult], notes: list[str]
    ) -> list[FactResult]:
        """Attach source episodes and the first available evidence quote."""
        observation_ids = list(
            dict.fromkeys(obs for fact in facts for obs in fact.view.evidence)
        )
        if not observation_ids:
            return list(facts)
        try:
            texts = await self._facts.observation_texts(observation_ids)
        except Exception as exc:
            if not _port_error(exc):
                raise
            notes.append(f"evidence unavailable: {exc}")
            return list(facts)
        return [
            replace(
                fact,
                episode_ids=_episode_ids(fact.view, texts),
                quote=_first_quote(fact.view, texts),
            )
            for fact in facts
        ]

    async def _episodes_result(
        self,
        vector_lane: LaneOutput,
        keyword_lane: LaneOutput,
        limit: int,
        fetch_limit: int,
        notes: list[str],
    ) -> list[EpisodeResult]:
        """Hydrate and return the fused episode passages."""
        lane_lists = [vector_lane.episodes, keyword_lane.episodes]
        fused = fuse(lane_lists, limit=limit, settings=self._settings)
        if not fused:
            return []
        try:
            episode_ids = [item_id for item_id, _ in fused]
            episodes = await self._episodes.get_episodes(episode_ids)
        except Exception as exc:
            if not _port_error(exc):
                raise
            raise RetrievalError("episode hydration failed") from exc
        found = {episode.id: episode for episode in episodes}
        missing = len(fused) - len(found)
        if missing:
            notes.append(f"dropped {missing} missing episode(s)")
        existing_lists = [
            [item_id for item_id in lane if item_id in found] for lane in lane_lists
        ]
        refusion = fuse(existing_lists, limit=limit, settings=self._settings)
        result: list[EpisodeResult] = []
        for episode_id, score in refusion[:fetch_limit]:
            episode = found[episode_id]
            lanes = tuple(
                name
                for name, lane in zip(
                    ("vector", "keyword"), existing_lists, strict=True
                )
                if episode_id in lane
            )
            result.append(
                EpisodeResult(
                    episode_id=episode.id,
                    text=episode.text,
                    occurred_at=episode.occurred_at,
                    source_id=episode.source.id,
                    score=score,
                    lanes=lanes,
                )
            )
        return result

    async def _entity_cards(
        self,
        entity_lane: LaneOutput,
        facts: Sequence[FactResult],
        notes: list[str],
    ) -> list[EntityCard]:
        """Load cards for entity-lane hits, then for fact endpoints."""
        source_map = {
            key: list(value) for key, value in entity_lane.entity_sources.items()
        }
        ordered_ids = list(entity_lane.entities)
        for fact in facts:
            for entity_id in (fact.view.subject_id, fact.view.object_id):
                if entity_id not in ordered_ids:
                    ordered_ids.append(entity_id)
                source_map.setdefault(entity_id, [])
                if "fact" not in source_map[entity_id]:
                    source_map[entity_id].append("fact")
        if not ordered_ids:
            return []
        try:
            entities = await self._ordered_entities(ordered_ids)
        except Exception as exc:
            if not _port_error(exc):
                raise
            notes.append(f"entity cards unavailable: {exc}")
            return []
        cards: list[EntityCard] = []
        for entity in entities:
            try:
                aliases = await self._entities.aliases_of(entity.entity_id)
            except Exception as exc:
                if not _port_error(exc):
                    raise
                notes.append(f"aliases unavailable for {entity.entity_id}: {exc}")
                aliases = []
            cards.append(
                EntityCard(
                    entity_id=entity.entity_id,
                    name=entity.canonical_name,
                    type_label=entity.type_label,
                    aliases=tuple(alias.text for alias in aliases),
                    summary=entity.summary,
                    lanes=tuple(source_map.get(entity.entity_id, ("fact",))),
                )
            )
            if len(cards) == self._settings.max_cards:
                break
        return cards

    async def _ordered_entities(self, ids: Sequence[str]) -> list[Entity]:
        """Resolve merged ids and preserve the first candidate's position."""
        direct = await self._entities.get_entities(ids)
        known = {entity.entity_id: entity for entity in direct}
        while True:
            missing = sorted(
                {
                    entity.merged_into
                    for entity in known.values()
                    if entity.merged_into is not None
                    and entity.merged_into not in known
                }
            )
            if not missing:
                break
            fetched = await self._entities.get_entities(missing)
            if not fetched:
                break
            known.update({entity.entity_id: entity for entity in fetched})

        ordered: list[Entity] = []
        seen: set[str] = set()
        for entity_id in ids:
            current = known.get(entity_id)
            visited: set[str] = set()
            while current is not None and current.merged_into is not None:
                if current.entity_id in visited:
                    current = None
                    break
                visited.add(current.entity_id)
                current = known.get(current.merged_into)
            if (
                current is not None
                and current.status is EntityStatus.ACTIVE
                and current.entity_id not in seen
            ):
                ordered.append(current)
                seen.add(current.entity_id)
        return ordered


def _find_view(key: FactKey, lanes: Sequence[LaneOutput]) -> FactView | None:
    """Find the hydrated view for one fact key in the named lanes."""
    return next(
        (view for lane in lanes for view in lane.facts if _fact_key(view) == key), None
    )


def _episode_ids(view: FactView, texts: dict[str, ObservationText]) -> tuple[str, ...]:
    """Return unique supporting episode ids in evidence order."""
    return tuple(
        dict.fromkeys(
            texts[obs_id].episode_id for obs_id in view.evidence if obs_id in texts
        )
    )


def _first_quote(view: FactView, texts: dict[str, ObservationText]) -> str:
    """Return the first non-empty evidence quote."""
    return next(
        (
            texts[obs_id].quote
            for obs_id in view.evidence
            if obs_id in texts and texts[obs_id].quote
        ),
        "",
    )
