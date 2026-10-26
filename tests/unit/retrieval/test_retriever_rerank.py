"""Public Retriever behavior when optional passage reranking is enabled."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import pytest

from ontogr_mem.llm import estimate_tokens
from ontogr_mem.retrieval import (
    ContextBlock,
    FactResult,
    Passage,
    Reranker,
    RerankerError,
    RetrievalSettings,
    Retriever,
)
from tests.support.retrieval_world import QUERY, RetrievalWorld, build_world


class _ReverseReranker:
    """Give later passages higher scores and retain each scored lane."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Passage, ...]]] = []

    async def score(
        self, query: str, passages: Sequence[Passage]
    ) -> Mapping[str, float]:
        self.calls.append((query, tuple(passages)))
        return {passage.id: float(rank) for rank, passage in enumerate(passages)}


class _FailingReranker:
    """Raise one chosen exception from the optional scoring port."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def score(
        self, query: str, passages: Sequence[Passage]
    ) -> Mapping[str, float]:
        raise self.error


class _MappedReranker:
    """Return configured scores and a neutral score for other passages."""

    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = scores

    async def score(
        self, query: str, passages: Sequence[Passage]
    ) -> Mapping[str, float]:
        return {passage.id: self.scores.get(passage.id, 0.5) for passage in passages}


def _retriever(
    world: RetrievalWorld,
    *,
    settings: RetrievalSettings | None = None,
    reranker: Reranker | None = None,
    count_tokens: Callable[[str], int] = estimate_tokens,
) -> Retriever:
    """Bind a second Retriever to the real stores in one memory world."""
    return Retriever(
        vectors=world.vectors,
        keywords=world.keywords,
        facts=world.facts_reader,
        entities=world.resolver_store,
        episodes=world.episodes,
        settings=settings,
        reranker=reranker,
        count_tokens=count_tokens,
    )


def _fact_key(fact: FactResult) -> str:
    """Return the public passage identity for one result fact."""
    return f"{fact.view.fact_id}@{fact.view.version}"


class TestRetrieverRerank:
    """Verify reranking through the public retrieval interface."""

    async def test_reranks_fact_and_episode_results_and_bounds_episode_passages(self):
        """Both result lists reverse independently and episode text is truncated."""
        world = await build_world()
        [episode] = await world.episodes.get_episodes(["e1"])
        await world.episodes.write_episodes(
            [replace(episode, content=f"{QUERY} {'x' * 500}")]
        )
        reranker = _ReverseReranker()
        settings = RetrievalSettings(
            passage_chars=32,
            rerank_weight=1.0,
            disputed_penalty=1.0,
            overridden_penalty=1.0,
        )
        request = world.request(limit=10)

        result = await _retriever(world, settings=settings, reranker=reranker).retrieve(
            request
        )

        fact_call = next(call for call in reranker.calls if "@" in call[1][0].id)
        episode_call = next(call for call in reranker.calls if "@" not in call[1][0].id)
        fact_passages = fact_call[1]
        episode_passages = episode_call[1]
        assert len(fact_passages) > 1
        assert len(episode_passages) > 1
        assert [_fact_key(fact) for fact in result.facts] == [
            passage.id for passage in reversed(fact_passages)
        ][: request.limit]
        assert [episode.episode_id for episode in result.episodes] == [
            passage.id for passage in reversed(episode_passages)
        ][: request.limit]
        assert (
            next(passage.text for passage in episode_passages if passage.id == "e1")
            == f"{QUERY} {'x' * 500}"[: settings.passage_chars]
        )
        assert all(
            len(passage.text) <= settings.passage_chars for passage in episode_passages
        )

    async def test_typed_reranker_failure_keeps_rankings_and_lane_records(self):
        """A recoverable scoring failure leaves ordinary results and lanes intact."""
        world = await build_world()
        request = world.request(limit=5)
        plain = await world.retriever.retrieve(request)
        failed = await _retriever(
            world, reranker=_FailingReranker(RerankerError("offline"))
        ).retrieve(request)

        assert [_fact_key(fact) for fact in failed.facts] == [
            _fact_key(fact) for fact in plain.facts
        ]
        assert [episode.episode_id for episode in failed.episodes] == [
            episode.episode_id for episode in plain.episodes
        ]
        assert failed.lanes == plain.lanes
        assert sum(note == "reranker failed: offline" for note in failed.notes) == 2

    async def test_unexpected_reranker_value_error_propagates(self):
        """Only the typed reranker failure is handled as a recoverable error."""
        world = await build_world()

        with pytest.raises(ValueError, match="bad score"):
            await _retriever(
                world, reranker=_FailingReranker(ValueError("bad score"))
            ).retrieve(world.request(limit=5))

    async def test_disputed_penalty_changes_a_reranked_tie_after_blending(self):
        """A tie from reranking breaks in favor of the active fact after penalty."""
        world = await build_world()
        request = world.request(limit=100)
        neutral_settings = RetrievalSettings(
            rerank_weight=1.0,
            disputed_penalty=1.0,
            overridden_penalty=1.0,
        )
        active_key = f"{world.facts['F1']}@1"
        disputed_key = f"{world.facts['F4']}@1"
        reranker = _MappedReranker({active_key: 1.0, disputed_key: 1.0})

        tied = await _retriever(
            world, settings=neutral_settings, reranker=reranker
        ).retrieve(request)
        penalized_settings = neutral_settings.model_copy(
            update={"disputed_penalty": 0.8}
        )
        penalized = await _retriever(
            world, settings=penalized_settings, reranker=reranker
        ).retrieve(request)
        tied_by_id = {_fact_key(fact): fact for fact in tied.facts}
        penalized_by_id = {_fact_key(fact): fact for fact in penalized.facts}
        assert tied_by_id[disputed_key].view.disputed
        assert tied_by_id[active_key].score == 1.0
        assert tied_by_id[disputed_key].score == 1.0
        assert penalized_by_id[active_key].score == 1.0
        assert penalized_by_id[disputed_key].score == 0.8
        assert [_fact_key(fact) for fact in penalized.facts].index(active_key) < [
            _fact_key(fact) for fact in penalized.facts
        ].index(disputed_key)

    async def test_custom_token_counter_controls_admission_and_context_is_public(self):
        """The caller's counter controls context admission and result usage."""
        world = await build_world()
        request = world.request(budget_tokens=2000)
        plain = await world.retriever.retrieve(request)

        def count_tokens(text: str) -> int:
            if "Ana works at Acme" in text:
                return 100_000
            return estimate_tokens(text)

        custom = await _retriever(world, count_tokens=count_tokens).retrieve(request)

        assert isinstance(custom.context, ContextBlock)
        assert "Ana works at Acme" in plain.context.text
        assert "Ana works at Acme" not in custom.context.text
        assert custom.context.dropped > 0
        assert custom.context.tokens_used == count_tokens(custom.context.text)

    async def test_omitting_reranker_preserves_the_default_rank(self):
        """The optional reranker leaves the ordinary result order unchanged."""
        world = await build_world()
        request = world.request(limit=8)
        default = await world.retriever.retrieve(request)
        explicit_none = await _retriever(world, reranker=None).retrieve(request)

        assert [_fact_key(fact) for fact in explicit_none.facts] == [
            _fact_key(fact) for fact in default.facts
        ]
        assert [episode.episode_id for episode in explicit_none.episodes] == [
            episode.episode_id for episode in default.episodes
        ]
        assert isinstance(explicit_none.context, ContextBlock)
