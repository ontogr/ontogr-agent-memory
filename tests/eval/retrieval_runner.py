"""Report lane contributions on the approved synthetic retrieval labels.

Run with ``uv run python -m tests.eval.retrieval_runner``. Mock vectors prove
wiring, not semantic relevance. The small fixture does not estimate real quality.
"""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from statistics import mean
from typing import TypedDict

from ontogr_mem.graphstore.memory import MemoryResolverStore
from ontogr_mem.retrieval import (
    KeywordKind,
    Passage,
    RetrievalResult,
    RetrievalSettings,
    Retriever,
)
from ontogr_mem.retrieval.text import terms
from ontogr_mem.vectorstore import VectorKind
from tests.eval.retrieval_labelled import LABELLED, LabelledQuery
from tests.eval.retrieval_ports import SilentKeywords, SilentVectors
from tests.support.retrieval_world import RetrievalWorld, build_world


Metrics = tuple[float, float, float]
CONFIGURATIONS = (
    "vector_only",
    "keyword_only",
    "entity_graph",
    "fused",
    "fused_reranked",
    "fused_no_penalties",
    "fused_importance",
)
IMPORTANCE = {
    "F1": 0.8,
    "F2": 0.2,
    "F3": 0.6,
    "F4": 0.1,
    "F5": 0.9,
    "F6": 0.3,
    "F7": 0.7,
    "F8": 0.4,
    "F9": 0.5,
    "F10": 1.0,
    "F11": 0.0,
}


class MetricSummary(TypedDict):
    """Rounded ranking means and the number of nonempty targets."""

    hit_at_1: float
    recall_at_5: float
    mrr: float
    queries: int


class ConfigurationReport(TypedDict):
    """Fact and episode metrics separated by query scope."""

    facts: dict[str, MetricSummary]
    episodes: dict[str, MetricSummary]
    missing_facts_at_5: list[str]


class AblationReport(TypedDict):
    """Deterministic report with its evaluation limitations."""

    limitation: str
    configurations: dict[str, ConfigurationReport]


def list_metrics(ranked: Sequence[str], expected: Sequence[str]) -> Metrics | None:
    """Return hit at one, recall at five, and first reciprocal rank.

    An empty expected list is unscored, rather than treated as a perfect result.
    """
    targets = set(expected)
    if not targets:
        return None
    first = float(bool(ranked) and ranked[0] in targets)
    recall = len(targets.intersection(ranked[:5])) / len(targets)
    reciprocal = next(
        (1 / rank for rank, key in enumerate(ranked, 1) if key in targets), 0.0
    )
    return first, recall, reciprocal


def mean_metrics(samples: Sequence[Metrics | None]) -> MetricSummary:
    """Average scored lists only, rounded to four decimal places."""
    scored = [sample for sample in samples if sample is not None]
    return MetricSummary(
        hit_at_1=round(mean(sample[0] for sample in scored), 4) if scored else 0.0,
        recall_at_5=round(mean(sample[1] for sample in scored), 4) if scored else 0.0,
        mrr=round(mean(sample[2] for sample in scored), 4) if scored else 0.0,
        queries=len(scored),
    )


class _TermReranker:
    """Score passages by shared query terms without consulting expected labels."""

    async def score(
        self, query: str, passages: Sequence[Passage]
    ) -> Mapping[str, float]:
        query_terms = set(terms(query))
        return {
            passage.id: float(len(query_terms.intersection(terms(passage.text))))
            for passage in passages
        }


def _retriever(world: RetrievalWorld, configuration: str) -> Retriever:
    """Build a read-only configuration without changing production retrieval."""
    vector_kinds = frozenset[VectorKind]()
    keyword_kinds = frozenset[KeywordKind]()
    if configuration == "vector_only":
        vector_kinds = frozenset({VectorKind.NODE})
        keyword_kinds = frozenset(KeywordKind)
    elif configuration == "keyword_only":
        vector_kinds = frozenset(VectorKind)
        keyword_kinds = frozenset({KeywordKind.ENTITY})
    elif configuration == "entity_graph":
        vector_kinds = frozenset({VectorKind.EDGE, VectorKind.EPISODE})
        keyword_kinds = frozenset({KeywordKind.OBSERVATION, KeywordKind.EPISODE})
    settings = RetrievalSettings()
    if configuration in {"vector_only", "keyword_only"}:
        settings = settings.model_copy(update={"seed_cap": 0})
    if configuration == "fused_no_penalties":
        settings = settings.model_copy(
            update={"disputed_penalty": 1.0, "overridden_penalty": 1.0}
        )
    return Retriever(
        vectors=SilentVectors(world.vectors, kinds=vector_kinds),
        keywords=SilentKeywords(world.keywords, kinds=keyword_kinds),
        facts=world.facts_reader,
        entities=(
            MemoryResolverStore()
            if configuration in {"vector_only", "keyword_only"}
            else world.resolver_store
        ),
        episodes=world.episodes,
        settings=settings,
        reranker=_TermReranker() if configuration == "fused_reranked" else None,
    )


def _cohorts(
    samples: Sequence[tuple[LabelledQuery, Metrics | None]],
) -> dict[str, MetricSummary]:
    """Separate all queries, focused queries, and broad entity summaries."""
    return {
        "all": mean_metrics([metrics for _, metrics in samples]),
        "non_overview": mean_metrics(
            [metrics for item, metrics in samples if not item.overview]
        ),
        "overview": mean_metrics(
            [metrics for item, metrics in samples if item.overview]
        ),
    }


def _ranked_facts(
    result: RetrievalResult, configuration: str, importance: Mapping[str, float]
) -> list[str]:
    """Return distinct fact ids, applying only the fixture importance variant."""
    facts = result.facts
    if configuration == "fused_importance":
        facts = tuple(
            sorted(
                (
                    replace(
                        fact,
                        score=fact.score
                        * (0.75 + 0.5 * importance.get(fact.view.fact_id, 0.5)),
                    )
                    for fact in facts
                ),
                key=lambda fact: (-fact.score, fact.view.fact_id, fact.view.version),
            )
        )
    return list(dict.fromkeys(fact.view.fact_id for fact in facts))


async def run_ablation(world: RetrievalWorld) -> AblationReport:
    """Run every fixed configuration over the owner-reviewed labels.

    Importance is a fixture-only multiplier. Neither defaults nor stored scores
    change. Queries without expected ids are omitted from that list's averages.
    """
    configurations: dict[str, ConfigurationReport] = {}
    importance = {world.facts[key]: value for key, value in IMPORTANCE.items()}
    for configuration in CONFIGURATIONS:
        retriever = _retriever(world, configuration)
        fact_samples: list[tuple[LabelledQuery, Metrics | None]] = []
        episode_samples: list[tuple[LabelledQuery, Metrics | None]] = []
        missing: list[str] = []
        for item in LABELLED:
            result = await retriever.retrieve(
                world.request(item.text, limit=10, **item.request_options)
            )
            ranked = _ranked_facts(result, configuration, importance)
            expected = [world.facts[key] for key in item.expected_facts]
            fact_samples.append((item, list_metrics(ranked, expected)))
            if set(expected) - set(ranked[:5]):
                missing.append(item.id)
            episode_samples.append(
                (
                    item,
                    list_metrics(
                        [episode.episode_id for episode in result.episodes],
                        [name.lower() for name in item.expected_episodes],
                    ),
                )
            )
        configurations[configuration] = ConfigurationReport(
            facts=_cohorts(fact_samples),
            episodes=_cohorts(episode_samples),
            missing_facts_at_5=missing,
        )
    return AblationReport(
        limitation=(
            "Approved synthetic labels and mock vectors prove wiring, not semantic "
            "relevance or real-world retrieval quality. Importance is a stand-in."
        ),
        configurations=configurations,
    )


async def main() -> None:
    """Print the report as JSON and a compact non-overview fact table."""
    report = await run_ablation(await build_world())
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nconfiguration          hit@1 recall@5 mrr")
    for name, configuration in report["configurations"].items():
        metrics = configuration["facts"]["non_overview"]
        print(
            f"{name:<22} {metrics['hit_at_1']:.4f} "
            f"{metrics['recall_at_5']:.4f} {metrics['mrr']:.4f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
