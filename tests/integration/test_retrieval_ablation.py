"""Repeatable lane ablation on the approved deterministic retrieval world."""

from ontogr_mem.retrieval import KeywordKind, LaneStatus, Retriever
from ontogr_mem.vectorstore import VectorKind
from tests.eval.retrieval_ports import SilentKeywords, SilentVectors
from tests.eval.retrieval_runner import run_ablation
from tests.support.graph_backend import GraphBackend
from tests.support.retrieval_backends import graph_retrieval_backend
from tests.support.retrieval_world import VECTOR, build_world


class TestRetrievalAblation:
    """Compare the report and prove that disabled lanes cannot retrieve data."""

    async def test_disabled_ports_and_lanes_return_nothing(
        self, graph_backend: GraphBackend
    ):
        """Suppressed reads leave every result empty and graph expansion skipped."""
        world = await build_world(await graph_retrieval_backend(graph_backend))
        vectors = SilentVectors(world.vectors, kinds=frozenset(VectorKind))
        keywords = SilentKeywords(world.keywords, kinds=frozenset(KeywordKind))
        for kind in VectorKind:
            assert await vectors.search(kind, VECTOR, k=10) == []
        for kind in KeywordKind:
            assert await keywords.search(kind, "Ana", limit=10) == []
        retriever = Retriever(
            vectors=vectors,
            keywords=keywords,
            entities=(await graph_backend.open_stores()).resolver_store,
            facts=world.facts_reader,
            episodes=world.episodes,
        )
        result = await retriever.retrieve(world.request(mentions=("Ana",)))
        assert result.facts == ()
        assert result.episodes == ()
        assert result.entities == ()
        assert [record.status for record in result.lanes] == [
            LaneStatus.EMPTY,
            LaneStatus.EMPTY,
            LaneStatus.EMPTY,
            LaneStatus.SKIPPED,
        ]

    async def test_report_repeats_and_fusion_retains_single_lane_rank(
        self, graph_backend: GraphBackend
    ):
        """The report repeats exactly and names the metrics each lane adds."""
        world = await build_world(await graph_retrieval_backend(graph_backend))
        empty_resolver = (await graph_backend.open_stores()).resolver_store
        report = await run_ablation(world, empty_resolver=empty_resolver)
        assert await run_ablation(world, empty_resolver=empty_resolver) == report

        configurations = report["configurations"]
        fused = configurations["fused"]["facts"]["non_overview"]
        fused_episodes = configurations["fused"]["episodes"]["non_overview"]
        best_single = max(
            configurations[name]["facts"]["non_overview"]["mrr"]
            for name in ("vector_only", "keyword_only", "entity_graph")
        )

        assert fused == {
            "hit_at_1": 0.6471,
            "recall_at_5": 0.8824,
            "mrr": 0.7717,
            "queries": 17,
        }
        assert fused["mrr"] >= best_single, (
            "Fusion trails the best single lane. Report the finding to the owner; "
            "do not lower the threshold or alter the labels."
        )
        assert fused_episodes["recall_at_5"] > max(
            configurations[name]["episodes"]["non_overview"]["recall_at_5"]
            for name in ("vector_only", "keyword_only")
        )
        assert "paraphrase-01" in configurations["keyword_only"]["missing_facts_at_5"]
        assert (
            "paraphrase-01" not in configurations["vector_only"]["missing_facts_at_5"]
        )
        assert "multi-hop-01" in configurations["vector_only"]["missing_facts_at_5"]
        assert (
            "multi-hop-01" not in configurations["keyword_only"]["missing_facts_at_5"]
        )
        assert "paraphrase-01" not in configurations["fused"]["missing_facts_at_5"]
        assert "multi-hop-01" not in configurations["fused"]["missing_facts_at_5"]
        assert (
            configurations["fused_reranked"]["episodes"]["non_overview"]["mrr"]
            > fused_episodes["mrr"]
        )
        assert (
            configurations["fused_no_penalties"]["facts"]["non_overview"]["recall_at_5"]
            > fused["recall_at_5"]
        )
        assert (
            configurations["fused_importance"]["facts"]["non_overview"]["mrr"]
            < fused["mrr"]
        )
