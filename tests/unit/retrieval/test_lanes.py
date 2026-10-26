"""Behavior of the retrieval lanes over small in-memory ports."""

from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models import Entity, EntityStatus, FactStatus
from ontogr_mem.retrieval import (
    KeywordHit,
    KeywordKind,
    LaneStatus,
    RetrievalSettings,
)
from ontogr_mem.retrieval.errors import KeywordTimeoutError
from ontogr_mem.retrieval.lanes.entity import run_entity_lane
from ontogr_mem.retrieval.lanes.keyword import run_keyword_lane
from ontogr_mem.retrieval.lanes.vector import run_vector_lane
from ontogr_mem.temporal.query import FactView, TemporalQuery
from ontogr_mem.vectorstore import Hit, VectorKind
from ontogr_mem.vectorstore.errors import QueryTimeoutError


NOW = datetime(2026, 10, 23, tzinfo=UTC)
SETTINGS = RetrievalSettings()


def entity(
    entity_id: str, name: str, status: EntityStatus = EntityStatus.ACTIVE
) -> Entity:
    """Build a resolver entity for a lane result."""
    return Entity(
        entity_id=entity_id,
        type_label="Person",
        canonical_name=name,
        strict_key=name.casefold(),
        loose_key=name.casefold(),
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


def view(
    fact_id: str,
    subject_id: str = "a",
    object_id: str = "b",
    evidence: tuple[str, ...] = (),
) -> FactView:
    """Build a visible fact for a graph walk."""
    return FactView(
        fact_id=fact_id,
        version=1,
        slot_key=f"slot-{fact_id}",
        subject_id=subject_id,
        object_id=object_id,
        relation_type_id=1,
        valid_from=NOW,
        valid_to=None,
        valid_from_precision=None,
        valid_to_precision=None,
        recorded_at=NOW,
        expired_at=None,
        status=FactStatus.ACTIVE,
        evidence=evidence,
        statement=f"{subject_id} knows {object_id}",
    )


class VectorPort:
    """Fixed vector search results."""

    def __init__(self, failing: set[VectorKind] | None = None) -> None:
        """Choose the vector kinds that should raise a timeout."""
        self.failing = failing or set()

    async def search(self, kind, vector, *, k, filters=None, min_score=None):
        """Return fixed hits or raise for a configured kind."""
        if kind in self.failing:
            raise QueryTimeoutError(f"{kind.value} timed out")
        if kind is VectorKind.EDGE:
            return [Hit("f2", 0.9, 0), Hit("f1", 0.8, 1)]
        if kind is VectorKind.EPISODE:
            return [Hit("e2", 0.9, 0)]
        return [Hit("entity-1", 0.9, 0)]


class KeywordPort:
    """Fixed keyword search results."""

    async def search(self, kind, query, *, limit):
        """Return fixed hits for each keyword kind."""
        if kind is KeywordKind.OBSERVATION:
            return [KeywordHit("o2", 2.0, 0), KeywordHit("o1", 1.0, 1)]
        if kind is KeywordKind.EPISODE:
            return [KeywordHit("e1", 1.0, 0)]
        return [KeywordHit("entity-2", 1.0, 0)]


class FactPort:
    """Reader with the facts addressed by observations and graph endpoints."""

    async def by_observations(self, ids, query):
        """Return facts supported by the fixed observation ids."""
        return [view("f1", evidence=("o1",)), view("f2", evidence=("o2", "o1"))]

    async def by_ids(self, ids, query):
        """Return facts for the requested ids."""
        return [view(fact_id) for fact_id in ids]

    async def touching(self, ids, query):
        """Return facts touching the requested entities."""
        return [view("f1", "a", "b"), view("f2", "b", "c")]


class EntityPort:
    """Resolver with strict and loose mention candidates."""

    async def entities_by_strict_key(self, key):
        """Return the exact-key candidate when the key matches."""
        return [entity("entity-2", "Zurich")] if key == "zurich" else []

    async def entities_by_loose_key(self, key):
        """Return the loose-key candidate when the key matches."""
        return [entity("entity-1", "Zürich")] if key == "zurich" else []

    async def get_entities(self, ids, *, follow_merged=False):
        """Return the fixed set of entity cards."""
        return [
            entity("entity-1", "Zürich"),
            entity("entity-2", "Zurich"),
            entity("entity-3", "Archived", EntityStatus.ARCHIVED),
        ]


class TestVectorLane:
    """Vector lane search and typed-error behavior."""

    async def test_keeps_edge_and_episode_ranks_and_degrades_partially(self):
        """An edge timeout preserves episode hits and marks partial failure."""
        output = await run_vector_lane(
            VectorPort({VectorKind.EDGE}),
            [0.1, 0.2, 0.3],
            limit=2,
            settings=SETTINGS,
            facts=FactPort(),
            query=TemporalQuery(now=NOW),
        )

        assert output.facts == ()
        assert output.episodes == ("e2",)
        assert output.record.status is LaneStatus.OK
        assert output.record.failed == ("edge",)
        assert "edge" in output.record.detail

    async def test_reports_failed_when_both_searches_fail(self):
        """Two failed vector searches fail the complete lane."""
        output = await run_vector_lane(
            VectorPort({VectorKind.EDGE, VectorKind.EPISODE}),
            [0.1, 0.2, 0.3],
            limit=2,
            settings=SETTINGS,
        )

        assert output.record.status is LaneStatus.FAILED
        assert output.record.failed == ("edge", "episode")

    async def test_duplicate_history_hits_keep_the_best_first_rank(self):
        """A later version hit does not demote the same fact's best rank."""

        class InterleavedHits(VectorPort):
            async def search(self, kind, vector, *, k, filters=None, min_score=None):
                if kind is VectorKind.EDGE:
                    return [Hit("f2", 0.99, 0), Hit("f1", 0.8, 1), Hit("f2", 0.7, 2)]
                return []

        output = await run_vector_lane(
            InterleavedHits(),
            [0.1, 0.2, 0.3],
            limit=2,
            settings=SETTINGS,
            facts=FactPort(),
            query=TemporalQuery(now=NOW),
        )

        assert [item.fact_id for item in output.facts] == ["f2", "f1"]

    async def test_propagates_value_error_from_a_port(self):
        """The vector lane does not hide invalid request errors."""

        class InvalidVectorPort(VectorPort):
            async def search(self, kind, vector, *, k, filters=None, min_score=None):
                raise ValueError("invalid filter")

        with pytest.raises(ValueError, match="invalid filter"):
            await run_vector_lane(
                InvalidVectorPort(), [0.1, 0.2, 0.3], limit=2, settings=SETTINGS
            )


class TestKeywordLane:
    """Keyword lane maps observation hits to distinct fact ids."""

    async def test_orders_facts_by_best_observation_rank_and_keeps_episodes(self):
        """Best evidence rank orders facts and episode hits remain available."""
        output = await run_keyword_lane(
            KeywordPort(),
            FactPort(),
            limit=2,
            query="Acme",
            temporal_query=TemporalQuery(now=NOW),
            settings=SETTINGS,
        )

        assert [fact.fact_id for fact in output.facts] == ["f2", "f1"]
        assert output.observation_ids == ("o2", "o1")
        assert output.episodes == ("e1",)
        assert output.record.status is LaneStatus.OK


class TestEntityLane:
    """Entity lane combines explicit mentions, text and node vectors."""

    async def test_uses_loose_candidates_without_merging_identity(self):
        """Loose-key candidates join strict results without identity merging."""
        output = await run_entity_lane(
            VectorPort(),
            KeywordPort(),
            EntityPort(),
            [0.1, 0.2, 0.3],
            query="Zürich",
            mentions=("Zurich",),
            settings=SETTINGS,
        )

        assert "entity-1" in output.entities
        assert output.entity_sources["entity-1"] == ("mention", "vector")
        assert "entity-2" in output.entities
        assert output.entity_sources["entity-2"] == ("mention", "keyword")
        assert "entity-3" not in output.entities
        assert output.record.status is LaneStatus.OK

    async def test_missing_embedding_degrades_only_the_vector_search(self):
        """A missing embedding fails node search but keeps other candidates."""
        output = await run_entity_lane(
            VectorPort(),
            KeywordPort(),
            EntityPort(),
            None,
            query="Zürich",
            mentions=(),
            settings=SETTINGS,
            embedding_error="model unavailable",
        )

        assert output.record.status is LaneStatus.OK
        assert output.record.failed == ("node",)
        assert output.entities

    async def test_no_mentions_counts_only_keyword_and_node_searches(self):
        """Both actual entity searches failing marks the lane failed."""

        class FailingKeywordPort:
            async def search(self, kind, query, *, limit):
                raise KeywordTimeoutError("index is unavailable")

        output = await run_entity_lane(
            VectorPort({VectorKind.NODE}),
            FailingKeywordPort(),
            EntityPort(),
            [0.1, 0.2, 0.3],
            query="Acme",
            mentions=(),
            settings=SETTINGS,
        )

        assert output.record.status is LaneStatus.FAILED
        assert output.record.failed == ("keyword", "node")
