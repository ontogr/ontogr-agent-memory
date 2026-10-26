"""Behavior of the capped breadth-first graph lane."""

from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models import FactStatus
from ontogr_mem.retrieval import LaneStatus, RetrievalSettings
from ontogr_mem.retrieval.lanes.graph import run_graph_lane
from ontogr_mem.temporal.errors import TemporalError
from ontogr_mem.temporal.query import FactView, TemporalQuery


NOW = datetime(2026, 10, 23, tzinfo=UTC)
QUERY = TemporalQuery(now=NOW, include_history=True)


def make_view(
    fact_id: str, subject: str, object_: str, statement: str, *, disputed: bool = False
) -> FactView:
    """Build a fact in the literal graph."""
    return FactView(
        fact_id=fact_id,
        version=1,
        slot_key=f"slot-{fact_id}",
        subject_id=subject,
        object_id=object_,
        relation_type_id=1,
        valid_from=NOW,
        valid_to=None,
        valid_from_precision=None,
        valid_to_precision=None,
        recorded_at=NOW,
        expired_at=None,
        status=FactStatus.ACTIVE,
        evidence=(),
        statement=statement,
        disputed=disputed,
    )


class GraphReader:
    """Read the facts touching a frontier, with an optional second-hop failure."""

    def __init__(self, facts: list[FactView], *, fail_hop: int | None = None) -> None:
        """Store the graph facts and optional hop that should fail."""
        self.facts = facts
        self.fail_hop = fail_hop
        self.frontiers: list[tuple[str, ...]] = []

    async def touching(self, entity_ids, query):
        """Return facts that touch the requested entities."""
        self.frontiers.append(tuple(entity_ids))
        if self.fail_hop == len(self.frontiers):
            raise TemporalError("reader unavailable")
        frontier = set(entity_ids)
        return [
            fact
            for fact in self.facts
            if fact.subject_id in frontier or fact.object_id in frontier
        ]


class TestGraphLane:
    """Breadth-first walk order, caps and failures."""

    async def test_walks_to_the_requested_depth_and_keeps_first_hop(self):
        """Each hop adds only the facts first reached at that distance."""
        reader = GraphReader(
            [
                make_view("f1", "a", "b", "Ana at Acme"),
                make_view("f2", "b", "c", "Acme owns Globex"),
                make_view("f3", "c", "d", "Globex near Initech"),
            ]
        )

        shallow = await run_graph_lane(
            reader,
            seeds=("a",),
            query=QUERY,
            graph_depth=1,
            settings=RetrievalSettings(),
        )
        deep = await run_graph_lane(
            reader,
            seeds=("a",),
            query=QUERY,
            graph_depth=3,
            settings=RetrievalSettings(),
        )

        assert [view.fact_id for view in shallow.facts] == ["f1"]
        assert [view.fact_id for view in deep.facts] == ["f1", "f2", "f3"]
        assert deep.record.status is LaneStatus.OK

    async def test_ranks_by_hop_then_query_overlap_then_fact_key(self):
        """The lane sorts by hop, overlap and then the stable fact key."""
        reader = GraphReader(
            [
                make_view("z", "a", "b", "unrelated claim"),
                make_view("c", "a", "c", "Ana works Acme"),
                make_view("b", "a", "d", "Ana works Globex"),
            ]
        )

        output = await run_graph_lane(
            reader,
            seeds=("a",),
            query=TemporalQuery(now=NOW, include_history=True),
            graph_depth=1,
            settings=RetrievalSettings(),
            query_text="Ana works Acme",
        )

        assert [view.fact_id for view in output.facts] == ["c", "b", "z"]

    async def test_caps_next_frontier_in_entity_id_order(self):
        """The next frontier is sorted and truncated at the configured cap."""
        facts = [
            make_view(f"f{i:03}", "root", f"node-{i:03}", "neighbor")
            for i in range(300)
        ]
        reader = GraphReader(facts)

        output = await run_graph_lane(
            reader,
            seeds=("root",),
            query=QUERY,
            graph_depth=2,
            settings=RetrievalSettings(frontier_cap=200),
        )

        assert len(output.facts) == 300
        assert len(reader.frontiers[1]) == 200
        assert reader.frontiers[1][0] == "node-000"
        assert reader.frontiers[1][-1] == "node-199"

    async def test_a_later_hop_failure_discards_earlier_facts(self):
        """A failed later read invalidates facts gathered at earlier hops."""
        reader = GraphReader([make_view("f1", "a", "b", "claim")], fail_hop=2)

        output = await run_graph_lane(
            reader,
            seeds=("a",),
            query=QUERY,
            graph_depth=2,
            settings=RetrievalSettings(),
        )

        assert output.facts == ()
        assert output.record.status is LaneStatus.FAILED
        assert output.record.failed == ("touching",)

    async def test_skips_without_seeds_and_propagates_value_error(self):
        """No seeds skip the lane, while programming errors still propagate."""
        skipped = await run_graph_lane(
            GraphReader([]),
            seeds=(),
            query=QUERY,
            graph_depth=1,
            settings=RetrievalSettings(),
        )
        assert skipped.record.status is LaneStatus.SKIPPED

        class InvalidReader(GraphReader):
            async def touching(self, entity_ids, query):
                raise ValueError("invalid entity ids")

        with pytest.raises(ValueError, match="invalid entity ids"):
            await run_graph_lane(
                InvalidReader([]),
                seeds=("a",),
                query=QUERY,
                graph_depth=1,
                settings=RetrievalSettings(),
            )

    async def test_excludes_disputed_views(self):
        """The exclude mode removes disputed facts from graph results."""
        reader = GraphReader(
            [
                make_view("f1", "a", "b", "settled"),
                make_view("f2", "a", "c", "disputed", disputed=True),
            ]
        )

        output = await run_graph_lane(
            reader,
            seeds=("a",),
            query=QUERY,
            graph_depth=1,
            settings=RetrievalSettings(),
            disputed="exclude",
        )

        assert [view.fact_id for view in output.facts] == ["f1"]
