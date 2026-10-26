"""Pure grouping helpers: label propagation over a graph and bucket planning.

These functions take plain data and return plain data, so the tests call them
directly with small hand-made graphs and check literal results.
"""

import pytest

from ontogr_mem.consolidation.grouping.buckets import (
    BucketPlan,
    bucket_id,
    plan_buckets,
)
from ontogr_mem.consolidation.grouping.propagation import label_propagation


Edge = tuple[str, str, int]


def _graph(*edges: Edge) -> dict[str, dict[str, int]]:
    neighbours: dict[str, dict[str, int]] = {}
    for a, b, weight in edges:
        neighbours.setdefault(a, {})[b] = weight
        neighbours.setdefault(b, {})[a] = weight
    return neighbours


TWO_CLIQUES = (
    ("a", "b", 1),
    ("b", "c", 1),
    ("a", "c", 1),
    ("x", "y", 1),
    ("y", "z", 1),
    ("x", "z", 1),
    ("c", "x", 1),
)


class TestLabelPropagation:
    """Community detection over undirected weighted graphs."""

    def test_two_cliques_joined_by_one_edge_are_two_communities(self) -> None:
        """A single bridge edge does not merge two dense groups."""
        assert label_propagation(_graph(*TWO_CLIQUES)) == [
            ["a", "b", "c"],
            ["x", "y", "z"],
        ]

    @pytest.mark.parametrize("reverse_insertion", [False, True])
    def test_result_is_the_same_when_edges_are_inserted_in_another_order(
        self, reverse_insertion: bool
    ) -> None:
        """Edge insertion order does not change the groups."""
        edges = list(reversed(TWO_CLIQUES)) if reverse_insertion else TWO_CLIQUES

        assert label_propagation(_graph(*edges)) == [
            ["a", "b", "c"],
            ["x", "y", "z"],
        ]

    def test_node_insertion_order_does_not_change_the_groups(self) -> None:
        """Node insertion order does not change the groups."""
        forward = {"a": {"b": 1}, "b": {"a": 1}, "c": {"d": 1}, "d": {"c": 1}}
        backward = {"d": {"c": 1}, "c": {"d": 1}, "b": {"a": 1}, "a": {"b": 1}}

        assert label_propagation(forward) == [["a", "b"], ["c", "d"]]
        assert label_propagation(backward) == [["a", "b"], ["c", "d"]]

    def test_path_collapses_into_one_community_when_ties_go_to_smallest_label(
        self,
    ) -> None:
        """Equal pulls on the middle node resolve to the smallest label."""
        assert label_propagation(_graph(("a", "b", 1), ("b", "c", 1))) == [
            ["a", "b", "c"]
        ]

    def test_one_sweep_already_collapses_a_path(self) -> None:
        """A path of five nodes joins into one group after a single sweep."""
        path = _graph(("a", "b", 1), ("b", "c", 1), ("c", "d", 1), ("d", "e", 1))

        assert label_propagation(path, max_iterations=1) == [["a", "b", "c", "d", "e"]]

    def test_zero_iterations_leave_every_node_alone(self) -> None:
        """With no sweeps every node keeps its own label and no group forms."""
        path = _graph(("a", "b", 1), ("b", "c", 1), ("c", "d", 1))

        assert label_propagation(path, max_iterations=0) == []

    def test_heavier_edges_win_over_a_lighter_bridge(self) -> None:
        """A heavy edge pulls its endpoints together over a light middle edge."""
        graph = _graph(("a", "b", 5), ("b", "c", 1), ("c", "d", 5))

        assert label_propagation(graph) == [["a", "b"], ["c", "d"]]

    def test_isolated_node_and_pair_below_min_size_are_not_communities(self) -> None:
        """Groups smaller than min_size are dropped."""
        graph = _graph(("a", "b", 1))
        graph["lonely"] = {}

        assert label_propagation(graph, min_size=3) == []
        assert label_propagation(graph, min_size=2) == [["a", "b"]]

    @pytest.mark.parametrize(
        "graph",
        [
            {},
            {"a": {"a": 4}},
            {"a": {}, "b": {}},
        ],
    )
    def test_empty_graphs_and_self_loops_form_no_community(
        self, graph: dict[str, dict[str, int]]
    ) -> None:
        """A node with no real neighbour never forms a group."""
        assert label_propagation(graph) == []

    def test_communities_are_ordered_by_their_first_member(self) -> None:
        """Groups are returned sorted by their first id."""
        graph = _graph(("y", "z", 1), ("c", "d", 1), ("a", "b", 1))

        assert label_propagation(graph) == [["a", "b"], ["c", "d"], ["y", "z"]]


class TestPlanBuckets:
    """Bucket hierarchy planning for community summaries."""

    @pytest.mark.parametrize("item_ids", [[], ["a"]])
    def test_fewer_than_two_items_need_no_hierarchy(self, item_ids: list[str]) -> None:
        """Zero or one item produces no bucket."""
        assert plan_buckets(item_ids, 20) == []

    def test_two_items_share_one_bucket_in_id_order(self) -> None:
        """Two items share one root bucket with children in id order."""
        assert plan_buckets(["b", "a"], 20) == [BucketPlan(1, 0, ("a", "b"))]

    @pytest.mark.parametrize("size", [19, 20, 21])
    def test_plan_is_the_same_for_any_input_order(self, size: int) -> None:
        """Reversing the input does not change the plan."""
        ids = [f"c{i:03d}" for i in range(size)]

        assert plan_buckets(list(reversed(ids)), 20) == plan_buckets(ids, 20)

    def test_nineteen_items_fit_one_bucket(self) -> None:
        """Below the maximum, one bucket holds everything."""
        plans = plan_buckets([f"c{i:03d}" for i in range(19)], 20)

        assert [(p.level, p.index, len(p.children)) for p in plans] == [(1, 0, 19)]

    def test_twenty_items_fit_exactly_one_bucket(self) -> None:
        """At the maximum, one bucket still holds everything."""
        plans = plan_buckets([f"c{i:03d}" for i in range(20)], 20)

        assert [(p.level, p.index, len(p.children)) for p in plans] == [(1, 0, 20)]

    def test_twenty_one_items_overflow_into_a_second_level(self) -> None:
        """One item over the maximum adds a level with a root above it."""
        plans = plan_buckets([f"c{i:03d}" for i in range(21)], 20)

        assert [(p.level, p.index, len(p.children)) for p in plans] == [
            (1, 0, 20),
            (1, 1, 1),
            (2, 0, 2),
        ]
        assert plans[-1].children == ("bucket-1-0", "bucket-1-1")

    def test_four_hundred_items_reach_a_single_root(self) -> None:
        """Large inputs build levels until one root bucket remains."""
        plans = plan_buckets([f"c{i:04d}" for i in range(400)], 20)

        assert plans[-1] == BucketPlan(2, 0, tuple(f"bucket-1-{i}" for i in range(20)))
        assert sum(1 for p in plans if p.level == 1) == 20
        assert sum(1 for p in plans if p.level == 2) == 1
        assert max(len(p.children) for p in plans) == 20

    def test_small_bucket_size_builds_levels_of_two(self) -> None:
        """A bucket size of two splits three items into a leftover bucket."""
        assert plan_buckets(["a", "b", "c"], 2) == [
            BucketPlan(1, 0, ("a", "b")),
            BucketPlan(1, 1, ("c",)),
            BucketPlan(2, 0, ("bucket-1-0", "bucket-1-1")),
        ]

    @pytest.mark.parametrize("max_bucket_size", [0, 1])
    def test_bucket_size_below_two_is_refused(self, max_bucket_size: int) -> None:
        """A bucket that cannot hold two children is rejected with ValueError."""
        with pytest.raises(ValueError):
            plan_buckets(["a", "b"], max_bucket_size)

    @pytest.mark.parametrize(
        ("level", "index", "expected"),
        [(1, 0, "bucket-1-0"), (2, 3, "bucket-2-3")],
    )
    def test_bucket_id_names_level_and_index(
        self, level: int, index: int, expected: str
    ) -> None:
        """Bucket ids have the form bucket-<level>-<index>."""
        assert bucket_id(level, index) == expected
