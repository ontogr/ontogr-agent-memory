"""Check the proposed retrieval labels against the scenario names and D5 rules."""

from collections import Counter
from datetime import datetime

from tests.eval.retrieval_labelled import LABELLED
from tests.support.retrieval_world import _scenario_specs


def test_labels_are_unique_well_formed_and_match_scenario_names() -> None:
    """The review set covers D5 and uses only names present in the world."""
    assert len(LABELLED) == 24
    assert len({item.id for item in LABELLED}) == len(LABELLED)

    category_counts = Counter(item.id.rsplit("-", 1)[0] for item in LABELLED)
    assert set(category_counts) == {
        "exact-term",
        "paraphrase",
        "name-variant",
        "multi-hop",
        "time",
        "dispute",
        "overridden",
        "episode-only",
        "history",
        "overview",
    }
    assert all(count >= 2 for count in category_counts.values())
    assert sum(item.overview for item in LABELLED) >= 2

    scenario = _scenario_specs()
    fact_names = {fact.key for episode in scenario for fact in episode.facts}
    episode_names = {episode.episode_id.upper() for episode in scenario}
    for item in LABELLED:
        assert item.why.strip(), item.id
        assert "\n" not in item.why, item.id
        assert set(item.expected_facts) <= fact_names, item.id
        assert set(item.expected_episodes) <= episode_names, item.id
        assert "F11" not in item.expected_facts, item.id
        assert set(item.request_options) <= {
            "valid_at",
            "overlaps",
            "include_history",
            "graph_depth",
            "mentions",
        }
        if "valid_at" in item.request_options:
            value = item.request_options["valid_at"]
            assert isinstance(value, datetime) and value.utcoffset() is not None
        if "overlaps" in item.request_options:
            start, end = item.request_options["overlaps"]
            assert start.utcoffset() is not None and end.utcoffset() is not None
            assert start < end
        if "include_history" in item.request_options:
            assert isinstance(item.request_options["include_history"], bool)
        if "graph_depth" in item.request_options:
            depth = item.request_options["graph_depth"]
            assert depth in (1, 2, 3)
            if item.id.startswith("multi-hop-"):
                assert depth in (2, 3)
        if "mentions" in item.request_options:
            assert item.request_options["mentions"]

    for item in LABELLED:
        if "F8" in item.expected_facts:
            assert item.id.startswith("history-")
            assert item.request_options.get("include_history") is True
