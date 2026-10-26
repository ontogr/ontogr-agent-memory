"""Owner-reviewed synthetic queries for the offline retrieval ablation.

The expected names refer to the retrieval scenario, not generated database ids.
These fixture labels do not establish real-world retrieval quality.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict


class RequestOptions(TypedDict, total=False):
    """Optional filters passed with one retrieval query."""

    valid_at: datetime
    overlaps: tuple[datetime, datetime]
    include_history: bool
    graph_depth: int
    mentions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LabelledQuery:
    """One owner-review query and its expected scenario names.

    Attributes:
        id: Stable query id. Its prefix names the D5 category.
        text: Query text sent to retrieval.
        expected_facts: Scenario fact names expected in the result.
        expected_episodes: Scenario episode names expected in the result.
        overview: Whether the query asks for a broad entity summary.
        request_options: Temporal, mention, history, or graph options.
        why: One-line reason for including the query.
    """

    id: str
    text: str
    expected_facts: tuple[str, ...]
    expected_episodes: tuple[str, ...]
    overview: bool
    request_options: RequestOptions
    why: str


_YEAR_2023 = datetime(2023, 6, 1, tzinfo=UTC)
_YEAR_2025 = datetime(2025, 6, 1, tzinfo=UTC)
_OVERLAP_START = datetime(2023, 1, 1, tzinfo=UTC)
_OVERLAP_END = datetime(2025, 1, 1, tzinfo=UTC)


LABELLED: tuple[LabelledQuery, ...] = (
    LabelledQuery(
        "exact-term-01",
        "Ana works at Acme",
        ("F1",),
        ("E1",),
        False,
        {},
        "Exact terms target the stable employment fact and its source episode.",
    ),
    LabelledQuery(
        "exact-term-02",
        "Ana lived in Berlin",
        ("F2",),
        ("E2",),
        False,
        {"valid_at": _YEAR_2023},
        (
            "Exact terms with a 2023 valid-time filter should select the "
            "closed Berlin fact."
        ),
    ),
    LabelledQuery(
        "exact-term-03",
        "cobaltsignal",
        (),
        ("E7",),
        False,
        {},
        "The rare exact term should find a standalone episode with no extracted fact.",
    ),
    LabelledQuery(
        "paraphrase-01",
        "Which organization has the person on its payroll?",
        ("F1",),
        ("E1",),
        False,
        {"mentions": ("Ana",)},
        "The mention anchors Ana while the employment wording has no exact fact terms.",
    ),
    LabelledQuery(
        "paraphrase-02",
        "Which business is linked with it?",
        ("F9",),
        ("E1",),
        False,
        {"mentions": ("Acme",)},
        (
            "The mention anchors Acme and linked is a paraphrase of its "
            "connection to Globex."
        ),
    ),
    LabelledQuery(
        "paraphrase-03",
        "Which organization is the person affiliated with?",
        ("F7",),
        ("E10",),
        False,
        {"mentions": ("Carol",)},
        (
            "The mention anchors Carol; affiliated is broad, and the intended "
            "chosen employer is F7."
        ),
    ),
    LabelledQuery(
        "name-variant-01",
        "Where does Ana Marie work?",
        ("F1",),
        ("E1",),
        False,
        {"mentions": ("Ana Marie",)},
        "Ana Marie is the merged alias for Ana, so the alias should reach F1.",
    ),
    LabelledQuery(
        "name-variant-02",
        "Which company employs Ana Marie?",
        ("F1",),
        ("E1",),
        False,
        {"mentions": ("Ana Marie",)},
        (
            "A second wording checks the same merged name alias without using "
            "Ana's canonical name."
        ),
    ),
    LabelledQuery(
        "multi-hop-01",
        "What is the final organization in Ana's chain through Acme and Globex?",
        ("F1", "F9", "F10"),
        ("E1",),
        False,
        {"mentions": ("Ana",), "graph_depth": 3},
        "The depth-three graph path from Ana through Acme and Globex reaches F10.",
    ),
    LabelledQuery(
        "multi-hop-02",
        "What organizations lie along the route from the named organization "
        "to its endpoint?",
        ("F9", "F10"),
        ("E1",),
        False,
        {"mentions": ("Acme",), "graph_depth": 2},
        (
            "The mention supplies the start node, and two graph hops cover "
            "the Acme to Initech chain."
        ),
    ),
    LabelledQuery(
        "time-01",
        "Where did Ana live in 2023?",
        ("F2",),
        ("E2",),
        False,
        {"valid_at": _YEAR_2023},
        "The 2023 valid-time filter selects Berlin and excludes its later successor.",
    ),
    LabelledQuery(
        "time-02",
        "Where did Ana live in 2025?",
        ("F3",),
        ("E3",),
        False,
        {"valid_at": _YEAR_2025},
        "The 2025 valid-time filter selects Munich after the 2024 change.",
    ),
    LabelledQuery(
        "time-03",
        "Which residence facts overlap 2023 to 2025?",
        ("F2", "F3"),
        ("E2", "E3"),
        False,
        {"overlaps": (_OVERLAP_START, _OVERLAP_END)},
        (
            "The overlap window crosses the 2024 boundary and should include "
            "both versions."
        ),
    ),
    LabelledQuery(
        "dispute-01",
        "What do the sources report about Bob's employer?",
        ("F4", "F5"),
        ("E4", "E10"),
        False,
        {"mentions": ("Bob",)},
        "Both disputed claims and their source episodes belong in the result.",
    ),
    LabelledQuery(
        "dispute-02",
        "Which two claims disagree about Bob's place of work?",
        ("F4", "F5"),
        ("E4", "E10"),
        False,
        {"mentions": ("Bob",)},
        "The query must preserve both sides and both source episodes.",
    ),
    LabelledQuery(
        "overridden-01",
        "Which employer did the decision keep for Carol?",
        ("F7",),
        ("E10",),
        False,
        {"mentions": ("Carol",)},
        "F7 is the chosen current claim and E10 supplies its source.",
    ),
    LabelledQuery(
        "overridden-02",
        "Was Carol's Alpha work claim replaced by Beta?",
        ("F6", "F7"),
        ("E4", "E10"),
        False,
        {"mentions": ("Carol",)},
        "F6 remains visible as overridden and F7 is chosen, with both sources.",
    ),
    LabelledQuery(
        "episode-only-01",
        "What standalone note mentions cobaltsignal?",
        (),
        ("E7",),
        False,
        {},
        "E7 is searchable episode content without an extracted fact.",
    ),
    LabelledQuery(
        "episode-only-02",
        "What does the structured note say about jsonneedle?",
        (),
        ("E8",),
        False,
        {},
        "The query targets the JSON episode text and should not invent a fact result.",
    ),
    LabelledQuery(
        "history-01",
        "Show the withdrawn relationship between Retained Person and Retained Org.",
        ("F8",),
        (),
        False,
        {"include_history": True},
        (
            "F8 is a withdrawn tombstone whose only source episode E5 was "
            "reset and removed."
        ),
    ),
    LabelledQuery(
        "history-02",
        "What claim did Orphan Person have before reset?",
        (),
        (),
        False,
        {"include_history": True},
        (
            "Orphan F11 and its entities were deleted, so history must not "
            "restore the orphan."
        ),
    ),
    LabelledQuery(
        "overview-01",
        "What do we know about Acme?",
        ("F1", "F9"),
        ("E1",),
        True,
        {"mentions": ("Acme",), "graph_depth": 1},
        (
            "This broad query includes Acme's employer and its direct "
            "connection, not a summary record."
        ),
    ),
    LabelledQuery(
        "overview-02",
        "What do we know about Ana?",
        ("F1", "F3", "F9", "F10"),
        ("E1", "E3"),
        True,
        {"mentions": ("Ana",), "graph_depth": 3},
        (
            "The current direct facts and Acme's two-hop chain are included; "
            "whether indirect facts belong is open to review."
        ),
    ),
    LabelledQuery(
        "overview-03",
        "What do we know about Globex?",
        ("F4", "F5", "F9", "F10"),
        ("E1", "E4", "E10"),
        True,
        {"mentions": ("Globex",), "graph_depth": 2},
        "The broad set includes Globex's dispute, connections, and their sources.",
    ),
)
