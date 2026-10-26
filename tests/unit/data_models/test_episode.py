"""The text of an episode."""

from datetime import UTC, datetime
from typing import Any

from ontogr_mem.data_models import Actor, Episode, Kind, Role, Source
from ontogr_mem.graphstore.episode_rows import episode_params


OCCURRED_AT = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)

STRUCTURED_CONTENT: dict[str, Any] = {
    "zeta": "last",
    "alpha": "Straße",
    "nested": {"b": 1, "a": "café"},
}

CANONICAL_CONTENT = '{"alpha":"Straße","nested":{"a":"café","b":1},"zeta":"last"}'


def make_episode(content: Any) -> Episode:
    """Build a sealed episode that carries `content`."""
    return Episode(
        id="ep1",
        kind=Kind.MESSAGE if isinstance(content, str) else Kind.EVENT,
        occurred_at=OCCURRED_AT,
        ingested_at=OCCURRED_AT,
        source=Source("chat", "ref-1", "chat", "session s1"),
        actor=Actor("u1", Role.USER, "Ana"),
        content=content,
        content_hash="hash-1",
    )


class TestEpisodeText:
    """The text of an episode."""

    def test_returns_string_content_unchanged(self):
        """Text content is its own text."""
        episode = make_episode("hello wörld")

        assert episode.text == "hello wörld"

    def test_returns_canonical_json_for_structured_content(self):
        """Structured content becomes JSON with sorted keys and no escaping."""
        episode = make_episode(STRUCTURED_CONTENT)

        assert episode.text == CANONICAL_CONTENT

    def test_matches_the_text_the_graph_stores(self):
        """The graph stores the same text that the episode reports."""
        episode = make_episode(STRUCTURED_CONTENT)

        assert episode_params(episode)["content"] == episode.text
