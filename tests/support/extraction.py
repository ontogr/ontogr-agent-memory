"""Builders and doubles for entity extraction tests."""

from collections.abc import Sequence
from typing import Any

from ontogr_mem.data_models import (
    EntityType,
    Episode,
    TypeCatalog,
)
from ontogr_mem.entity_extraction.extractors import RawSpan
from ontogr_mem.llm import LLMGate
from tests.support.episodes import make_episode
from tests.support.llm import ScriptedLLM


def catalog() -> TypeCatalog:
    """Return a catalog of Person (0), Database (1) and Company (2)."""
    return TypeCatalog(
        [EntityType(0, "Person"), EntityType(1, "Database"), EntityType(2, "Company")],
        [],
    )


def text_episode(text: str, episode_id: str = "ep1", **kwargs: Any) -> Episode:
    """Build a message episode with the given text."""
    return make_episode(episode_id, content=text, **kwargs)


def gate(llm: ScriptedLLM, max_input_tokens: int = 10_000) -> LLMGate:
    """Build a gate around a scripted client."""
    return LLMGate(llm, max_input_tokens=max_input_tokens)


def entities_reply(*entities: dict[str, Any]) -> dict[str, Any]:
    """Build the JSON object a model returns for the entity pass."""
    return {"entities": list(entities)}


class FakeGliner:
    """Mimics `GLiNER.predict_entities` with fixed hits."""

    def __init__(self, hits: Sequence[RawSpan]) -> None:
        """Keep the hits to return."""
        self.hits = list(hits)

    def predict_entities(
        self, text: str, labels: list[str], threshold: float
    ) -> list[dict[str, object]]:
        """Return the hits scoring at least `threshold`."""
        return [
            {
                "start": h.start,
                "end": h.end,
                "text": h.text,
                "label": h.label,
                "score": h.score,
            }
            for h in self.hits
            if h.score >= threshold
        ]


class FakePrevious:
    """A `PreviousEpisodes` over a fixed list of episodes."""

    def __init__(self, episodes: Sequence[Episode]) -> None:
        """Keep the episodes, oldest first."""
        self.episodes = list(episodes)
        self.limits: list[int] = []

    async def before(self, episode: Episode, limit: int) -> Sequence[Episode]:
        """Return up to `limit` episodes that occurred before `episode`."""
        self.limits.append(limit)
        earlier = [e for e in self.episodes if e.occurred_at < episode.occurred_at]
        return earlier[-limit:] if limit else []
