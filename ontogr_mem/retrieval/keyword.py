"""Tenant-bound keyword search over aliases, observations and episodes."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class KeywordKind(StrEnum):
    """Documents searched by the keyword port."""

    ENTITY = "entity"
    OBSERVATION = "observation"
    EPISODE = "episode"


@dataclass(frozen=True, slots=True)
class KeywordHit:
    """Document id, backend-local score and zero-based rank."""

    id: str
    score: float
    rank: int


class KeywordSearch(Protocol):
    """Search documents belonging to one tenant."""

    async def search(
        self, kind: KeywordKind, query: str, *, limit: int
    ) -> list[KeywordHit]:
        """Return at most `limit` distinct hits, ordered by descending score.

        Empty token lists return no hits. Scores are not comparable across
        backends. Entity hits name active survivors, not alias nodes.
        """
        ...
