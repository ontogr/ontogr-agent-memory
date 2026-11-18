"""Entity card provenance through the public retrieval interface."""

from datetime import UTC, datetime

from ontogr_mem.data_models import Entity, EntityStatus
from ontogr_mem.retrieval import (
    KeywordHit,
    KeywordKind,
    RetrievalRequest,
    RetrievalSettings,
    Retriever,
)
from ontogr_mem.vectorstore.embedder import EmbedOutcome


NOW = datetime(2026, 10, 23, tzinfo=UTC)


def _entity(entity_id: str, name: str, *, merged_into: str | None = None) -> Entity:
    """Build an entity that may point at a survivor."""
    return Entity(
        entity_id=entity_id,
        ontology_type_id=1,
        type_label="Person",
        canonical_name=name,
        strict_key=name.casefold(),
        loose_key=name.casefold(),
        status=EntityStatus.ACTIVE,
        merged_into=merged_into,
        created_at=NOW,
        updated_at=NOW,
    )


class _Gateway:
    """Return one fixed query embedding."""

    async def embed(self, texts, *, purpose):
        """Embed every text to the same vector."""
        return [EmbedOutcome(vector=(1.0, 0.0, 0.0)) for _ in texts]


class _Vectors:
    """Empty vector searches with a ready embedding gateway."""

    def __init__(self) -> None:
        """Bind the fixed gateway."""
        self.gateway = _Gateway()

    async def search(self, kind, embedding, *, k, filters=None, min_score=None):
        """Return no hits."""
        return []


class _Keywords:
    """Find one entity by keyword but no facts or episodes."""

    async def search(self, kind, query, *, limit):
        """Return the second entity for the entity-lane search."""
        if kind is KeywordKind.ENTITY:
            return [KeywordHit("B", 1.0, 0)]
        return []


class _Entities:
    """Two ids that both resolve to one survivor card."""

    def __init__(self) -> None:
        """Store the mentioned id, the keyword id and their survivor."""
        self.records = {
            "A": _entity("A", "Ada", merged_into="S"),
            "B": _entity("B", "Grace", merged_into="S"),
            "S": _entity("S", "Survivor"),
        }

    async def entities_by_strict_key(self, key):
        """Return the mentioned entity for its strict key."""
        return [self.records["A"]] if key == "ada" else []

    async def entities_by_loose_key(self, key):
        """Return no blocking-key candidates."""
        return []

    async def get_entities(self, ids, *, follow_merged=False):
        """Return the stored records for the known ids."""
        return [self.records[item] for item in ids if item in self.records]

    async def aliases_of(self, entity_id):
        """Return no aliases."""
        return []


class _Facts:
    """Fact reader with no visible facts."""

    async def by_ids(self, ids, query):
        """Return no hydrated facts."""
        return []

    async def by_observations(self, ids, query):
        """Return no evidence facts."""
        return []

    async def touching(self, ids, query):
        """Return no graph-walk facts."""
        return []

    async def observation_texts(self, ids):
        """Return no observation texts."""
        return {}


class _Episodes:
    """Episode store with no passages."""

    async def get_episodes(self, ids):
        """Return no episodes."""
        return []


class TestRetriever:
    """Verify retrieval results built from small in-memory ports."""

    async def test_entity_card_aggregates_survivor_lanes_before_cap(self):
        """A survivor card keeps lanes from every id that resolves to it."""
        retriever = Retriever(
            vectors=_Vectors(),
            keywords=_Keywords(),
            facts=_Facts(),
            entities=_Entities(),
            episodes=_Episodes(),
            settings=RetrievalSettings(max_cards=1),
        )

        result = await retriever.retrieve(
            RetrievalRequest(query="Ada and Grace", now=NOW, mentions=("Ada",))
        )

        assert len(result.entities) == 1
        assert result.entities[0].entity_id == "S"
        assert result.entities[0].lanes == ("mention", "keyword")
