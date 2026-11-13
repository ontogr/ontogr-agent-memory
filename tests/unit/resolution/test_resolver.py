"""Resolver behavior when one resolve call skips inline verification.

The store here is a dict-backed stand-in for the resolver store port. It
implements only the members that these tests reach, so it is not checked
against the shared store contract.
"""

import asyncio
from collections.abc import Sequence

from ontogr_mem.data_models import ExtractedEntity, Method
from ontogr_mem.data_models.entities import (
    Alias,
    DecisionMethod,
    Entity,
    EntityStatus,
    Mention,
    Outcome,
)
from ontogr_mem.llm import RoleModels
from ontogr_mem.resolution.blocking import BlockingRow
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.keys import stripped_key
from ontogr_mem.resolution.policy import ResolutionPolicy
from ontogr_mem.resolution.resolver import EntityResolver
from ontogr_mem.vectorstore import EmbeddingGateway, VectorStore
from ontogr_mem.vectorstore.memory import InMemoryBackend
from tests.support.clock import FixedClock
from tests.support.embedder import MockEmbedder
from tests.support.scripted_model import scripted_model, scripted_role_models
from tests.unit.resolution.helpers import entity, oracle_models


class _MemoryResolverStore:
    """Dict-backed resolver store seeded with entities."""

    def __init__(self, entities: Sequence[Entity] = ()) -> None:
        """Hold the seeded entities and start with no aliases or mentions."""
        self.resolve_lock = asyncio.Lock()
        self._entities = {entity.entity_id: entity for entity in entities}
        self._aliases: dict[str, Alias] = {}
        self._mentions: dict[str, Mention] = {}
        self._resolved_episodes: set[str] = set()

    async def episode_resolution_state(self, episode_id: str) -> bool:
        """Return whether mentions were already written for the episode."""
        return episode_id in self._resolved_episodes

    async def entities_by_strict_key(self, strict_key: str) -> list[Entity]:
        """Return the entities whose key or alias key matches, ordered by id."""
        ids = {
            entity.entity_id
            for entity in self._entities.values()
            if entity.strict_key == strict_key
        }
        ids.update(
            alias.entity_id
            for alias in self._aliases.values()
            if alias.strict_key == strict_key
        )
        return [self._entities[entity_id] for entity_id in sorted(ids)]

    async def get_entities(self, ids: Sequence[str]) -> list[Entity]:
        """Return the stored entities among ``ids``, ordered by id."""
        return [
            self._entities[entity_id]
            for entity_id in sorted(set(ids))
            if entity_id in self._entities
        ]

    async def blocking_rows(
        self, *, include_archived: bool = False
    ) -> list[BlockingRow]:
        """Return one blocking row per active entity, or archived too."""
        statuses = {EntityStatus.ACTIVE}
        if include_archived:
            statuses.add(EntityStatus.ARCHIVED)
        return [
            BlockingRow(
                entity_id=entity.entity_id,
                loose_key=entity.loose_key,
                stripped_key=stripped_key(entity.canonical_name),
                alias_keys=tuple(
                    alias.loose_key
                    for alias in self._aliases.values()
                    if alias.entity_id == entity.entity_id
                ),
            )
            for entity in sorted(self._entities.values(), key=lambda row: row.entity_id)
            if entity.status in statuses
        ]

    async def write_entity_batch(
        self, entities: Sequence[Entity], aliases: Sequence[Alias]
    ) -> list[Entity]:
        """Store the entities and aliases and return the entities in order."""
        for row in entities:
            self._entities[row.entity_id] = row
        for alias in aliases:
            self._aliases[alias.alias_id] = alias
        return list(entities)

    async def write_mentions(
        self, episode_id: str, mentions: Sequence[Mention]
    ) -> None:
        """Store the mentions and mark the episode as resolved."""
        for mention in mentions:
            self._mentions[mention.mention_id] = mention
        self._resolved_episodes.add(episode_id)

    async def aliases_of(self, entity_id: str) -> list[Alias]:
        """Return the aliases of one entity, ordered by id."""
        return sorted(
            (alias for alias in self._aliases.values() if alias.entity_id == entity_id),
            key=lambda alias: alias.alias_id,
        )


def _seeded_store() -> _MemoryResolverStore:
    """Return a store with two entities whose names share one strict key."""
    return _MemoryResolverStore([entity("c1", "Acme"), entity("c2", "ACME")])


def _extracted(local_id: str, name: str, episode_id: str = "ep1") -> ExtractedEntity:
    """Build an extracted company mention."""
    return ExtractedEntity(
        local_id=local_id,
        name=name,
        type_id=0,
        confidence=0.9,
        method=Method.RECORD,
        episode_id=episode_id,
        type_label="Company",
        description="",
        attributes={},
    )


async def _resolver(store: _MemoryResolverStore, models: RoleModels) -> EntityResolver:
    """Build a resolver with inline verification on, which is off by default."""
    vectors = VectorStore(InMemoryBackend(), EmbeddingGateway(MockEmbedder()))
    await vectors.ensure_ready()
    return EntityResolver(
        store,
        vectors,
        ResolutionPolicy(),
        ResolverSettings(inline_verification=True),
        FixedClock(),
        models=models,
    )


class TestEntityResolver:
    """Resolve batches with and without the resolution model."""

    async def test_allow_model_false_still_creates_and_reuses_entities(self):
        """Deterministic matching still reuses entities when the model is skipped."""
        model, scripted = scripted_model()
        resolver = await _resolver(
            _seeded_store(), scripted_role_models(resolution=model)
        )

        first = await resolver.resolve(
            [_extracted("1", "Acme"), _extracted("2", "Initech")],
            allow_model=False,
        )
        second = await resolver.resolve(
            [_extracted("1", "Initech", episode_id="ep2")],
            allow_model=False,
        )

        assert len(first.entities) == 2
        assert len(first.mentions) == 2
        mentions = {mention.surface_form: mention for mention in first.mentions}
        assert mentions["Acme"].decision_outcome is Outcome.CREATED_PROVISIONAL
        [reused] = second.mentions
        assert reused.entity_id == mentions["Initech"].entity_id
        assert reused.decision_method is DecisionMethod.KEY
        assert scripted.calls == []

    async def test_default_still_verifies_when_a_verifier_is_configured(self):
        """The verifier decides an ambiguous name when allow_model is omitted."""
        models, scripted = oracle_models(lambda name_a, name_b: name_b == "Acme")
        resolver = await _resolver(_seeded_store(), models)

        result = await resolver.resolve([_extracted("1", "Acme")])

        assert len(scripted.calls) >= 1
        [mention] = result.mentions
        assert mention.decision_method is DecisionMethod.LLM
        assert mention.entity_id == "c1"
