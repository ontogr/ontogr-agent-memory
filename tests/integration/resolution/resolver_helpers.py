"""Builders for resolver behavior tests."""

from ontogr_mem.data_models import ExtractedEntity, Method, SystemClock
from ontogr_mem.data_models.entities import Mention
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.policy import ResolutionPolicy
from ontogr_mem.resolution.resolver import EntityResolver
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.vectorstore import EmbeddingGateway, VectorStore
from ontogr_mem.vectorstore.memory import InMemoryBackend
from tests.support.embedder import MockEmbedder


def ex(
    local_id: str,
    name: str,
    type_label: str = "Company",
    episode_id: str = "ep1",
    description: str = "",
    **attributes: object,
) -> ExtractedEntity:
    """Build an extracted entity."""
    return ExtractedEntity(
        local_id=local_id,
        name=name,
        type_id=0,
        type_label=type_label,
        confidence=0.9,
        method=Method.RECORD,
        episode_id=episode_id,
        description=description,
        attributes=dict(attributes),
    )


async def make(
    store: ResolverStore,
    *,
    llm: object | None = None,
    inline: bool = False,
    **settings: object,
) -> EntityResolver:
    """Build a resolver on the given store with a deterministic embedder."""
    vectors = VectorStore(InMemoryBackend(), EmbeddingGateway(MockEmbedder()))
    await vectors.ensure_ready()
    return EntityResolver(
        store,
        vectors,
        ResolutionPolicy(),
        ResolverSettings(inline_verification=inline, **settings),  # type: ignore[arg-type]
        SystemClock(),
        llm=llm,  # type: ignore[arg-type]
    )


async def latest_mention(store: ResolverStore, episode_id: str) -> Mention:
    """Return the last mention written for an episode."""
    found = [
        mention
        for entity in await store.active_entities()
        for mention in await store.mentions_of(entity.entity_id)
        if mention.episode_id == episode_id
    ]
    return max(found, key=lambda mention: mention.mention_id)


async def count_active(store: ResolverStore) -> int:
    """Return the number of active entities."""
    return len(await store.active_entities())
