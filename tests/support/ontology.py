"""Catalog builders for ontology tests."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ontogr_mem.graphstore.episode_store import GraphEpisodeStore
from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.types import GraphHandle
from ontogr_mem.ingestion import EpisodeStore
from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.models import (
    AddEntity,
    AddRelation,
    Attribute,
    ChangeSet,
    Ontology,
    PatternSpec,
    ValueType,
)
from ontogr_mem.ontology.ontology_store import OntologyStore
from tests.support.episodes import make_episode


def cs(cat: Ontology, *ops: object, **kw: object) -> ChangeSet:
    """Cs."""
    return ChangeSet(base_version=cat.version, ops=list(ops), **kw)  # type: ignore[arg-type]


def base() -> Ontology:
    """Build a small catalog.

    Unknown(0), Person(1), Employee(2, child of Person), Organization(3),
    works_at(Person -> Organization).
    """
    g = Ontology.genesis()
    return apply_changeset(
        g,
        cs(
            g,
            AddEntity(
                label="Person",
                description="A human being.",
                properties=[
                    Attribute(name="age", value_type=ValueType.INTEGER),
                    Attribute(name="nickname", max_length=5),
                    Attribute(name="email", required=True),
                ],
            ),
            AddEntity(label="Employee", parent="Person"),
            AddEntity(label="Organization", aliases=["Org"]),
            AddRelation(
                label="works_at",
                patterns=[PatternSpec(src="Person", tgt="Organization")],
            ),
        ),
    )


@dataclass
class OntologyHarness:
    """An ontology store with a hook that ingests an episode into the tenant."""

    store: OntologyStore
    add_episode: Callable[[str], Awaitable[None]]


async def commit_entity(store: OntologyStore, label: str) -> Ontology:
    """Add an entity type on top of the head and return the new head."""
    head = await store.get()
    change_set = cs(head, AddEntity(label=label))
    new = apply_changeset(head, change_set)
    await store.commit(new, change_set)
    return new


async def ingest_episode(episodes: EpisodeStore, episode_id: str) -> None:
    """Write an episode and record its hash, as a finished ingest does."""
    await episodes.write_episodes([make_episode(episode_id)])
    await episodes.record_content_hash(episode_id, f"hash-{episode_id}")


def graph_ontology_harness(
    store: GraphStore, handle: GraphHandle, *, tenant_keyed: bool
) -> OntologyHarness:
    """Build a harness for a graph backend from an opened tenant."""
    episodes = GraphEpisodeStore(store.runner(handle), tenant_keyed=tenant_keyed)

    async def add_episode(episode_id: str) -> None:
        await ingest_episode(episodes, episode_id)

    return OntologyHarness(
        GraphOntologyStore(store, handle, tenant_keyed=tenant_keyed), add_episode
    )
