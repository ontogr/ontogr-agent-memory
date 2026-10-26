"""Tests for the observer that writes a vector for every new fact."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ontogr_mem.data_models import (
    Fact,
    FactStatus,
    Observation,
    ObservationKind,
    Precision,
    to_micros,
)
from ontogr_mem.vectorstore import (
    SearchFilters,
    TemporalFilter,
    VectorKind,
    VectorStore,
    VectorWriter,
)
from ontogr_mem.vectorstore.fact_vectors import FactVectorObserver
from ontogr_mem.vectorstore.memory import InMemoryBackend
from tests.support.embedder import MockEmbedder, RecordingEmbedder, gateway
from tests.support.vectors import ORTHOGONAL, A


T0 = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
FIRST = "Ana uses Postgres"
SECOND = "Ana works for Acme"


@dataclass(frozen=True)
class Commit:
    """A slot commit with only the facts the observer reads.

    Attributes:
        new_facts: The fact versions the commit writes.
    """

    new_facts: tuple[Fact, ...]


def observation(obs_id: str, statement: str = "") -> Observation:
    """Build an observation that states `statement`."""
    return Observation(
        obs_id=obs_id,
        slot_key="slot-1",
        subject_id="s1",
        object_id="o1",
        relation_type_id=7,
        kind=ObservationKind.ASSERTION,
        episode_id="ep1",
        occurred_at=T0,
        ingested_at=T0,
        valid_from=T0,
        valid_from_precision=Precision.DAY,
        statement=statement,
    )


def fact(
    fact_id: str,
    *,
    evidence: tuple[str, ...] = (),
    subject_id: str = "s1",
    object_id: str = "o1",
    relation_type_id: int = 7,
    status: FactStatus = FactStatus.ACTIVE,
    valid_from: datetime = T0,
    valid_to: datetime | None = None,
    expired_at: datetime | None = None,
    version: int = 1,
) -> Fact:
    """Build a fact version with the given evidence."""
    return Fact(
        fact_id=fact_id,
        version=version,
        slot_key="slot-1",
        subject_id=subject_id,
        object_id=object_id,
        relation_type_id=relation_type_id,
        valid_from=valid_from,
        valid_to=valid_to,
        valid_from_precision=Precision.DAY,
        valid_to_precision=None,
        recorded_at=T0,
        expired_at=expired_at,
        status=status,
        evidence=evidence,
        last_confirmed_at=T0,
    )


async def ready_store(embedder: MockEmbedder) -> VectorStore:
    """Return a store that is ready to write."""
    store = VectorStore(InMemoryBackend(), gateway(embedder))
    await store.ensure_ready()
    return store


class TestFactVectorObserver:
    """Behavior of the observer that writes fact vectors."""

    async def test_writes_one_vector_per_new_fact(self) -> None:
        """Each new fact of the commit gets a vector."""
        embedder = RecordingEmbedder(vectors={FIRST: A, SECOND: ORTHOGONAL})
        store = await ready_store(embedder)
        observer = FactVectorObserver(VectorWriter(store))
        commit = Commit(
            (
                fact("f1", evidence=("o1",)),
                fact("f2", evidence=("o2",)),
            )
        )
        observations = {
            "o1": observation("o1", FIRST),
            "o2": observation("o2", SECOND),
        }

        await observer.after_slot_commit(commit, observations)

        assert await store.exists(VectorKind.EDGE, ["f1", "f2"]) == {"f1", "f2"}

    async def test_each_vector_carries_the_subject_object_and_relation_type(
        self,
    ) -> None:
        """A search filters on the properties of the fact it found."""
        embedder = RecordingEmbedder(vectors={FIRST: A, SECOND: ORTHOGONAL})
        store = await ready_store(embedder)
        observer = FactVectorObserver(VectorWriter(store))
        commit = Commit(
            (
                fact(
                    "f1",
                    evidence=("o1",),
                    subject_id="s1",
                    object_id="o1",
                    relation_type_id=7,
                ),
                fact(
                    "f2",
                    evidence=("o2",),
                    subject_id="s2",
                    object_id="o2",
                    relation_type_id=9,
                ),
            )
        )
        observations = {
            "o1": observation("o1", FIRST),
            "o2": observation("o2", SECOND),
        }

        await observer.after_slot_commit(commit, observations)

        by_subject = await store.search(
            VectorKind.EDGE, A, k=10, filters=SearchFilters(subject_ids={"s2"})
        )
        by_object = await store.search(
            VectorKind.EDGE, A, k=10, filters=SearchFilters(object_ids={"o1"})
        )
        by_relation = await store.search(
            VectorKind.EDGE, A, k=10, filters=SearchFilters(relation_type_ids={9})
        )

        assert [hit.id for hit in by_subject] == ["f2"]
        assert [hit.id for hit in by_object] == ["f1"]
        assert [hit.id for hit in by_relation] == ["f2"]

    async def test_each_vector_carries_the_validity_of_its_fact(self) -> None:
        """A search by instant keeps the fact that was valid then."""
        embedder = RecordingEmbedder(vectors={FIRST: A, SECOND: ORTHOGONAL})
        store = await ready_store(embedder)
        observer = FactVectorObserver(VectorWriter(store))
        commit = Commit(
            (
                fact("f1", evidence=("o1",)),
                fact("f2", evidence=("o2",), valid_to=T0 + timedelta(days=1)),
            )
        )
        observations = {
            "o1": observation("o1", FIRST),
            "o2": observation("o2", SECOND),
        }

        await observer.after_slot_commit(commit, observations)

        during = await store.search(
            VectorKind.EDGE,
            A,
            k=10,
            filters=SearchFilters(
                temporal=TemporalFilter(valid_at_us=to_micros(T0 + timedelta(hours=1)))
            ),
        )
        later = await store.search(
            VectorKind.EDGE,
            A,
            k=10,
            filters=SearchFilters(
                temporal=TemporalFilter(valid_at_us=to_micros(T0 + timedelta(days=2)))
            ),
        )

        assert [hit.id for hit in during] == ["f1", "f2"]
        assert [hit.id for hit in later] == ["f1"]

    async def test_skips_a_fact_whose_statement_is_empty(self) -> None:
        """A fact with no statement is counted and gets no vector."""
        embedder = RecordingEmbedder(vectors={FIRST: A})
        store = await ready_store(embedder)
        observer = FactVectorObserver(VectorWriter(store))
        commit = Commit((fact("f1", evidence=("o1",)),))
        observations = {"o1": observation("o1", "")}

        await observer.after_slot_commit(commit, observations)

        assert observer.skipped_empty == 1
        assert await store.exists(VectorKind.EDGE, ["f1"]) == set()

    async def test_skips_a_fact_whose_evidence_is_not_in_the_commit(self) -> None:
        """A fact whose evidence is unknown gets no vector."""
        embedder = RecordingEmbedder(vectors={FIRST: A})
        store = await ready_store(embedder)
        observer = FactVectorObserver(VectorWriter(store))
        commit = Commit((fact("f1", evidence=("gone",)),))

        await observer.after_slot_commit(commit, {})

        assert observer.skipped_empty == 1
        assert await store.exists(VectorKind.EDGE, ["f1"]) == set()

    async def test_takes_the_statement_from_the_first_evidence_that_has_one(
        self,
    ) -> None:
        """The first evidence with a statement names the vector."""
        embedder = RecordingEmbedder(vectors={FIRST: A, SECOND: ORTHOGONAL})
        store = await ready_store(embedder)
        observer = FactVectorObserver(VectorWriter(store))
        commit = Commit(
            (
                fact("f1", evidence=("gone", "blank", "o1", "o2")),
                fact("f2", evidence=("o2",)),
            )
        )
        observations = {
            "blank": observation("blank", ""),
            "o1": observation("o1", FIRST),
            "o2": observation("o2", SECOND),
        }

        await observer.after_slot_commit(commit, observations)

        vectors = await store.fetch_vectors(VectorKind.EDGE, ["f1", "f2"])

        assert vectors["f1"] == tuple(A)
        assert vectors["f2"] == tuple(ORTHOGONAL)

    async def test_writes_one_vector_for_a_fact_id_that_appears_twice(self) -> None:
        """The first version of a fact id names the vector."""
        embedder = RecordingEmbedder(vectors={FIRST: A})
        store = await ready_store(embedder)
        observer = FactVectorObserver(VectorWriter(store))
        commit = Commit(
            (
                fact("f1", evidence=("o1",)),
                fact(
                    "f1",
                    evidence=("o1",),
                    version=2,
                    valid_to=T0 + timedelta(days=1),
                ),
            )
        )
        observations = {"o1": observation("o1", FIRST)}

        await observer.after_slot_commit(commit, observations)

        later = await store.search(
            VectorKind.EDGE,
            A,
            k=10,
            filters=SearchFilters(
                temporal=TemporalFilter(valid_at_us=to_micros(T0 + timedelta(days=2)))
            ),
        )

        assert [hit.id for hit in later] == ["f1"]

    async def test_writes_no_new_vector_for_a_new_version_of_a_known_fact(self) -> None:
        """A second version of a fact does not embed its statement again."""
        embedder = RecordingEmbedder(vectors={FIRST: A})
        store = await ready_store(embedder)
        observer = FactVectorObserver(VectorWriter(store))
        observations = {"o1": observation("o1", FIRST)}

        await observer.after_slot_commit(
            Commit((fact("f1", evidence=("o1",)),)), observations
        )
        await observer.after_slot_commit(
            Commit((fact("f1", evidence=("o1",), version=2),)), observations
        )

        assert embedder.embedded.count(FIRST) == 1

    async def test_before_slot_commit_writes_nothing(self) -> None:
        """The observer writes its vectors after the commit, not before."""
        embedder = RecordingEmbedder(vectors={FIRST: A})
        store = await ready_store(embedder)
        observer = FactVectorObserver(VectorWriter(store))

        await observer.before_slot_commit(Commit((fact("f1", evidence=("o1",)),)))

        assert await store.exists(VectorKind.EDGE, ["f1"]) == set()
        assert observer.skipped_empty == 0
