"""Value types of the vector store."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from ontogr_mem.data_models import OPEN_MICROS, FactStatus


class VectorKind(StrEnum):
    """What a vector embeds."""

    NODE = "node"
    EDGE = "edge"
    EPISODE = "episode"


@dataclass(frozen=True, slots=True)
class Validity:
    """The current version of a fact, mirroring the properties on its edge.

    Attributes:
        status: Lifecycle state of this fact version.
        valid_from_us: Start of the fact's validity, included.
        recorded_at_us: When this version was recorded.
        valid_to_us: End of validity, excluded. `OPEN_MICROS` while open.
        expired_at_us: When this version stopped being current. `OPEN_MICROS` while
            current.
    """

    status: FactStatus
    valid_from_us: int
    recorded_at_us: int
    valid_to_us: int = OPEN_MICROS
    expired_at_us: int = OPEN_MICROS

    def __post_init__(self) -> None:
        """Normalise the status, so a bad one cannot reach a query."""
        try:
            status = FactStatus(self.status)
        except ValueError:
            msg = f"unknown status {self.status!r}"
            raise ValueError(msg) from None
        # The dataclass is frozen, so the normalised value is written in place.
        object.__setattr__(self, "status", status)


@dataclass(frozen=True, slots=True)
class NodeExtras:
    """Node properties a search can filter on.

    Attributes:
        entity_type: Numeric id of the entity type.
        status: Lifecycle status of the entity.
    """

    entity_type: int | None = None
    status: str | None = None


@dataclass(frozen=True, slots=True)
class EdgeExtras:
    """Edge properties a search can filter on.

    Attributes:
        subject_id: Id of the node the fact starts at.
        object_id: Id of the node the fact ends at.
        validity: Current version of the fact.
        relation_type_id: Numeric id of the relation type.
    """

    subject_id: str
    object_id: str
    validity: Validity
    relation_type_id: int | None = None


Extras = NodeExtras | EdgeExtras


@dataclass(frozen=True, slots=True)
class EmbedItem:
    """One object to embed.

    Attributes:
        id: Id of the graph node, the `fact_id` of the edge, or the episode id.
        text: Text to embed. The store normalises it by the template of the kind.
        extras: Kind-specific properties. Required for edges.
    """

    id: str
    text: str
    extras: Extras | None = None


@dataclass(frozen=True, slots=True)
class Hit:
    """A search hit.

    Attributes:
        id: Id of the matching object.
        score: Closeness in [0, 1], higher is closer: (1 + cosine) / 2.
        rank: Position in the result, starting at 0.
    """

    id: str
    score: float
    rank: int


class FailReason(StrEnum):
    """Why an item was not written."""

    EMPTY_TEXT = "empty_text"
    EMBED_FAILED = "embed_failed"
    EMBED_TIMEOUT = "embed_timeout"
    EMPTY_VECTOR = "empty_vector"
    ZERO_NORM = "zero_norm"
    NON_FINITE = "non_finite"
    BAD_DIMENSION = "bad_dimension"
    MISSING_NODE = "missing_node"
    REJECTED = "rejected"
    STORE_ERROR = "store_error"


@dataclass(frozen=True, slots=True)
class FailedItem:
    """An item that was attempted and not written."""

    id: str
    reason: FailReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Outcome of `VectorStore.upsert`.

    Attributes:
        attempted: Distinct ids passed in.
        embedded: Items sent to the embedder because their text changed.
        unchanged: Items skipped because the stored text hash matched.
        written: Vectors the backend stored.
        failed: Every item that was not written, with the reason.
    """

    attempted: int
    embedded: int
    unchanged: int
    written: int
    failed: tuple[FailedItem, ...] = ()

    @property
    def ok(self) -> bool:
        """True when nothing failed."""
        return not self.failed


@dataclass(frozen=True, slots=True)
class ValidityUpdate:
    """Change of the validity mirror of one fact, without re-embedding.

    Attributes:
        fact_id: Id of the fact whose edge carries the vector.
        validity: The new current version.
    """

    fact_id: str
    validity: Validity


@dataclass(frozen=True, slots=True)
class TemporalFilter:
    """Edge validity selection.

    The default is current belief: status `active`, not expired, and valid at
    `valid_at_us` when given. `history` disables every validity predicate.
    `known_at_us` asks about past belief, which only a backend with a validity
    mirror can answer.

    Attributes:
        valid_at_us: Keep facts valid at this instant.
        known_at_us: Keep facts believed at this instant.
        history: Keep every version of the fact.
    """

    valid_at_us: int | None = None
    known_at_us: int | None = None
    history: bool = False

    def __post_init__(self) -> None:
        """Reject a contradictory selection."""
        if self.history and (
            self.valid_at_us is not None or self.known_at_us is not None
        ):
            msg = "history excludes valid_at_us and known_at_us"
            raise ValueError(msg)

    @property
    def needs_graph(self) -> bool:
        """True when the mirror cannot answer and the graph must re-check."""
        return self.known_at_us is not None


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Typed filters; each applies only to the kinds that have the property.

    Attributes:
        entity_types: Keep nodes whose entity type is in this set.
        statuses: Keep nodes whose status is in this set.
        relation_type_ids: Keep edges whose relation type is in this set.
        subject_ids: Keep edges that start at one of these nodes.
        object_ids: Keep edges that end at one of these nodes.
        temporal: Edge validity selection. The default is current belief.
    """

    entity_types: frozenset[int] | None = None
    statuses: frozenset[str] | None = None
    relation_type_ids: frozenset[int] | None = None
    subject_ids: frozenset[str] | None = None
    object_ids: frozenset[str] | None = None
    temporal: TemporalFilter | None = None

    def for_kind(self, kind: VectorKind) -> "SearchFilters":
        """Return these filters after refusing the ones `kind` cannot honour.

        Raises:
            ValueError: If a filter names a property the kind does not have.
        """
        bad: list[str] = []
        if kind is not VectorKind.NODE:
            bad.extend(
                name
                for name, value in (
                    ("entity_types", self.entity_types),
                    ("statuses", self.statuses),
                )
                if value is not None
            )
        if kind is not VectorKind.EDGE:
            bad.extend(
                name
                for name, value in (
                    ("relation_type_ids", self.relation_type_ids),
                    ("subject_ids", self.subject_ids),
                    ("object_ids", self.object_ids),
                    ("temporal", self.temporal),
                )
                if value is not None
            )
        if bad:
            msg = f"filters {bad} do not apply to kind {kind.value}"
            raise ValueError(msg)
        return self

    @property
    def narrows(self) -> bool:
        """True when a property filter, other than the temporal default, is set."""
        return any(
            value is not None
            for value in (
                self.entity_types,
                self.statuses,
                self.relation_type_ids,
                self.subject_ids,
                self.object_ids,
            )
        )


@dataclass(frozen=True, slots=True)
class SpaceSpec:
    """The embedding space of one kind.

    Attributes:
        kind: The kind of vector.
        model: Embedding model id.
        dim: Vector length.
        text_template: Id of the text template.
        template_version: Version of the text template.
    """

    kind: VectorKind
    model: str
    dim: int
    text_template: str
    template_version: int

    def __post_init__(self) -> None:
        """Reject a dimension no backend can index."""
        if not 1 <= self.dim <= 4000:
            msg = "dim must be in 1..4000"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a backend can do; callers branch on these, not on type.

    Attributes:
        validity_mirror: The backend stores the validity of a fact, so it can
            answer `known_at_us` searches and accept validity updates.
        exact_scan: The backend can search every row.
        iterative_scan: The backend refines a filtered search in the index.
        ann_index: The backend searches an approximate nearest-neighbour index.
        filters_rows: The backend drops rows after the index returned them, so
            a search has to fetch more than `k` to fill it. The graph backends
            read the validity of a fact from the graph and drop the expired
            ones; the in-memory backend filters in Python.
    """

    validity_mirror: bool = True
    exact_scan: bool = True
    iterative_scan: bool = False
    ann_index: bool = False
    filters_rows: bool = False


@dataclass(frozen=True, slots=True)
class Row:
    """An embedded row handed to a backend."""

    id: str
    text_hash: str
    vector: tuple[float, ...]
    extras: Extras | None = None


@dataclass(frozen=True, slots=True)
class BackendWrite:
    """Backend outcome for one batch of rows.

    Attributes:
        written: Vectors stored.
        missing: Ids of rows whose graph object does not exist.
        rejected: Rows the database refused, with the reason it gave.
    """

    written: int
    missing: tuple[str, ...] = ()
    rejected: tuple[FailedItem, ...] = ()


def as_ids(ids: Sequence[str]) -> list[str]:
    """De-duplicate ids, keeping first-seen order."""
    return list(dict.fromkeys(ids))
