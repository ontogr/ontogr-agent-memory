"""Backend-neutral types for graph store reads and writes."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphHandle:
    """A tenant's graph, bound to the tenant it was opened for.

    Attributes:
        tenant: Tenant id.
        graph: Name of the tenant's graph on the server.
    """

    tenant: str
    graph: str


@dataclass(frozen=True)
class Node:
    """A node to write.

    Attributes:
        id: Identity value, stored in the `id` property.
        label: Primary label, the only label inside `MERGE`.
        props: Properties to set. Datetimes become integer microseconds, a
            mapping becomes a JSON string under `<key>_json`, and None is
            omitted.
        extra_labels: Labels added after the merge.
    """

    id: str
    label: str
    props: Mapping[str, Any] = field(default_factory=dict)
    extra_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Edge:
    """An edge to write between two existing nodes.

    The identity is `(src_id, dst_id, rel_type)` plus the properties the schema
    declares as edge keys for `rel_type`; those default to an empty string.
    """

    src_label: str
    src_id: str
    dst_label: str
    dst_id: str
    rel_type: str
    props: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeRow:
    """A node read back."""

    id: str
    labels: tuple[str, ...]
    props: Mapping[str, Any]


@dataclass(frozen=True)
class WriteError:
    """One failed item from a write."""

    index: int
    message: str


@dataclass(frozen=True)
class WriteResult:
    """Outcome of a write.

    Attributes:
        attempted: Number of items passed in.
        matched: Number of items that were written or already present.
        created: Number of nodes or edges the server created.
        fell_back_to_single: True if a failed batch was retried item by item.
        errors: Items the server rejected, by index.
        missing: Edges skipped because an endpoint does not exist, by index.
    """

    attempted: int
    matched: int
    created: int | None = None
    fell_back_to_single: bool = False
    errors: tuple[WriteError, ...] = ()
    missing: tuple[int, ...] = ()

    @property
    def complete(self) -> bool:
        """True when every attempted item matched and nothing errored."""
        return self.matched == self.attempted and not self.errors


@dataclass(frozen=True)
class NodeSelector:
    """Select nodes of `label` by id."""

    label: str
    ids: Sequence[str]


@dataclass(frozen=True)
class EdgeSelector:
    """Select edges of `rel_type` between `(src, dst)` id pairs."""

    rel_type: str
    src_label: str
    dst_label: str
    pairs: Sequence[tuple[str, str]]


@dataclass(frozen=True)
class GraphStats:
    """Counts and schema drift for one tenant graph."""

    node_count: int
    edge_count: int
    labels: Mapping[str, int]
    relationship_types: Mapping[str, int]
    drift: tuple[str, ...]


@dataclass(frozen=True)
class Health:
    """Server reachability."""

    ok: bool
    latency_ms: float
    breaker: str
    detail: str = ""


@dataclass(frozen=True)
class Guard:
    """The condition and update of a guarded write.

    Attributes:
        label: Label of the guard node.
        id: Id of the guard node.
        expect: Properties that must equal these values for the guard to hold.
            None means the property must be absent.
        set: Properties written to the guard node when it holds.
        create_with: If given, a missing guard node is created with these
            properties only when they satisfy the expectation.
    """

    label: str
    id: str
    expect: Mapping[str, Any]
    set: Mapping[str, Any]
    create_with: Mapping[str, Any] | None = None
