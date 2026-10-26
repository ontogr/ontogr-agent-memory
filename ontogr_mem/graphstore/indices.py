"""Backend-neutral declaration of the indexes and constraints a graph needs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ontogr_mem.graphstore.errors import ValidationError


@dataclass(frozen=True)
class RangeIndex:
    """Range index on a node label or relationship type."""

    label: str
    prop: str
    edge: bool = False


@dataclass(frozen=True)
class UniqueConstraint:
    """Unique constraint on one node property."""

    label: str
    prop: str


@dataclass(frozen=True)
class FulltextIndex:
    """Fulltext index over one or more properties of a node label."""

    label: str
    props: tuple[str, ...]


@dataclass(frozen=True)
class IndexSpec:
    """One index or constraint a schema requires.

    Attributes:
        kind: `range`, `unique`, or `fulltext`.
        label: Node label, or relationship type when `edge` is True.
        prop: Indexed property.
        edge: True when the index is on a relationship type.
        props: Properties of a fulltext index; `prop` is unused for that kind.
    """

    kind: Literal["range", "unique", "fulltext"]
    label: str
    prop: str
    edge: bool = False
    props: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchemaPart:
    """The labels, indexes and edge keys one component needs.

    Every label in `labels` gets a unique constraint on its `id` property.

    Attributes:
        name: Name of the component that owns the part.
        labels: Node labels identified by `id`.
        ranges: Range indexes beyond the `id` indexes.
        uniques: Unique constraints beyond the `id` constraints.
        fulltexts: Fulltext indexes on node properties.
        edge_keys: Properties that, with the endpoints, identify an edge of each
            relationship type.
        requires: Names of the other schema parts this part needs. `merge_parts`
            pulls them in, so a tenant opened with this part also gets the
            labels this part's statements refer to.
    """

    name: str
    labels: tuple[str, ...] = ()
    ranges: tuple[RangeIndex, ...] = ()
    uniques: tuple[UniqueConstraint, ...] = ()
    edge_keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    fulltexts: tuple[FulltextIndex, ...] = ()


def plan(schema: SchemaPart) -> tuple[IndexSpec, ...]:
    """Return the index and constraint requirements of `schema`.

    Every label gets a unique constraint on its `id`. A unique constraint
    implies an index on the same property, so a constrained property needs no
    range spec of its own. The backend creates the index that its server
    requires. Every backend builds the same plan, so their drift reports can be
    compared.
    """
    specs: list[IndexSpec] = [
        IndexSpec("range", r.label, r.prop, r.edge) for r in schema.ranges
    ]
    specs += [IndexSpec("unique", label, "id") for label in schema.labels]
    specs += [IndexSpec("unique", u.label, u.prop) for u in schema.uniques]
    specs += [
        IndexSpec("fulltext", index.label, "", props=index.props)
        for index in schema.fulltexts
    ]
    return tuple(dict.fromkeys(specs))


def merge_parts(
    parts: Sequence[SchemaPart], registry: Mapping[str, SchemaPart] | None = None
) -> SchemaPart:
    """Combine schema parts into one, dropping duplicates.

    A part names the other parts it needs in `requires`. Those parts are
    pulled in from `registry`, transitively and before the part that needs
    them, so the merged part holds every label its statements refer to.

    Args:
        parts: The parts to combine.
        registry: The parts a requirement may name, by name.

    Returns:
        One part holding the labels, indexes, constraints and edge keys of
        every part that was pulled in.

    Raises:
        ValidationError: If a required part is not in `registry`.
    """
    known: dict[str, SchemaPart] = dict(registry or {})
    for part in parts:
        known.setdefault(part.name, part)
    ordered: list[SchemaPart] = []
    seen: set[str] = set()

    def add(part: SchemaPart) -> None:
        if part.name in seen:
            return
        seen.add(part.name)
        for requirement in part.requires:
            required = known.get(requirement)
            if required is None:
                msg = (
                    f"schema part {part.name!r} requires {requirement!r}, "
                    "which is not a known schema part"
                )
                raise ValidationError(msg)
            add(required)
        ordered.append(part)

    for part in parts:
        add(part)
    edge_keys: dict[str, tuple[str, ...]] = {}
    for part in ordered:
        edge_keys.update(part.edge_keys)
    return SchemaPart(
        name="+".join(part.name for part in ordered),
        labels=tuple(dict.fromkeys(label for part in ordered for label in part.labels)),
        ranges=tuple(dict.fromkeys(r for part in ordered for r in part.ranges)),
        uniques=tuple(dict.fromkeys(u for part in ordered for u in part.uniques)),
        fulltexts=tuple(dict.fromkeys(f for part in ordered for f in part.fulltexts)),
        edge_keys=edge_keys,
    )


@dataclass(frozen=True)
class IndexResult:
    """Outcome of creating one index or constraint."""

    kind: str
    target: str
    status: Literal["created", "exists"]


@dataclass(frozen=True)
class DriftReport:
    """Differences between the declared schema and the live graph."""

    missing: tuple[str, ...] = ()
    not_operational: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when nothing differs."""
        return not (self.missing or self.not_operational)

    def lines(self) -> tuple[str, ...]:
        """Human-readable drift lines."""
        return (
            *(f"missing: {m}" for m in self.missing),
            *(f"not operational: {m}" for m in self.not_operational),
        )
