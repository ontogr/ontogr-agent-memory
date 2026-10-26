"""The catalog and the objects around it: Pydantic, JSON round-trip."""

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ontogr_mem.data_models.type_catalog import UNKNOWN_LABEL
from ontogr_mem.ontology.errors import InvalidChange
from ontogr_mem.ontology.names import normalize


UNKNOWN_ID = 0


class Kind(StrEnum):
    """Which list of the catalog a reference points into."""

    ENTITY = "entity"
    RELATION = "relation"


class ValueType(StrEnum):
    """Attribute value types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    LIST = "list"


class Status(StrEnum):
    """Lifecycle of a type, attribute or pattern. Nothing is ever removed."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"


class Origin(StrEnum):
    """Where a type came from."""

    DECLARED = "declared"
    IMPORTED = "imported"
    ACCEPTED = "accepted"
    AUTO_ACCEPTED = "auto_accepted"


class Owner(StrEnum):
    """Who may write an attribute."""

    USER = "user"
    SYSTEM = "system"
    TABLE = "table"


class Mode(StrEnum):
    """Enforcement level for one kind."""

    STRICT = "strict"
    PREFERRED = "preferred"
    OPEN = "open"


class UnknownEdges(StrEnum):
    """What to do with a pattern-checked edge that touches an Unknown entity."""

    KEEP = "keep"
    DROP = "drop"


class Frozen(BaseModel):
    """Base: unknown fields are errors."""

    model_config = ConfigDict(extra="forbid")


class Attribute(Frozen):
    """A typed property of an entity or relation type."""

    name: str
    value_type: ValueType = ValueType.STRING
    description: str = ""
    required: bool = False
    extractable: bool = True
    structured: bool = False
    owner: Owner = Owner.USER
    max_length: int | None = None
    status: Status = Status.ACTIVE


class Pattern(Frozen):
    """An ordered (source type, target type) pair a relation may connect."""

    src: int
    tgt: int
    status: Status = Status.ACTIVE


class Entity(Frozen):
    """An entity type."""

    type_id: int
    label: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    parent_id: int | None = None
    uri: str | None = None
    properties: list[Attribute] = Field(default_factory=list)
    status: Status = Status.ACTIVE
    merged_into: int | None = None
    origin: Origin = Origin.DECLARED


class Relation(Frozen):
    """A relation type. No patterns means any entity pair."""

    type_id: int
    label: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    symmetric: bool = False
    exclusive: bool = False
    patterns: list[Pattern] = Field(default_factory=list)
    properties: list[Attribute] = Field(default_factory=list)
    status: Status = Status.ACTIVE
    merged_into: int | None = None
    origin: Origin = Origin.DECLARED
    uri: str | None = None


class TableMapping(Frozen):
    """Maps a structured source's columns to attributes of one entity type."""

    source: str
    entity_id: int
    columns: dict[str, str]
    derived: bool = False


class Ontology(Frozen):
    """One immutable catalog version of a tenant."""

    version: int = 0
    parent_version: int | None = None
    content_hash: str = ""
    entity_mode: Mode = Mode.PREFERRED
    relation_mode: Mode = Mode.PREFERRED
    unknown_endpoint_edges: UnknownEdges = UnknownEdges.KEEP
    next_entity_id: int = 1
    next_relation_id: int = 0
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    tables: list[TableMapping] = Field(default_factory=list)

    @classmethod
    def genesis(cls) -> "Ontology":
        """Return version 0: the reserved Unknown entity and nothing else."""
        unknown = Entity(
            type_id=UNKNOWN_ID,
            label=UNKNOWN_LABEL,
            description="No catalog type fits this item.",
        )
        out = cls(entities=[unknown])
        return out.sealed()

    def compute_hash(self) -> str:
        """Hash of the content; version bookkeeping is left out on purpose."""
        data = self.model_dump(
            mode="json", exclude={"version", "parent_version", "content_hash"}
        )
        blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def sealed(self) -> "Ontology":
        """Return a copy with ``content_hash`` computed from the content."""
        return self.model_copy(update={"content_hash": self.compute_hash()})

    def entity(self, type_id: int) -> Entity | None:
        """Return the entity type with this id, or None."""
        return next((e for e in self.entities if e.type_id == type_id), None)

    def relation(self, type_id: int) -> Relation | None:
        """Return the relation type with this id, or None."""
        return next((r for r in self.relations if r.type_id == type_id), None)

    def relation_by_label(self, label: str) -> Relation | None:
        """Return an active relation matching its label or an alias."""
        normalized = normalize(label)
        return next(
            (
                relation
                for relation in self.relations
                if relation.status is Status.ACTIVE
                and normalized in self.labels_of(relation)
            ),
            None,
        )

    def entity_types(self) -> list[Entity]:
        """Return the entity types."""
        return list(self.entities)

    def relation_types(self) -> list[Relation]:
        """Return the relation types."""
        return list(self.relations)

    def types(self, kind: Kind) -> Sequence[Entity | Relation]:
        """The entity or the relation types, whichever ``kind`` names."""
        return self.entities if kind is Kind.ENTITY else self.relations

    def type_of(self, kind: Kind, type_id: int) -> Entity | Relation | None:
        """Return the type of ``kind`` with this id, or None."""
        return next((t for t in self.types(kind) if t.type_id == type_id), None)

    def mode(self, kind: Kind) -> Mode:
        """Return the enforcement mode that applies to ``kind``."""
        return self.entity_mode if kind is Kind.ENTITY else self.relation_mode

    def follow(self, kind: Kind, type_id: int) -> int:
        """Follow ``merged_into`` pointers to the live type id.

        Returns ``type_id`` itself when the type was not merged, and also when the
        catalog has no type with that id, so there is nothing to follow. A pointer
        loop stops at the id already seen.

        Raises:
            InvalidChange: A pointer in the chain names a type the catalog does not
                have.
        """
        seen = {type_id}
        cur = type_id
        t = self.type_of(kind, cur)
        if t is None:
            return cur
        label = t.label
        while t.merged_into is not None and t.merged_into not in seen:
            cur = t.merged_into
            seen.add(cur)
            t = self.type_of(kind, cur)
            if t is None:
                raise InvalidChange("bad merge pointer", item=label)
        return cur

    def ancestors(self, type_id: int) -> list[int]:
        """Return the parent chain of an entity type, nearest parent first."""
        out: list[int] = []
        cur = self.entity(type_id)
        while cur is not None and cur.parent_id is not None:
            out.append(cur.parent_id)
            cur = self.entity(cur.parent_id)
        return out

    def active_entities(self) -> list[Entity]:
        """Return the entity types whose status is active, Unknown included."""
        return [e for e in self.entities if e.status is Status.ACTIVE]

    def active_relations(self) -> list[Relation]:
        """Return the relation types whose status is active."""
        return [r for r in self.relations if r.status is Status.ACTIVE]

    def has_patterns(self) -> bool:
        """True if any active relation declares an active pattern."""
        return any(
            p.status is Status.ACTIVE
            for r in self.active_relations()
            for p in r.patterns
        )

    def pattern_allows(self, rel: Relation, src: int, tgt: int) -> bool:
        """Ordered check with subsumption; an empty pattern list allows anything."""
        pats = [p for p in rel.patterns if p.status is Status.ACTIVE]
        if not pats:
            return True
        s_chain = {src, *self.ancestors(src)}
        t_chain = {tgt, *self.ancestors(tgt)}
        return any(
            self.follow(Kind.ENTITY, p.src) in s_chain
            and self.follow(Kind.ENTITY, p.tgt) in t_chain
            for p in pats
        )

    def labels_of(self, t: Entity | Relation) -> set[str]:
        """Return the normalized label and aliases of a type, as a set."""
        return {normalize(t.label), *(normalize(a) for a in t.aliases)}


Ref = int | str
"""A type reference: its id, or its label (resolved when the op is applied)."""


class PatternSpec(Frozen):
    """A pattern whose endpoints may be labels of types added in the same set."""

    src: Ref
    tgt: Ref


class AddEntity(Frozen):
    """Add an entity type."""

    op: Literal["add_entity"] = "add_entity"
    label: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    parent: Ref | None = None
    uri: str | None = None
    properties: list[Attribute] = Field(default_factory=list)
    origin: Origin = Origin.DECLARED


class AddRelation(Frozen):
    """Add a relation type."""

    op: Literal["add_relation"] = "add_relation"
    label: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    symmetric: bool = False
    exclusive: bool = False
    patterns: list[PatternSpec] = Field(default_factory=list)
    properties: list[Attribute] = Field(default_factory=list)
    uri: str | None = None
    origin: Origin = Origin.DECLARED


class AddAttribute(Frozen):
    """Add an attribute to an existing type."""

    op: Literal["add_attribute"] = "add_attribute"
    kind: Kind
    target: Ref
    attribute: Attribute


class AddPattern(Frozen):
    """Add an ordered pattern to a relation type."""

    op: Literal["add_pattern"] = "add_pattern"
    relation: Ref
    pattern: PatternSpec


class AddAlias(Frozen):
    """Add an alias to a type."""

    op: Literal["add_alias"] = "add_alias"
    kind: Kind
    target: Ref
    alias: str


class Rename(Frozen):
    """Change a type's label; the old label stays as an alias."""

    op: Literal["rename"] = "rename"
    kind: Kind
    target: Ref
    new_label: str


class SetDescription(Frozen):
    """Replace a type's description."""

    op: Literal["set_description"] = "set_description"
    kind: Kind
    target: Ref
    description: str


class SetExclusive(Frozen):
    """Set whether a subject holds at most one current value of a relation."""

    op: Literal["set_exclusive"] = "set_exclusive"
    target: Ref
    exclusive: bool


class DeprecateType(Frozen):
    """Retire a type. It stays resolvable but is no longer offered."""

    op: Literal["deprecate_type"] = "deprecate_type"
    kind: Kind
    target: Ref


class DeprecateAttribute(Frozen):
    """Retire an attribute."""

    op: Literal["deprecate_attribute"] = "deprecate_attribute"
    kind: Kind
    target: Ref
    name: str


class DeprecatePattern(Frozen):
    """Retire a pattern."""

    op: Literal["deprecate_pattern"] = "deprecate_pattern"
    relation: Ref
    pattern: PatternSpec


class Reactivate(Frozen):
    """Make a deprecated type active again."""

    op: Literal["reactivate"] = "reactivate"
    kind: Kind
    target: Ref


class MergeInto(Frozen):
    """Deprecate ``source`` and point it at ``into``."""

    op: Literal["merge_into"] = "merge_into"
    kind: Kind
    source: Ref
    into: Ref


class SetModes(Frozen):
    """Change the enforcement modes."""

    op: Literal["set_modes"] = "set_modes"
    entity_mode: Mode | None = None
    relation_mode: Mode | None = None
    unknown_endpoint_edges: UnknownEdges | None = None


class AddTable(Frozen):
    """Declare a table mapping."""

    op: Literal["add_table"] = "add_table"
    table: TableMapping


Op = Annotated[
    AddEntity
    | AddRelation
    | AddAttribute
    | AddPattern
    | AddAlias
    | Rename
    | SetDescription
    | SetExclusive
    | DeprecateType
    | DeprecateAttribute
    | DeprecatePattern
    | Reactivate
    | MergeInto
    | SetModes
    | AddTable,
    Field(discriminator="op"),
]

ADDITIVE_OPS = (AddEntity, AddRelation, AddAttribute, AddPattern, AddAlias)


class Channel(StrEnum):
    """How a change set reached the commit path."""

    USER = "user"
    PROPOSAL = "proposal"
    IMPORT = "import"
    TABLE = "table"
    OPEN_MODE = "open_mode"


class ChangeSet(Frozen):
    """An ordered list of ops against one base version."""

    base_version: int
    ops: list[Op]
    actor: str = "system"
    reason: str = ""
    channel: Channel = Channel.USER
    created_at: datetime | None = None

    def is_additive(self) -> bool:
        """Return True if every op only adds a type, attribute, pattern or alias."""
        return all(isinstance(o, ADDITIVE_OPS) for o in self.ops)


class UnmatchedKind(StrEnum):
    """What extraction could not place."""

    ENTITY_TYPE = "entity_type"
    RELATION_TYPE = "relation_type"
    ATTRIBUTE = "attribute"
    PATTERN = "pattern"
    FUZZY_ENTITY = "fuzzy_entity"
    FUZZY_RELATION = "fuzzy_relation"


class Unmatched(Frozen):
    """One item extraction could not place, with its evidence."""

    kind: UnmatchedKind
    raw_label: str
    normalized_label: str
    src_label: str = ""
    tgt_label: str = ""
    episode_id: str
    quote: str = ""
    catalog_version: int

    @property
    def row_id(self) -> str:
        """Deterministic id within an episode; recording twice is a no-op."""
        key = "|".join(
            [
                self.kind.value,
                self.normalized_label,
                normalize(self.src_label),
                normalize(self.tgt_label),
            ]
        )
        return hashlib.sha256(key.encode()).hexdigest()[:32]


class ProposalStatus(StrEnum):
    """Lifecycle of a proposal."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class ProposalSource(StrEnum):
    """Where a proposal came from."""

    OBSERVED = "observed"
    DISCOVERY = "discovery"
    IMPORT = "import"
    USER = "user"


class Example(Frozen):
    """Evidence for a proposal."""

    episode_id: str
    quote: str


class Proposal(Frozen):
    """An additions-only change set waiting for a decision."""

    proposal_id: str
    ops: list[Op]
    labels: list[str]
    support: int
    examples: list[Example] = Field(default_factory=list)
    source: ProposalSource = ProposalSource.OBSERVED
    status: ProposalStatus = ProposalStatus.PENDING
    decided_by: str = ""
    decided_at: datetime | None = None
    reason: str = ""
    base_version: int = 0


class RenderedView(Frozen):
    """What the prompt showed: position ``i`` means type ``shown[i]``."""

    catalog_version: int
    content_hash: str
    kind: Kind
    mode: Mode
    shown: list[int]
    rendered_at: datetime | None = None

    def decode(self, position: int) -> int | None:
        """Type id for a position, or None when it was not in the list."""
        if 0 <= position < len(self.shown):
            return self.shown[position]
        return None


class Reason(StrEnum):
    """Why an item was dropped or flagged."""

    UNKNOWN_TYPE = "unknown_type"
    FUZZY_MATCH = "fuzzy_match"
    DEPRECATED_TYPE = "deprecated_type"
    TYPE_ID_OUT_OF_RANGE = "type_id_out_of_range"
    AMBIGUOUS_TYPE = "ambiguous_type"
    UNDECLARED_RELATION = "undeclared_relation"
    PATTERN_MISMATCH = "pattern_mismatch"
    UNKNOWN_ENDPOINT = "unknown_endpoint"
    DIRECTION_SWAPPED = "direction_swapped"
    ENDPOINT_DROPPED = "endpoint_dropped"
    ATTRIBUTE_UNDECLARED = "attribute_undeclared"
    ATTRIBUTE_UNCOERCIBLE = "attribute_uncoercible"
    ATTRIBUTE_TOO_LONG = "attribute_too_long"
    ATTRIBUTE_REQUIRED_MISSING = "attribute_required_missing"
