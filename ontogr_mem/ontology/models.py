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


class Status(StrEnum):
    """Lifecycle of a type, attribute or pattern. Nothing is ever removed."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"


class Origin(StrEnum):
    """Where a type came from."""

    DECLARED = "declared"
    IMPORTED = "imported"


class RelationKind(StrEnum):
    """How the values of a relation relate to one another."""

    SINGLE = "single"
    MANY = "many"
    JUDGED = "judged"


class Frozen(BaseModel):
    """Base: unknown fields are errors."""

    model_config = ConfigDict(extra="forbid")


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
    identity_attributes: list[str] = Field(default_factory=list)
    parent_id: int | None = None
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
    kind: RelationKind = RelationKind.MANY
    patterns: list[Pattern] = Field(default_factory=list)
    status: Status = Status.ACTIVE
    merged_into: int | None = None
    origin: Origin = Origin.DECLARED


class Ontology(Frozen):
    """One immutable catalog version of a tenant."""

    version: int = 0
    parent_version: int | None = None
    content_hash: str = ""
    next_entity_id: int = 1
    next_relation_id: int = 0
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

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
    identity_attributes: list[str] = Field(default_factory=list)
    parent: Ref | None = None
    origin: Origin = Origin.DECLARED


class AddRelation(Frozen):
    """Add a relation type."""

    op: Literal["add_relation"] = "add_relation"
    label: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    symmetric: bool = False
    kind: RelationKind = RelationKind.MANY
    patterns: list[PatternSpec] = Field(default_factory=list)
    origin: Origin = Origin.DECLARED


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


class SetParent(Frozen):
    """Set or clear the parent of an entity type."""

    op: Literal["set_parent"] = "set_parent"
    target: Ref
    parent: Ref | None


class SetIdentityAttributes(Frozen):
    """Replace the identity attribute names of an entity type."""

    op: Literal["set_identity_attributes"] = "set_identity_attributes"
    target: Ref
    names: list[str]


class SetRelationKind(Frozen):
    """Set how the values of a relation relate to one another."""

    op: Literal["set_relation_kind"] = "set_relation_kind"
    target: Ref
    kind: RelationKind


class SetSymmetric(Frozen):
    """Set whether a relation reads the same in both directions."""

    op: Literal["set_symmetric"] = "set_symmetric"
    target: Ref
    symmetric: bool


class DeprecateType(Frozen):
    """Retire a type. It stays resolvable but is no longer offered."""

    op: Literal["deprecate_type"] = "deprecate_type"
    kind: Kind
    target: Ref


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


Op = Annotated[
    AddEntity
    | AddRelation
    | AddPattern
    | AddAlias
    | Rename
    | SetDescription
    | SetParent
    | SetIdentityAttributes
    | SetRelationKind
    | SetSymmetric
    | DeprecateType
    | DeprecatePattern
    | Reactivate
    | MergeInto,
    Field(discriminator="op"),
]


class Channel(StrEnum):
    """How a change set reached the commit path."""

    USER = "user"
    IMPORT = "import"


class ChangeSet(Frozen):
    """An ordered list of ops against one base version."""

    base_version: int
    ops: list[Op]
    actor: str = "system"
    reason: str = ""
    channel: Channel = Channel.USER
    created_at: datetime | None = None


class Reason(StrEnum):
    """Why an item was dropped or flagged."""

    UNKNOWN_TYPE = "unknown_type"
    FUZZY_MATCH = "fuzzy_match"
    DEPRECATED_TYPE = "deprecated_type"
    AMBIGUOUS_TYPE = "ambiguous_type"
    UNDECLARED_RELATION = "undeclared_relation"
    PATTERN_MISMATCH = "pattern_mismatch"
    UNKNOWN_ENDPOINT = "unknown_endpoint"
    DIRECTION_SWAPPED = "direction_swapped"
