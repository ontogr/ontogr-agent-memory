"""Group a batch of extracted entities by name.

Two mentions join the same group when their strict keys match, their types are
compatible and their identity attributes do not conflict. A group carries the
merged description, attributes and surface forms of its members.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ontogr_mem.data_models import ExtractedEntity
from ontogr_mem.resolution.keys import loose_key, strict_key
from ontogr_mem.resolution.policy import (
    ResolutionPolicy,
    TypeRel,
    TypeRelations,
    identity_conflict,
    identity_of,
)


@dataclass(slots=True)
class Group:
    """Mentions of one name inside one batch that collapsed together.

    Attributes:
        members: Extracted entities in the group, in batch order.
        name: Canonical name of the group.
        type_label: Most specific type label seen in the group.
        strict: Strict key of the name.
        loose: Blocking key of the name.
        descriptions: Unique descriptions in the group.
        attributes: Attributes collected from the members.
        uri: Bound identifier, if any member carried one.
        identity: Normalized identity attributes of the group.
        surface_forms: Names as written by the members.
    """

    members: list[ExtractedEntity]
    name: str
    type_label: str
    strict: str
    loose: str
    descriptions: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    uri: str | None = None
    identity: dict[str, str] = field(default_factory=dict)
    surface_forms: list[str] = field(default_factory=list)


def build_groups(
    entities: Sequence[ExtractedEntity],
    policy: ResolutionPolicy,
    relations: TypeRelations,
) -> list[Group]:
    """Collapse same-key, compatible-type, non-conflicting mentions of a batch."""
    groups: list[Group] = []
    for extracted in sorted(entities, key=lambda e: (e.episode_id, e.local_id)):
        key = strict_key(extracted.name)
        if not key:
            continue
        label = extracted.type_label or ""
        identity = identity_of(policy, label, extracted.attributes)
        target: Group | None = None
        for group in groups:
            if group.strict != key:
                continue
            relation = policy.relation(group.type_label, label, relations)
            if relation not in (TypeRel.SAME, TypeRel.NESTED):
                continue
            if identity_conflict(group.identity, identity):
                continue
            target = group
            break
        uri = extracted.attributes.get("uri")
        if target is None:
            target = Group(
                members=[],
                name=extracted.name.strip(),
                type_label=label,
                strict=key,
                loose=loose_key(extracted.name),
                uri=str(uri) if uri else None,
            )
            groups.append(target)
        else:
            target.type_label = relations.more_specific(target.type_label, label)
            target.uri = target.uri or (str(uri) if uri else None)
        target.members.append(extracted)
        if extracted.description and extracted.description not in target.descriptions:
            target.descriptions.append(extracted.description)
        for name, value in extracted.attributes.items():
            target.attributes.setdefault(name, value)
        for name, value in identity.items():
            target.identity.setdefault(name, value)
        if extracted.name.strip() not in target.surface_forms:
            target.surface_forms.append(extracted.name.strip())
    return groups


def group_text(group: Group) -> str:
    """Return the text a group is embedded as."""
    description = group.descriptions[0] if group.descriptions else ""
    return f"{group.name} | {group.type_label} | {description[:200]}"
