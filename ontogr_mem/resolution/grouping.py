"""Group a batch of extracted entities by name.

Two mentions join the same group when their strict keys match, their types are
compatible and their identity attributes do not conflict. A group carries the
merged description, attributes and surface forms of its members.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ontogr_mem.data_models import ExtractedEntity
from ontogr_mem.ontology.links import TypeLinks, TypeRel
from ontogr_mem.ontology.models import UNKNOWN_ID
from ontogr_mem.resolution.keys import loose_key, strict_key
from ontogr_mem.resolution.policy import identity_conflict, identity_of


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
        ontology_type_id: Stable id of the group type.
        proposed_type: Type name the model gave while the type is Unknown.
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
    ontology_type_id: int = 0
    proposed_type: str = ""


def _promote_group_type(
    target: Group,
    type_id: int,
    extracted: ExtractedEntity,
    links: TypeLinks,
) -> None:
    """Promote the group's type triple when a member's type is more specific."""
    winner = links.more_specific(target.ontology_type_id, type_id)
    if winner != target.ontology_type_id:
        target.ontology_type_id = winner
        target.type_label = links.label(winner)
    target.proposed_type = (
        target.proposed_type or extracted.proposed_type if winner == UNKNOWN_ID else ""
    )


def _find_group(
    groups: Sequence[Group],
    key: str,
    type_id: int,
    identity: dict[str, str],
    links: TypeLinks,
) -> Group | None:
    """Return the first group that accepts this mention, or None."""
    for group in groups:
        if group.strict != key:
            continue
        if links.relation(group.ontology_type_id, type_id) is TypeRel.DISTINCT:
            continue
        if identity_conflict(group.identity, identity):
            continue
        return group
    return None


def build_groups(
    entities: Sequence[ExtractedEntity],
    links: TypeLinks,
) -> list[Group]:
    """Collapse same-key, compatible-type, non-conflicting mentions of a batch."""
    groups: list[Group] = []
    for extracted in sorted(entities, key=lambda e: (e.episode_id, e.local_id)):
        key = strict_key(extracted.name)
        if not key:
            continue
        if extracted.ontology_type_id is None:
            raise ValueError(
                f"entity {extracted.local_id} was not stamped with an ontology type"
            )
        type_id = links.canonical(extracted.ontology_type_id)
        identity = identity_of(links, type_id, extracted.attributes)
        target = _find_group(groups, key, type_id, identity, links)
        uri = extracted.attributes.get("uri")
        if target is None:
            target = Group(
                members=[],
                name=extracted.name.strip(),
                type_label=links.label(type_id),
                ontology_type_id=type_id,
                proposed_type=extracted.proposed_type if type_id == UNKNOWN_ID else "",
                strict=key,
                loose=loose_key(extracted.name),
                uri=str(uri) if uri else None,
            )
            groups.append(target)
        else:
            _promote_group_type(target, type_id, extracted, links)
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
