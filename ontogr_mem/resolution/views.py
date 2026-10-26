"""The stored rows one decision reads.

A decision looks at three rows of the store: the entities that carry the
group's uri, the entities an alias of the group's key resolves to, and the
candidate entities themselves. The view holds them in one place.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ontogr_mem.data_models.entities import Entity
from ontogr_mem.resolution.grouping import Group
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.resolution.scoring import Candidate


@dataclass(frozen=True, slots=True)
class DecisionView:
    """The stored rows one decision reads.

    Attributes:
        uri_entities: Entities carrying the group's uri.
        alias_hits: Entities an alias of the group's strict key resolves to,
            with True when the entity's canonical name is the hit.
        candidates: Candidate entities by id.
    """

    uri_entities: tuple[Entity, ...] = ()
    alias_hits: tuple[tuple[Entity, bool], ...] = ()
    candidates: Mapping[str, Entity] = field(default_factory=dict)


async def view_for(
    store: ResolverStore, group: Group, candidates: Sequence[Candidate]
) -> DecisionView:
    """Read the rows `collect_matches` needs for one group."""
    uri_entities = tuple(await store.entities_by_uri(group.uri)) if group.uri else ()
    alias_hits = tuple(
        (entity, entity.strict_key == group.strict)
        for entity in await store.entities_by_strict_key(group.strict)
    )
    rows = await store.get_entities([candidate.entity_id for candidate in candidates])
    return DecisionView(
        uri_entities=uri_entities,
        alias_hits=alias_hits,
        candidates={entity.entity_id: entity for entity in rows},
    )
