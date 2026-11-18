"""Type and modality filters for one retrieval request."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ontogr_mem.data_models.facts import Modality
from ontogr_mem.ontology.links import TypeLinks
from ontogr_mem.service.errors import InvalidArgumentError
from ontogr_mem.temporal.query import ACTUAL_ONLY


class _ScopeRequest(Protocol):
    @property
    def entity_types(self) -> Sequence[str]: ...

    @property
    def exclude_entity_types(self) -> Sequence[str]: ...

    @property
    def relation_types(self) -> Sequence[str]: ...

    @property
    def exclude_relation_types(self) -> Sequence[str]: ...

    @property
    def modalities(self) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class SearchScope:
    """The type and modality filters of one search, as stored ids."""

    entity_types: frozenset[int] | None = None
    exclude_entity_types: frozenset[int] = frozenset()
    relation_types: frozenset[int] | None = None
    exclude_relation_types: frozenset[int] = frozenset()
    modalities: frozenset[Modality] = ACTUAL_ONLY

    @property
    def filters_facts(self) -> bool:
        """Return whether any type filter is set."""
        return any(
            (
                self.entity_types is not None,
                bool(self.exclude_entity_types),
                self.relation_types is not None,
                bool(self.exclude_relation_types),
            )
        )


async def build_scope(
    tenant_id: str,
    request: _ScopeRequest,
    load_links: Callable[[], Awaitable[TypeLinks]],
) -> SearchScope:
    """Resolve request labels against the tenant's ontology snapshot."""
    lists = (
        request.entity_types,
        request.exclude_entity_types,
        request.relation_types,
        request.exclude_relation_types,
    )
    if not any(lists):
        return SearchScope(
            modalities=frozenset(Modality(value) for value in request.modalities)
        )
    links = await load_links()

    def resolve(labels: Sequence[str], *, entity: bool) -> frozenset[int] | None:
        if not labels:
            return None if entity else frozenset()
        family = links.entity_family if entity else links.relation_family
        result: set[int] = set()
        kind = "entity" if entity else "relation"
        for label in labels:
            found = family(label)
            if found is None:
                raise InvalidArgumentError(
                    f"unknown {kind} type {label!r} in tenant {tenant_id}"
                )
            result.update(found)
        return frozenset(result)

    return SearchScope(
        entity_types=resolve(request.entity_types, entity=True),
        exclude_entity_types=resolve(request.exclude_entity_types, entity=True)
        or frozenset(),
        relation_types=resolve(request.relation_types, entity=False)
        if request.relation_types
        else None,
        exclude_relation_types=resolve(request.exclude_relation_types, entity=False)
        or frozenset(),
        modalities=frozenset(Modality(value) for value in request.modalities),
    )
