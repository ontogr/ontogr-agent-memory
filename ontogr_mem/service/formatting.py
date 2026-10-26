"""Shape stored facts, entities and episodes for service responses."""

from collections.abc import Callable, Mapping

from ontogr_mem.data_models.entities import Alias, Entity
from ontogr_mem.retrieval.models import EntityCard, FactResult
from ontogr_mem.service.contracts import EntityOut, EntityRef, FactOut
from ontogr_mem.temporal.query import FactView


def default_statement(subject: str, relation: str, object_: str) -> str:
    """Build the plain statement used for structured facts."""
    return f"{subject} {relation.replace('_', ' ').lower()} {object_}"


def entity_out(tenant_id: str, entity: Entity, aliases: list[Alias]) -> EntityOut:
    """Format a canonical entity and its alternate names."""
    return EntityOut(
        entity_id=entity.entity_id,
        tenant=tenant_id,
        name=entity.canonical_name,
        type_label=entity.type_label,
        summary=entity.summary,
        aliases=[alias.text for alias in aliases],
    )


def entity_card_out(tenant_id: str, card: EntityCard) -> EntityOut:
    """Format an entity card returned by retrieval."""
    return EntityOut(
        entity_id=card.entity_id,
        tenant=tenant_id,
        name=card.name,
        type_label=card.type_label,
        summary=card.summary,
        aliases=list(card.aliases),
    )


def fact_out(
    tenant_id: str,
    view: FactView,
    names: Mapping[str, str],
    label_of: Callable[[int], str],
    *,
    score: float = 0.0,
    episode_ids: tuple[str, ...] = (),
) -> FactOut:
    """Format one temporal fact and resolve its names and relation label."""
    relation = label_of(view.relation_type_id)
    statement = view.statement or default_statement(
        names.get(view.subject_id, view.subject_id),
        relation,
        names.get(view.object_id, view.object_id),
    )
    return FactOut(
        fact_id=view.fact_id,
        tenant=tenant_id,
        statement=statement,
        subject=EntityRef(
            entity_id=view.subject_id,
            name=names.get(view.subject_id, ""),
        ),
        relation=relation,
        object=EntityRef(
            entity_id=view.object_id,
            name=names.get(view.object_id, ""),
        ),
        valid_from=view.valid_from,
        valid_to=view.valid_to,
        recorded_at=view.recorded_at,
        expired_at=view.expired_at,
        status=view.status.value,
        disputed=view.disputed,
        counterpart_ids=list(view.counterpart_ids),
        evidence_count=len(view.evidence),
        episode_ids=list(episode_ids),
        score=score,
    )


def retrieved_fact_out(
    tenant_id: str,
    result: FactResult,
    names: Mapping[str, str],
    label_of: Callable[[int], str],
) -> FactOut:
    """Format a ranked fact result."""
    return fact_out(
        tenant_id,
        result.view,
        names,
        label_of,
        score=result.score,
        episode_ids=result.episode_ids,
    )
