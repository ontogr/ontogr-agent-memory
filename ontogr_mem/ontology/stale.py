"""Which episodes may give a different result after a catalog change."""

from collections.abc import Iterable

from ontogr_mem.ontology.diff import diff
from ontogr_mem.ontology.models import (
    AddAlias,
    AddAttribute,
    AddEntity,
    AddPattern,
    AddRelation,
    ChangeSet,
    Kind,
    Ontology,
    Unmatched,
    UnmatchedKind,
)
from ontogr_mem.ontology.names import normalize
from ontogr_mem.ontology.resolve import How, TypeResolver


def stale_episodes(
    old: Ontology,
    new: Ontology,
    unmatched: Iterable[Unmatched],
    known_episodes: Iterable[str],
) -> set[str]:
    """Return the episode ids to queue for re-extraction after a catalog change.

    If the diff only adds (types, aliases, attributes, patterns), an episode is stale
    when one of its stored unmatched rows now resolves differently or names something a
    new item covers. Any other change (rename, deprecation, merge, mode change) reports
    every known episode: over-reporting is safe, under-reporting is not. Nothing is
    queued here.

    Args:
        old: The catalog the episodes were last extracted under.
        new: The catalog after the change.
        unmatched: Stored unmatched rows, one set per episode.
        known_episodes: The episode ids to consider, as read from the graph.

    Returns:
        The episode ids that may give a different result under ``new``.
    """
    cs = diff(old, new)
    known = set(known_episodes)
    if not cs.ops:
        return set()
    if not cs.is_additive():
        return known
    hits = _covered(new, cs)
    return {
        u.episode_id
        for u in unmatched
        if u.episode_id in known and _row_matches(old, new, u, hits)
    }


def _covered(new: Ontology, cs: ChangeSet) -> dict[str, set[str]]:
    """Collect the normalized labels that the added ops cover, grouped by row kind."""
    ent, rel = set(), set()
    attrs, rel_ids = set(), set()
    for op in cs.ops:
        if isinstance(op, AddEntity):
            ent |= {normalize(op.label), *(normalize(a) for a in op.aliases)}
            attrs |= {normalize(a.name) for a in op.properties}
        elif isinstance(op, AddRelation):
            rel |= {normalize(op.label), *(normalize(a) for a in op.aliases)}
            rel_ids.add(normalize(op.label))
            attrs |= {normalize(a.name) for a in op.properties}
        elif isinstance(op, AddAlias):
            (ent if op.kind is Kind.ENTITY else rel).add(normalize(op.alias))
        elif isinstance(op, AddAttribute):
            attrs.add(normalize(op.attribute.name))
        elif isinstance(op, AddPattern):
            t = new.relation(op.relation) if isinstance(op.relation, int) else None
            name = t.label if t else str(op.relation)
            rel_ids.add(normalize(name))
    return {"entity": ent, "relation": rel, "attribute": attrs, "pattern": rel_ids}


def _resolution(
    cat: Ontology, kind: Kind, raw: str
) -> tuple[tuple[int | None, How | None], tuple[int, ...]]:
    """Resolve ``raw`` in ``cat`` and return a comparable summary of the outcome.

    The summary is the matched type id and how it matched, plus any ambiguous
    candidates.
    """
    res = TypeResolver(cat, kind).resolve(raw)
    m = res.match
    return ((m.type_id, m.how) if m else (None, None)), res.ambiguous


def _row_matches(
    old: Ontology, new: Ontology, u: Unmatched, hits: dict[str, set[str]]
) -> bool:
    """Return True if an unmatched row may give a different result under ``new``.

    Type rows (including fuzzy ones) stay valid as long as their label resolves the
    same way in ``old`` and ``new``. Attribute rows are stale when an added attribute
    has the same name. Pattern rows are stale when a relation gained a pattern or an
    entity type was added.
    """
    kinds = {
        UnmatchedKind.ENTITY_TYPE: Kind.ENTITY,
        UnmatchedKind.FUZZY_ENTITY: Kind.ENTITY,
        UnmatchedKind.RELATION_TYPE: Kind.RELATION,
        UnmatchedKind.FUZZY_RELATION: Kind.RELATION,
    }
    kind = kinds.get(u.kind)
    if kind is not None:
        return _resolution(old, kind, u.raw_label) != _resolution(
            new, kind, u.raw_label
        )
    if u.kind is UnmatchedKind.ATTRIBUTE:
        return u.normalized_label in hits["attribute"]
    return u.normalized_label in hits["pattern"] or bool(hits["entity"])
