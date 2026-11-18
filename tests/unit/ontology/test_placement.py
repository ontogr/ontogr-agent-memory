"""Check relation labels are placed only when their endpoint types fit."""

from dataclasses import replace

from ontogr_mem.data_models import ExtractedEntity, Method
from ontogr_mem.data_models.facts import (
    ExtractedFact,
    FactKind,
    Modality,
    Polarity,
    TimeBasis,
)
from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.models import AddAlias, Kind
from ontogr_mem.ontology.placement import place_relations
from ontogr_mem.ontology.projection import project
from tests.support.ontology import base, cs


def _fact(label: str | None, relation_type_id: int | None = None) -> ExtractedFact:
    return ExtractedFact(
        fact_id="f1",
        text="Ana works at Acme",
        entity_refs=("s", "t"),
        relation_type_id=relation_type_id,
        relation_label=label,
        fact_kind=FactKind.STATE,
        polarity=Polarity.AFFIRMED,
        modality=Modality.ACTUAL,
        asserted_by=None,
        valid_from=None,
        valid_to=None,
        precision=None,
        time_expression=None,
        time_basis=TimeBasis.NONE,
        evidence_quote="Ana works at Acme",
        evidence_span=None,
        confidence=1.0,
        method=Method.LLM,
        episode_id="episode",
    )


def _entities(source_type: int, target_type: int) -> tuple[ExtractedEntity, ...]:
    return (
        ExtractedEntity("s", "Ana", source_type, 1.0, Method.LLM, "episode"),
        ExtractedEntity("t", "Acme", target_type, 1.0, Method.LLM, "episode"),
    )


class TestPlaceRelations:
    """Check label resolution and endpoint validation for relation placement."""

    def test_label_placement_matches_alias_and_close_spelling(self):
        """Place labels that resolve exactly, by alias, or by close spelling."""
        catalog = base()
        catalog = apply_changeset(
            catalog,
            cs(
                catalog,
                AddAlias(kind=Kind.RELATION, target="works_at", alias="employer"),
            ),
        )
        projection = project(catalog)
        entities = _entities(0, 2)
        placed = place_relations(
            [_fact("works_at"), _fact("employer"), _fact("work at")],
            entities,
            projection,
        )
        assert [fact.relation_type_id for fact in placed] == [0, 0, 0]

    def test_unknown_label_stays_unplaced(self):
        """Keep unknown relation words on an unplaced fact."""
        placed = place_relations([_fact("admires")], _entities(0, 2), project(base()))
        assert placed[0].relation_type_id is None
        assert placed[0].relation_label == "admires"

    def test_pattern_violation_keeps_label_and_clears_type(self):
        """Keep the label when endpoint types violate its pattern."""
        placed = place_relations([_fact("works_at")], _entities(1, 1), project(base()))
        assert placed[0].relation_type_id is None
        assert placed[0].relation_label == "works_at"

    def test_unknown_endpoint_does_not_violate_pattern(self):
        """Allow placement when an endpoint type is Unknown."""
        placed = place_relations([_fact("works_at")], _entities(-1, 2), project(base()))
        assert placed[0].relation_type_id == 0

    def test_self_relation_stays_unplaced(self):
        """Keep a self relation unplaced even when its label resolves."""
        fact = _fact("works_at")
        fact = replace(fact, entity_refs=("s", "s"))
        placed = place_relations([fact], _entities(0, 2), project(base()))
        assert placed[0].relation_type_id is None
