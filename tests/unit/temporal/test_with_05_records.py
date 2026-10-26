"""Real record facts from the extraction layer through the adapter."""

from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models import (
    Actor,
    EntityType,
    Episode,
    Kind,
    Method,
    RelationType,
    Role,
    Source,
    TypeCatalog,
    TypeView,
)
from ontogr_mem.data_models.extraction import ExtractedEntity
from ontogr_mem.data_models.facts import FactKind, Precision, TypedFact
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.fact_extraction import FactMapping, FactTemplate, RecordFactMapper
from ontogr_mem.temporal.adapter import SkipReason, observations_from_facts
from ontogr_mem.temporal.cardinality import Cardinality
from ontogr_mem.temporal.policy import from_type_view
from ontogr_mem.temporal.reconcile import derive


CATALOG = TypeCatalog(
    [EntityType(0, "Person"), EntityType(1, "City")],
    [RelationType(0, "LIVES_IN", "", frozenset({0}), frozenset({1}))],
)
VIEW = TypeView(CATALOG)
STATE = FactTemplate(
    "{person} lives in {city}",
    ("person", "city"),
    entity_fields=("person", "city"),
    fact_kind=FactKind.STATE,
    relation_type_label="LIVES_IN",
)
EVENT = FactTemplate("{person} visited {city}", ("person", "city"))


class NameResolver:
    """Local refs e0 and e1 are the person and the city of the episode."""

    def __init__(self) -> None:
        """Start with no names."""
        self.names: dict[tuple[str, str], str] = {}

    async def entity_id_for_ref(self, episode_id: str, local_id: str) -> str | None:
        """Return the stable id registered for the local ref."""
        return self.names.get((episode_id, local_id))


def record(
    resolver: NameResolver,
    number: int,
    month: int,
    city: str,
    template: FactTemplate,
) -> tuple[Episode, list[TypedFact]]:
    """Build a CRM record, map it with the real mapper and type its facts."""
    when = datetime(2024, month, 1, tzinfo=UTC)
    episode = Episode(
        id=f"ep{number}",
        kind=Kind.RECORD,
        occurred_at=when,
        ingested_at=when,
        source=Source("crm", f"r{number}", "crm"),
        actor=Actor("svc", Role.SERVICE),
        content={"person": "Ana", "city": city},
        content_hash=f"h{number}",
        metadata={"table": "people"},
    )
    entities = [
        ExtractedEntity("e0", "Ana", 0, 1.0, Method.RECORD, episode.id),
        ExtractedEntity("e1", city, 1, 1.0, Method.RECORD, episode.id),
    ]
    resolver.names[(episode.id, "e0")] = "ana"
    resolver.names[(episode.id, "e1")] = city.lower()
    mapper = RecordFactMapper({"crm/people": FactMapping((template,))})
    result = mapper.map(episode, entities, VIEW)
    facts = [
        TypedFact(fact, relation_type=0, catalog_version=VIEW.version)
        for fact in result.facts
    ]
    return episode, facts


class TestRecordFacts:
    """Record templates exercised through the real mapper and the adapter."""

    @pytest.mark.parametrize("order", [(1, 2), (2, 1)])
    async def test_record_updates_become_a_timeline_in_any_arrival_order(self, order):
        """Berlin in January and Paris in June give Berlin then Paris."""
        resolver = NameResolver()
        policy = from_type_view(VIEW, (0,))
        rows = {
            1: record(resolver, 1, 1, "Berlin", STATE),
            2: record(resolver, 2, 6, "Paris", STATE),
        }
        observations: list[Observation] = []
        slots: set[str] = set()
        for number in order:
            episode, facts = rows[number]
            converted, skipped = await observations_from_facts(
                facts, episode, resolver, policy
            )
            assert not skipped
            assert all(fact.relation_type == 0 for fact in facts)
            observations.extend(converted)
            slots.update(observation.slot_key for observation in converted)

        assert len(slots) == 1
        derived = derive(slots.pop(), observations, Cardinality.ONE)
        assert sorted(
            (run.value_id, run.valid_from, run.valid_to) for run in derived.runs
        ) == [
            (
                "berlin",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 6, 1, tzinfo=UTC),
            ),
            ("paris", datetime(2024, 6, 1, tzinfo=UTC), None),
        ]
        assert {run.valid_from_precision for run in derived.runs} == {Precision.TIME}

    async def test_default_event_templates_do_not_reach_reconcile(self):
        """An event template is skipped and yields no observation."""
        resolver = NameResolver()
        policy = from_type_view(VIEW, (0,))
        episode, facts = record(resolver, 1, 1, "Berlin", EVENT)
        converted, skipped = await observations_from_facts(
            facts, episode, resolver, policy
        )
        assert converted == []
        assert [item.reason for item in skipped] == [SkipReason.NOT_STATE]
