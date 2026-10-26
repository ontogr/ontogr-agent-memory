"""Tests for the record mapper."""

from ontogr_mem.data_models import DropReason, Method, Status
from ontogr_mem.entity_extraction.extractors import (
    FieldEntity,
    RecordMapper,
    RecordMapping,
)
from tests.support.episodes import make_episode
from tests.support.extraction import catalog


MAPPING = RecordMapping(
    (
        FieldEntity("customer", "Person", ("tier",)),
        FieldEntity("vendor", "Company"),
        FieldEntity("db", "Warehouse"),
    )
)


class TestRecordMapper:
    """Behavior of RecordMapper."""

    def test_maps_fields_to_entities_and_drops_what_cannot_be_mapped(self):
        """A missing field and an unknown type are dropped with a reason."""
        episode = make_episode(
            "r1", content={"customer": "Ana", "tier": "gold", "db": "wh1"}
        )
        mapper = RecordMapper({"chat": MAPPING})

        result = mapper.map(episode, catalog())

        assert result.status is Status.OK
        (entity,) = result.entities
        assert (entity.name, entity.type_id, entity.confidence, entity.method) == (
            "Ana",
            0,
            1.0,
            Method.RECORD,
        )
        assert entity.attributes == {"tier": "gold"}
        assert [d.reason for d in result.dropped] == [
            DropReason.MISSING_FIELD,
            DropReason.TYPE_OUT_OF_RANGE,
        ]

    def test_table_specific_mapping_wins_over_the_system_mapping(self):
        """`system/table` is looked up before `system`."""
        other = RecordMapping((FieldEntity("vendor", "Company"),))
        mapper = RecordMapper({"chat": MAPPING, "chat/orders": other})
        episode = make_episode(
            "r1", content={"vendor": "Acme"}, metadata={"table": "orders"}
        )

        result = mapper.map(episode, catalog())

        assert [e.name for e in result.entities] == ["Acme"]

    def test_episode_without_a_mapping_is_skipped(self):
        """No mapping means skipped, with the reason."""
        result = RecordMapper({}).map(make_episode("r1", content={"a": 1}), catalog())

        assert (result.status, result.detail) == (Status.SKIPPED, "no record mapping")
