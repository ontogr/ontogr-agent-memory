"""Formatting behavior for facts returned by the memory service."""

from datetime import UTC, datetime
from typing import cast

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.data_models.facts import FactKind, Modality, TimeBasis
from ontogr_mem.service.formatting import fact_out
from ontogr_mem.temporal.query import FactView


def test_unplaced_fact_uses_its_relation_words_and_exposes_kind() -> None:
    """Use stored relation words when a fact has no relation type."""
    now = datetime(2030, 1, 1, tzinfo=UTC)
    view = cast(
        FactView,
        type(
            "FactViewStub",
            (),
            {
                "fact_id": "f1",
                "relation_type_id": None,
                "relation_label": "works with",
                "statement": "",
                "subject_id": "e1",
                "object_id": "e2",
                "valid_from": now,
                "valid_to": None,
                "recorded_at": now,
                "expired_at": None,
                "status": FactStatus.ACTIVE,
                "disputed": False,
                "counterpart_ids": (),
                "evidence": (),
                "modality": Modality.DESIRED,
                "fact_kind": FactKind.STATE,
                "time_basis": TimeBasis.NONE,
            },
        )(),
    )

    fact = fact_out("t1", view, {"e1": "Ada", "e2": "Lin"}, lambda _: "")

    assert fact.relation == "works with"
    assert fact.statement == "Ada works with Lin"
    assert fact.modality == "desired"
    assert fact.kind == "state"
