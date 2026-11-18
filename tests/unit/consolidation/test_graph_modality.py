"""Behavior checks for modality filtering in the consolidation graph."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

from ontogr_mem.consolidation.graph import build_entity_graph
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.data_models.entities import Entity
from ontogr_mem.data_models.facts import FactKind, Modality
from ontogr_mem.temporal.query import FactView
from tests.unit.consolidation.stage_fakes import (
    MemoryResolution,
    MemoryStore,
    MemoryTemporal,
)


async def test_entity_graph_keeps_actual_facts_and_events_only() -> None:
    """Keep actual state and event facts while excluding non-actual plans."""
    now = datetime(2030, 1, 1, tzinfo=UTC)
    entities = [
        Entity(
            entity_id="e1",
            canonical_name="Ada",
            type_label="Person",
            ontology_type_id=0,
            strict_key="ada",
            loose_key="ada",
            created_at=now,
            updated_at=now,
        ),
        Entity(
            entity_id="e2",
            canonical_name="Lin",
            type_label="Person",
            ontology_type_id=0,
            strict_key="lin",
            loose_key="lin",
            created_at=now,
            updated_at=now,
        ),
    ]

    def fact(
        fact_id: str, modality: Modality, kind: FactKind = FactKind.STATE
    ) -> FactView:
        return cast(
            FactView,
            SimpleNamespace(
                fact_id=fact_id,
                subject_id="e1",
                object_id="e2",
                modality=modality,
                fact_kind=kind,
            ),
        )

    temporal = MemoryTemporal(
        (
            fact("actual", Modality.ACTUAL),
            fact("event", Modality.ACTUAL, FactKind.EVENT),
            fact("planned", Modality.PLANNED),
            fact("desired", Modality.DESIRED),
        )
    )
    resolution = MemoryResolution(*entities)
    ctx = cast(
        StageContext,
        SimpleNamespace(
            temporal=temporal,
            resolution=resolution,
            store=MemoryStore(),
            entity_graph=None,
        ),
    )

    graph = await build_entity_graph(ctx)

    assert {item.fact_id for item in graph.facts} == {"actual", "event"}
