"""Builders and doubles for fact extraction tests."""

import json
from datetime import UTC, datetime
from typing import Any

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
)
from ontogr_mem.data_models.extraction import ExtractedEntity
from ontogr_mem.llm import LLMGate
from tests.support.llm import ScriptedLLM


T0 = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
TEXT = "I switched to Postgres in March 2024 because Acme needs transactions."


def fact_catalog() -> TypeCatalog:
    """Return Person (0), Database (1), Company (2), USES (0) and WORKS_FOR (1)."""
    return TypeCatalog(
        [
            EntityType(0, "Person"),
            EntityType(1, "Database"),
            EntityType(2, "Company"),
        ],
        [
            RelationType(0, "USES", "uses a tool", frozenset({0, 2}), frozenset({1})),
            RelationType(1, "WORKS_FOR", "", frozenset({0}), frozenset({2})),
        ],
    )


def make_episode(
    text: str = TEXT,
    *,
    kind: Kind | str = Kind.MESSAGE,
    episode_id: str = "ep1",
    content: str | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    source_system: str = "chat",
) -> Episode:
    """Build an episode with an exact occurrence time."""
    return Episode(
        id=episode_id,
        kind=kind,  # type: ignore[arg-type]
        occurred_at=T0,
        ingested_at=T0,
        source=Source(source_system, "m1", source_system),
        actor=Actor("u1", Role.USER, "Ana"),
        content=text if content is None else content,
        content_hash="h",
        metadata=dict(metadata or {}),
    )


def ent(local_id: str, name: str, type_id: int) -> ExtractedEntity:
    """Build a candidate entity with the type label of its dense id."""
    labels = {0: "Person", 1: "Database", 2: "Company"}
    return ExtractedEntity(
        local_id=local_id,
        name=name,
        type_id=type_id,
        type_label=labels[type_id],
        confidence=0.9,
        method=Method.GLINER,
        episode_id="ep1",
    )


ANA = ent("e0", "Ana", 0)
PG = ent("e1", "Postgres", 1)
ACME = ent("e2", "Acme", 2)


def fact(**kw: object) -> dict[str, object]:
    """Build one fact object as a model would write it."""
    base: dict[str, object] = {
        "text": "Ana switched to Postgres",
        "entity_refs": ["e0", "e1"],
        "relation_type_id": None,
        "relation_label": None,
        "fact_kind": "event",
        "polarity": "affirmed",
        "modality": "actual",
        "evidence_quote": "I switched to Postgres",
    }
    return base | kw


def reply(*facts: dict[str, object]) -> str:
    """Build the JSON reply containing `facts`."""
    return json.dumps({"facts": list(facts)})


def gate(llm: ScriptedLLM, max_input_tokens: int = 20_000) -> LLMGate:
    """Build a model gate around a scripted client."""
    return LLMGate(llm, max_input_tokens=max_input_tokens)
