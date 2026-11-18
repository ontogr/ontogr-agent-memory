"""Shared fakes and builders for resolution tests."""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime

from ontogr_mem.data_models.entities import Entity
from ontogr_mem.llm import RoleModels
from tests.support.scripted_model import (
    Scripted,
    ScriptedCall,
    scripted_model,
    scripted_role_models,
)
from tests.support.type_world import WORLD


FIXED_TIME = datetime(2026, 10, 21, tzinfo=UTC)

_EXTRA_TYPE_IDS: dict[str, int] = {}


def type_id_for(label: str) -> int:
    """Return a stable type id for a label, extending WORLD for extra labels."""
    if label in WORLD.ids:
        return WORLD.ids[label]
    if label not in _EXTRA_TYPE_IDS:
        _EXTRA_TYPE_IDS[label] = 1000 + len(_EXTRA_TYPE_IDS)
    return _EXTRA_TYPE_IDS[label]


def entity(
    entity_id: str,
    name: str,
    type_label: str = "Company",
    summary: str = "",
    ontology_type_id: int | None = None,
) -> Entity:
    """Build an entity with a fixed clock and a stable type id."""
    return Entity(
        entity_id=entity_id,
        type_label=type_label,
        ontology_type_id=(
            type_id_for(type_label) if ontology_type_id is None else ontology_type_id
        ),
        canonical_name=name,
        strict_key=name.casefold(),
        loose_key=name.casefold(),
        summary=summary,
        created_at=FIXED_TIME,
        updated_at=FIXED_TIME,
    )


_PAIR = re.compile(
    r'<pair n=(\d+)>\nA: name=("(?:[^"\\]|\\.)*").*?\nB: name=("(?:[^"\\]|\\.)*")',
    re.DOTALL,
)


def verifier_models(
    answer: Callable[[ScriptedCall], object],
) -> tuple[RoleModels, Scripted]:
    """Build a resolution model that records its typed requests."""
    model, scripted = scripted_model(answer=answer)
    return scripted_role_models(resolution=model), scripted


def oracle_models(
    decide: Callable[[str, str], bool | None],
) -> tuple[RoleModels, Scripted]:
    """Answer each shown pair from a ``(name_a, name_b) -> bool`` rule."""

    def answer(call: ScriptedCall) -> object:
        votes = []
        for match in _PAIR.finditer(call.prompt):
            vote = decide(json.loads(match.group(2)), json.loads(match.group(3)))
            if vote is not None:
                votes.append(
                    {
                        "id": int(match.group(1)),
                        "vote": "YES" if vote else "NO",
                        "rule": "test",
                    }
                )
        return {"pairs": votes}

    return verifier_models(answer)
