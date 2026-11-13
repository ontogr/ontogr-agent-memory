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


FIXED_TIME = datetime(2026, 10, 21, tzinfo=UTC)


def entity(
    entity_id: str,
    name: str,
    type_label: str = "Company",
    summary: str = "",
) -> Entity:
    """Build an entity with a fixed clock."""
    return Entity(
        entity_id=entity_id,
        type_label=type_label,
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
