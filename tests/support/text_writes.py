"""Scripted model answers for text writes, keyed by the episode text.

The entity pass and the fact pass share one scripted model. The helper tells the two
passes apart by their prompts and looks type ids up by label in the prompt blocks, so
a test names catalog types and relations and never dense ids.
"""

import json
import re
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

from ontogr_mem.client import Memory
from ontogr_mem.llm import RoleModels
from ontogr_mem.llm.config import ModelRole
from ontogr_mem.llm.prompt import fence
from ontogr_mem.ontology.loader import OntologySource
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.service.bootstrap import ServiceSettings
from tests.support.embedder import MockEmbedder
from tests.support.scripted_model import (
    Scripted,
    ScriptedCall,
    scripted_model,
    scripted_role_models,
)


@dataclass(frozen=True, slots=True)
class ScriptedEntity:
    """An entity that the scripted entity pass returns."""

    name: str
    type_label: str | None = None
    proposed_label: str | None = None
    attributes: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ScriptedFact:
    """A fact that the scripted fact pass returns."""

    text: str
    subject: int
    object: int
    relation_label: str
    evidence_quote: str
    fact_kind: str = "state"
    modality: str = "actual"
    polarity: str = "affirmed"
    valid_from: str | None = None
    time_expression: str | None = None


Script = Mapping[str, tuple[Sequence[ScriptedEntity], Sequence[ScriptedFact]]]


def _block(prompt: str, tag: str) -> str:
    """Return the body of one fenced prompt block."""
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", prompt, re.DOTALL)
    if match is None:
        raise AssertionError(f"missing {tag} block")
    return match.group(1)


def _id_of(prompt: str, tag: str, label: str) -> int | None:
    """Return the prompt id of the type with this label, or None."""
    items = json.loads(_block(prompt, tag))
    return next(
        (item["id"] for item in items if item["label"].casefold() == label.casefold()),
        None,
    )


def _entity_item(prompt: str, entity: ScriptedEntity) -> dict[str, object]:
    """Return one entity of the scripted entity pass reply."""
    type_id = -1
    if entity.type_label is not None:
        found = _id_of(prompt, "entity_types", entity.type_label)
        if found is None:
            raise AssertionError(f"no entity type {entity.type_label}")
        type_id = found
    item: dict[str, object] = {
        "name": entity.name,
        "type_id": type_id,
        "confidence": 1.0,
    }
    if entity.proposed_label is not None:
        item["type_label"] = entity.proposed_label
    if entity.attributes is not None:
        item["attributes"] = dict(entity.attributes)
    return item


def _fact_item(prompt: str, fact: ScriptedFact) -> dict[str, object]:
    """Return one fact of the scripted fact pass reply."""
    return {
        "text": fact.text,
        "entity_refs": [f"e{fact.subject}", f"e{fact.object}"],
        "relation_type_id": (
            _id_of(prompt, "relation_types", fact.relation_label)
            if fact.relation_label
            else None
        ),
        "relation_label": fact.relation_label,
        "fact_kind": fact.fact_kind,
        "polarity": fact.polarity,
        "modality": fact.modality,
        "valid_from": fact.valid_from,
        "valid_to": None,
        "precision": "day" if fact.valid_from is not None else None,
        "time_expression": fact.time_expression,
        "evidence_quote": fact.evidence_quote,
        "confidence": 1.0,
    }


COMBINED_MARKER = "Task: extract the entities and the facts"


def _answer(
    script: Script,
    closures: Mapping[str, Sequence[str]],
    closure_error: Exception | None,
) -> Callable[[ScriptedCall], object]:
    """Return the answer function that follows ``script`` for all prompt shapes."""
    fenced: dict[str, tuple[Sequence[ScriptedEntity], Sequence[ScriptedFact]]] = {}
    for text, value in script.items():
        key = _block(fence("episode_text", text), "episode_text")
        if key in fenced:
            raise AssertionError(f"two scripted episode texts share one prompt: {key}")
        fenced[key] = value

    def answer(call: ScriptedCall) -> object:
        prompt = call.prompt
        if "Task: list the existing facts that the new fact contradicts" in prompt:
            if closure_error is not None:
                raise closure_error
            new_fact = _block(prompt, "new_fact")
            statement = new_fact.strip()
            wanted = set(closures.get(statement, ()))
            existing = json.loads(_block(prompt, "existing_facts"))
            ids = [item["id"] for item in existing if item["text"] in wanted]
            return {"contradicted": ids}
        entities, facts = fenced[_block(prompt, "episode_text")]
        if prompt.startswith(COMBINED_MARKER):
            return {
                "entities": [
                    {**_entity_item(prompt, entity), "ref": f"e{index}"}
                    for index, entity in enumerate(entities)
                ],
                "facts": [_fact_item(prompt, fact) for fact in facts],
            }
        if "Task: list the entities" in prompt:
            return {"entities": [_entity_item(prompt, entity) for entity in entities]}
        return {"facts": [_fact_item(prompt, fact) for fact in facts]}

    return answer


def counted_text_models(
    script: Script,
    *,
    closures: Mapping[str, Sequence[str]] | None = None,
    closure_error: Exception | None = None,
    judge: bool = True,
) -> tuple[RoleModels, Scripted]:
    """Return a scripted model pool and the recorder of its calls."""
    model, scripted = scripted_model(
        answer=_answer(script, closures or {}, closure_error)
    )
    if judge:
        return scripted_role_models(model), scripted
    return scripted_role_models(
        **{role.value: model for role in ModelRole if role is not ModelRole.JUDGE}
    ), scripted


def text_models(
    script: Script,
    *,
    closures: Mapping[str, Sequence[str]] | None = None,
    closure_error: Exception | None = None,
    judge: bool = True,
) -> RoleModels:
    """Return a model pool whose extraction answers follow ``script``."""
    return counted_text_models(
        script, closures=closures, closure_error=closure_error, judge=judge
    )[0]


@asynccontextmanager
async def open_memory(
    tenant: str,
    models: RoleModels,
    ontology: OntologySource | None = None,
) -> AsyncGenerator[Memory]:
    """Open a Memory client on FalkorDB with ephemeral state."""
    async with Memory.open(
        tenant=tenant,
        ontology=ontology,
        service_settings=ServiceSettings(backend="falkordb", _env_file=None),
        state_settings=StateSettings(
            backend="memory", allow_ephemeral=True, _env_file=None
        ),
        models=models,
        embedder=MockEmbedder(),
    ) as memory:
        yield memory
