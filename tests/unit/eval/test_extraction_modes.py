"""Input, output and comparison rules for the extraction mode evaluation."""

import json
import re

import pytest

from ontogr_mem.data_models import Kind
from ontogr_mem.llm import RoleModels
from tests.eval import extraction_modes
from tests.eval.extraction_modes import load_episodes, validate_output_path
from tests.support.scripted_model import (
    Scripted,
    ScriptedCall,
    scripted_model,
    scripted_role_models,
)


TEXT = "Ana works at Acme and uses Postgres."
ENTITIES = (("Ana", "Person"), ("Acme", "Organization"), ("Postgres", "Object"))


def _block(prompt: str, tag: str) -> str:
    """Return the body of one fenced prompt block."""
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", prompt, re.DOTALL)
    assert match is not None, f"missing {tag} block"
    return match.group(1)


def _type_id(prompt: str, tag: str, label: str) -> int:
    """Return the dense id the prompt assigns to a type label."""
    items = json.loads(_block(prompt, tag))
    return next(item["id"] for item in items if item["label"] == label)


def _fact(prompt: str) -> dict[str, object]:
    """Return one grounded fact that references the first two entities."""
    return {
        "text": "Ana works at Acme",
        "entity_refs": ["e0", "e1"],
        "relation_type_id": _type_id(prompt, "relation_types", "WORKS_AT"),
        "relation_label": "WORKS_AT",
        "fact_kind": "state",
        "polarity": "affirmed",
        "modality": "actual",
        "evidence_quote": "Ana works at Acme",
        "confidence": 1.0,
    }


def _answer(call: ScriptedCall) -> object:
    """Answer each extraction prompt with the same entities and one fact."""
    prompt = call.prompt
    if prompt.startswith("Task: extract the entities and the facts"):
        return {
            "entities": [
                {
                    "ref": f"e{index}",
                    "name": name,
                    "type_id": _type_id(prompt, "entity_types", label),
                    "type_label": label,
                    "description": name,
                    "confidence": 1.0,
                }
                for index, (name, label) in enumerate(ENTITIES)
            ],
            "facts": [_fact(prompt)],
        }
    if prompt.startswith("Task: list the entities"):
        return {
            "entities": [
                {
                    "name": name,
                    "type_id": _type_id(prompt, "entity_types", label),
                    "confidence": 1.0,
                }
                for name, label in ENTITIES
            ]
        }
    return {"facts": [_fact(prompt)]}


def _scripted_models() -> tuple[RoleModels, Scripted]:
    """Build a deterministic model pool and its call recorder."""
    model, scripted = scripted_model(answer=_answer)
    return scripted_role_models(model), scripted


def test_load_episodes_skips_blank_lines_and_applies_defaults(tmp_path):
    """Episode defaults follow the physical JSONL line number."""
    source = tmp_path / "episodes.jsonl"
    source.write_text('\n{"text":"hello"}\n{"text":"bye","id":"custom"}\n')

    episodes = load_episodes(source)

    assert len(episodes) == 2
    assert episodes[0].id == "ep2"
    assert episodes[0].kind == "message"
    assert episodes[0].actor == "user"
    assert episodes[1].id == "custom"


def test_load_episodes_accepts_documents(tmp_path):
    """Document rows are valid comparison input."""
    source = tmp_path / "episodes.jsonl"
    source.write_text('{"text":"hello","kind":"document"}\n')

    assert load_episodes(source)[0].kind is Kind.DOCUMENT


@pytest.mark.parametrize("kind", ["tool_output", "event", "record"])
def test_load_episodes_rejects_unsupported_kinds(tmp_path, kind):
    """The comparison loader accepts only message and document rows."""
    source = tmp_path / "episodes.jsonl"
    source.write_text(json.dumps({"text": "hello", "kind": kind}) + "\n")

    with pytest.raises(ValueError, match="unsupported kind"):
        load_episodes(source)


@pytest.mark.parametrize("payload", ["{}", '{"text": 2}', "[]"])
def test_load_episodes_rejects_invalid_rows(tmp_path, payload):
    """Malformed rows fail with their source line number."""
    source = tmp_path / "episodes.jsonl"
    source.write_text(payload + "\n")

    with pytest.raises(ValueError, match="line 1"):
        load_episodes(source)


def test_output_path_must_be_outside_repository(tmp_path):
    """Results cannot be written into the source checkout."""
    with pytest.raises(ValueError, match="outside the repository"):
        validate_output_path(tmp_path / "nested/result.json", repository_root=tmp_path)

    destination = tmp_path.parent / "result.json"
    assert validate_output_path(destination, repository_root=tmp_path) == destination


async def test_run_compares_modes_on_fact_bearing_input(tmp_path):
    """Both extraction paths produce a report that marks unreferenced entities."""
    source = tmp_path / "episodes.jsonl"
    source.write_text(json.dumps({"text": TEXT, "id": "ep1", "actor": "Ana"}) + "\n")
    out = tmp_path / "report.json"
    models, scripted = _scripted_models()

    report = await extraction_modes.run(load_episodes(source), out, models)

    assert json.loads(out.read_text()) == report
    compared = report["episodes"][0]["modes"]
    for mode in ("two_pass", "combined"):
        result = compared[mode]
        assert result["statuses"] == ["ok", "ok"]
        assert result["bound_fact_count"] == 1
        assert result["relation_fact_count"] == 1
        assert result["dropped_fact_count"] == 0
        assert result["entity_names"] == ["Acme", "Ana", "Postgres"]
        assert result["orphaned_entities"] == ["Postgres"]
    assert report["episodes"][0]["overlap"] == {
        "entity_name_jaccard": 1.0,
        "fact_quote_jaccard": 1.0,
    }
    assert report["totals"]["two_pass"]["requests"] == 2
    assert report["totals"]["combined"]["requests"] == 1
    assert len(scripted.calls) == 3
