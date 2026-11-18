"""Compare two-pass and combined extraction on a caller-provided JSONL file."""

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ontogr_mem.combined_extraction.extractor import CombinedExtractor
from ontogr_mem.data_models import Actor, Episode, Kind, Role, Source
from ontogr_mem.entity_extraction import EntityExtraction
from ontogr_mem.entity_extraction.extractors import LLMExtractor, RecordMapper
from ontogr_mem.fact_extraction import FactExtraction, InMemoryFactStore
from ontogr_mem.fact_extraction.extractor import LLMFactExtractor
from ontogr_mem.llm import LLMConfig, ModelRole, RoleModels, build_models
from ontogr_mem.ontology.declare import build_target
from ontogr_mem.ontology.default import default_ontology
from ontogr_mem.ontology.models import Ontology
from ontogr_mem.ontology.projection import project
from ontogr_mem.ontology.stamp import stamp_types as _stamp_types


COMPARISON_KINDS = frozenset({Kind.MESSAGE, Kind.DOCUMENT})
"""Episode kinds whose text both extraction paths can process."""


@dataclass(frozen=True, slots=True)
class InputEpisode:
    """An episode row loaded from JSONL."""

    id: str
    kind: Kind
    actor: str
    text: str
    occurred_at: datetime


def load_episodes(path: str | Path) -> list[InputEpisode]:
    """Load JSONL episodes, applying defaults based on physical line numbers.

    Only message and document rows are accepted because both extraction paths
    process text content.
    """
    episodes = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError("text must be a string")
            kind = Kind(row.get("kind", "message"))
            if kind not in COMPARISON_KINDS:
                raise ValueError(f"unsupported kind {kind.value!r} for comparison")
            occurred_at = datetime.fromisoformat(
                row.get("occurred_at", "2026-01-01T12:00:00+00:00")
            )
            if occurred_at.tzinfo is None:
                raise ValueError("occurred_at must include a timezone")
            episodes.append(
                InputEpisode(
                    str(row.get("id", f"ep{line_number}")),
                    kind,
                    str(row.get("actor", "user")),
                    row["text"],
                    occurred_at,
                )
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"invalid episode at line {line_number}: {exc}") from exc
    return episodes


def validate_output_path(
    path: str | Path, repository_root: str | Path | None = None
) -> Path:
    """Resolve an output path and reject paths inside the source repository."""
    destination = Path(path).expanduser().resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root
        else Path(__file__).resolve().parents[2]
    )
    if destination == root or root in destination.parents:
        raise ValueError("output path must be outside the repository")
    return destination


def _episode(row: InputEpisode) -> Episode:
    return Episode(
        id=row.id,
        kind=row.kind,
        occurred_at=row.occurred_at,
        ingested_at=row.occurred_at,
        source=Source("eval", row.id, "eval"),
        actor=Actor(row.actor, Role.USER, row.actor),
        content=row.text,
        content_hash=hashlib.sha256(row.text.encode()).hexdigest(),
    )


async def _run_episode(
    episode: Episode, mode: str, models: RoleModels, projection: Any
) -> dict[str, Any]:
    """Run one episode through the requested extraction structure."""
    started = time.perf_counter()
    if mode == "combined":
        entities, facts = await CombinedExtractor(models).extract(
            episode, projection.view, None
        )
        results = [entities, facts]
        entity_items = entities.entities
        fact_items = facts.facts
    else:
        entity_report = await EntityExtraction(
            [LLMExtractor(models)], RecordMapper({}), projection.view.catalog
        ).extract([episode])
        entity_result = entity_report.results[0]
        stamped = _stamp_types(entity_result.entities, projection)
        fact_report = await FactExtraction(
            LLMFactExtractor(models), InMemoryFactStore(), projection.view
        ).extract([episode], {episode.id: stamped})
        fact_result = fact_report.results[0]
        results = [entity_result, fact_result]
        entity_items = stamped
        fact_items = fact_result.facts
    usage = [getattr(result, "usage", None) for result in results]
    buckets = [bucket for item in usage if item for bucket in item.buckets.values()]
    names_by_id = {entity.local_id: entity.name for entity in entity_items}
    fact_names = {
        names_by_id[ref]
        for fact in fact_items
        for ref in fact.entity_refs
        if ref in names_by_id
    }
    names = set(names_by_id.values())
    return {
        "seconds": time.perf_counter() - started,
        "requests": sum(b.calls for b in buckets),
        "input_tokens": sum(b.input_tokens for b in buckets),
        "output_tokens": sum(b.output_tokens for b in buckets),
        "entity_count": len(entity_items),
        "bound_fact_count": len(fact_items),
        "relation_fact_count": sum(
            fact.relation_type_id is not None for fact in fact_items
        ),
        "dropped_fact_count": len(getattr(results[-1], "dropped", [])),
        "statuses": [result.status.value for result in results],
        "orphaned_entities": sorted(names - fact_names),
        "entity_names": sorted(names),
        "fact_quotes": sorted(fact.evidence_quote.casefold() for fact in fact_items),
    }


async def run(
    episodes: list[InputEpisode], out: Path, models: RoleModels
) -> dict[str, Any]:
    """Run sequential mode comparisons and write the detailed report.

    Args:
        episodes: Input episodes to compare.
        out: Destination for the JSON report.
        models: Shared model pool used by both extraction paths.

    Returns:
        The report written to `out`.

    Raises:
        SystemExit: If either extraction model is not configured.
    """
    required = (ModelRole.ENTITY_EXTRACTION, ModelRole.FACT_EXTRACTION)
    missing = [role.value for role in required if not models.configured(role)]
    if missing:
        raise SystemExit(f"configure extraction models: {', '.join(missing)}")
    ontology = build_target(Ontology.genesis(), default_ontology())
    projection = project(ontology)
    output: dict[str, Any] = {"episodes": []}
    totals = {
        mode: {"requests": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
        for mode in ("two_pass", "combined")
    }
    for row in episodes:
        episode = _episode(row)
        modes = {
            mode: await _run_episode(episode, mode, models, projection)
            for mode in ("two_pass", "combined")
        }
        for mode, result in modes.items():
            for key in totals[mode]:
                totals[mode][key] += result[key]
        a, b = (set(modes[m]["entity_names"]) for m in ("two_pass", "combined"))
        qa, qb = (set(modes[m]["fact_quotes"]) for m in ("two_pass", "combined"))
        output["episodes"].append(
            {
                "id": row.id,
                "modes": modes,
                "overlap": {
                    "entity_name_jaccard": len(a & b) / len(a | b) if a | b else 1.0,
                    "fact_quote_jaccard": len(qa & qb) / len(qa | qb)
                    if qa | qb
                    else 1.0,
                },
            }
        )
    output["totals"] = totals
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print("mode       requests  input  output  seconds")
    for mode, values in totals.items():
        print(
            f"{mode:<10} {values['requests']:>8}  {values['input_tokens']:>5}  "
            f"{values['output_tokens']:>6}  {values['seconds']:.2f}"
        )
    return output


async def _run_live(episodes: list[InputEpisode], out: Path) -> dict[str, Any]:
    """Run the comparison with the models configured in the environment."""
    models = build_models(LLMConfig())
    try:
        return await run(episodes, out, models)
    finally:
        await models.aclose()


def main() -> None:
    """Run the real endpoint comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    destination = validate_output_path(args.out)
    asyncio.run(_run_live(load_episodes(args.episodes), destination))


if __name__ == "__main__":
    main()
