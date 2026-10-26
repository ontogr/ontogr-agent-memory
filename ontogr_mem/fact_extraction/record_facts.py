"""No-LLM mapping for structured episodes (`Kind.RECORD` and `Kind.EVENT`).

Each declared template turns the fields of a structured payload into one
statement (confidence 1.0, no model call). The evidence of such a fact is the
list of field paths it was built from.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ontogr_mem.data_models import (
    Episode,
    ExtractedEntity,
    Method,
    Status,
    TypeView,
)
from ontogr_mem.data_models.facts import (
    ExtractedFact,
    FactKind,
    Modality,
    Polarity,
    Precision,
    TimeBasis,
)
from ontogr_mem.fact_extraction.binding import check_relation_type
from ontogr_mem.fact_extraction.ids import assign_ids
from ontogr_mem.fact_extraction.models import (
    FactDropped,
    FactDropReason,
    FactResult,
    SkipReason,
)
from ontogr_mem.fact_extraction.text import normalize_text
from ontogr_mem.fact_extraction.timeparse import parse_instant, truncate


_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


@dataclass(frozen=True, slots=True)
class FactTemplate:
    """One templated statement over a declared set of fields.

    `template` uses `{field}` placeholders; `fields` lists every field path it
    reads (dotted for nested payloads) and is the fact's evidence. Every listed
    field must be present and non-blank, else the template is dropped with
    `missing_field`. `entity_fields` name the fields whose values are candidate
    entities, in link order (source first). `time_field` names the field holding
    the fact's time; without one the record's `occurred_at` is the time.
    """

    template: str
    fields: tuple[str, ...]
    entity_fields: tuple[str, ...] = ()
    fact_kind: FactKind = FactKind.EVENT
    time_field: str | None = None
    relation_type_label: str | None = None


@dataclass(frozen=True, slots=True)
class FactMapping:
    """How one table or event type maps to facts."""

    templates: tuple[FactTemplate, ...]


def _get(data: Mapping[str, Any], path: str) -> Any:
    """Read a dotted path from nested mappings; None if any step is missing."""
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


class RecordFactMapper:
    """Renders facts from structured content without an LLM.

    A mapping is looked up by `"<source.system>/<metadata.table>"` and then by
    `source.system`. An episode with no mapping is skipped (no model call). A
    field's entity is the one candidate whose name equals the field value; none
    or several is recorded and the ref left out.
    """

    version = "record-fact-1"

    def __init__(self, mappings: Mapping[str, FactMapping]) -> None:
        """Register mappings by key."""
        self._mappings = dict(mappings)

    def _lookup(self, episode: Episode) -> FactMapping | None:
        """Return the most specific mapping: `system/table`, then `system`."""
        table = episode.metadata.get("table")
        key = f"{episode.source.system}/{table}"
        if table is not None and key in self._mappings:
            return self._mappings[key]
        return self._mappings.get(episode.source.system)

    def map(
        self,
        episode: Episode,
        entities: Sequence[ExtractedEntity],
        view: TypeView | None,
    ) -> FactResult:
        """Build the result for one structured episode."""
        result = FactResult(episode.id, Status.EMPTY)
        mapping = self._lookup(episode)
        data = episode.content
        if mapping is None or not isinstance(data, dict):
            result.status = Status.SKIPPED
            result.skip_reason = SkipReason.UNMAPPED_RECORD
            result.detail = (
                f"no fact mapping for {episode.source.system!r}"
                if mapping is None
                else "content is not a structured payload"
            )
            return result

        facts = [
            fact
            for spec in mapping.templates
            if (
                fact := self._fact(
                    episode, spec, data, entities=entities, view=view, result=result
                )
            )
        ]
        result.facts = assign_ids(episode.id, facts)
        if result.facts:
            result.status = Status.OK
        return result

    @staticmethod
    def _bind_entity(
        value: object, entities: Sequence[ExtractedEntity]
    ) -> tuple[ExtractedEntity | None, FactDropReason | None]:
        """Find the candidate entity whose name equals a field value.

        Exactly one candidate must have a name equal to the field value. None
        gives `unknown_entity_ref` and several give `ambiguous_entity_ref`; in
        both cases the ref is left out and the reason recorded.
        """
        wanted = normalize_text(str(value))
        found = [e for e in entities if normalize_text(e.name) == wanted]
        if len(found) == 1:
            return found[0], None
        return None, (
            FactDropReason.AMBIGUOUS_ENTITY_REF
            if found
            else FactDropReason.UNKNOWN_ENTITY_REF
        )

    @staticmethod
    def _fact(
        episode: Episode,
        spec: FactTemplate,
        data: dict[str, Any],
        *,
        entities: Sequence[ExtractedEntity],
        view: TypeView | None,
        result: FactResult,
    ) -> ExtractedFact | None:
        """Build one fact from a template; record a drop and return None on failure."""
        info: dict[str, Any] = {"template": spec.template}

        def drop(reason: FactDropReason, detail: str = "") -> None:
            """Record one drop for this template."""
            result.dropped.append(FactDropped(info, reason, detail))

        values = {path: _get(data, path) for path in spec.fields}
        missing = [p for p, v in values.items() if v is None or not str(v).strip()]
        if missing:
            drop(FactDropReason.MISSING_FIELD, ", ".join(missing))
            return None
        text = " ".join(
            _PLACEHOLDER.sub(
                lambda m: str(values.get(m.group(1), m.group(0))), spec.template
            ).split()
        )
        if not text:
            drop(FactDropReason.EMPTY_TEXT)
            return None

        refs: list[str] = []
        for path in spec.entity_fields:
            value = _get(data, path)
            if value is None or not str(value).strip():
                drop(FactDropReason.MISSING_FIELD, path)
                continue
            entity, reason = RecordFactMapper._bind_entity(value, entities)
            if entity is None:
                if reason is None:
                    raise RuntimeError("missing entity binding reason")
                drop(reason, path)
            else:
                refs.append(entity.local_id)

        type_id: int | None = None
        if spec.relation_type_label is not None:
            found = (
                view.catalog.relation_by_label(spec.relation_type_label)
                if view
                else None
            )
            by_id = {entity.local_id: entity for entity in entities}
            check = check_relation_type(
                found.id if found else -1, tuple(refs), by_id, view
            )
            type_id = check.type_id
            if check.problem is not None:
                drop(*check.problem)

        valid_from: Any = episode.occurred_at
        precision: Precision | None = Precision.TIME
        basis = TimeBasis.REFERENCE
        if spec.time_field is not None:
            raw_time = _get(data, spec.time_field)
            try:
                if raw_time is None:
                    raise ValueError("missing")
                parsed, found_precision = parse_instant(str(raw_time))
                valid_from = truncate(parsed, found_precision)
                precision, basis = found_precision, TimeBasis.STATED
            except ValueError:
                drop(FactDropReason.TIME_UNPARSED, f"{spec.time_field}={raw_time!r}")
                valid_from, precision, basis = None, None, TimeBasis.NONE

        return ExtractedFact(
            fact_id="",
            text=text,
            entity_refs=tuple(dict.fromkeys(refs)),
            relation_type_id=type_id,
            relation_label=spec.relation_type_label,
            fact_kind=spec.fact_kind,
            polarity=Polarity.AFFIRMED,
            modality=Modality.ACTUAL,
            asserted_by=None,
            valid_from=valid_from,
            valid_to=None,
            precision=precision,
            time_expression=None,
            time_basis=basis,
            evidence_quote=",".join(spec.fields),
            evidence_span=None,
            confidence=1.0,
            method=Method.RECORD,
            episode_id=episode.id,
            fields={p: _get(data, p) for p in (*spec.fields, *spec.entity_fields)},
        )
