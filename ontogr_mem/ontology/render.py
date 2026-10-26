"""Render a catalog snapshot for a prompt, and keep the view that decodes it."""

import json
from datetime import datetime
from typing import TypeVar

from ontogr_mem.ontology.models import (
    UNKNOWN_ID,
    Entity,
    Kind,
    Mode,
    Ontology,
    Relation,
    RenderedView,
    Status,
)


_WORDING = {
    Mode.STRICT: "Allowed {what}. Use only these.",
    Mode.PREFERRED: "Preferred {what}. Use one of these when it fits.",
    Mode.OPEN: "Suggested {what} (hints only; other labels are accepted).",
}

_TypeT = TypeVar("_TypeT", bound=Entity | Relation)


def first_paragraph(text: str, limit: int) -> str:
    """Return the first paragraph of ``text``, cut to ``limit`` characters."""
    para = text.strip().split("\n\n", 1)[0].replace("\n", " ")
    return para if len(para) <= limit else para[: max(limit - 1, 0)].rstrip() + "…"


def _row(t: _TypeT, index: int, max_desc_chars: int) -> dict[str, object]:
    """Return the row that describes one type in the prompt."""
    row: dict[str, object] = {
        "id": index,
        "label": t.label,
        "description": first_paragraph(t.description, max_desc_chars),
    }
    attrs = [
        {
            "name": a.name,
            "type": a.value_type.value,
            "required": a.required,
            "description": first_paragraph(a.description, max_desc_chars),
        }
        for a in t.properties
        if a.status is Status.ACTIVE and a.extractable and not a.structured
    ]
    if attrs:
        row["attributes"] = attrs
    return row


def _relation_extras(cat: Ontology, rel: Relation, row: dict[str, object]) -> None:
    """Add the patterns and the symmetry of a relation to its row."""
    pairs = []
    for p in rel.patterns:
        if p.status is Status.ACTIVE:
            s, g = cat.entity(p.src), cat.entity(p.tgt)
            if s and g:
                pairs.append([s.label, g.label])
    if pairs:
        row["patterns"] = pairs
    if rel.symmetric:
        row["symmetric"] = True


def render(
    cat: Ontology,
    kind: Kind,
    *,
    max_types: int = 100,
    max_desc_chars: int = 200,
    include: list[int] | None = None,
    now: datetime | None = None,
) -> tuple[str, RenderedView]:
    """Numbered JSON list of the types offered to the LLM.

    Positions in the list are the ids the LLM returns; ``RenderedView.shown``
    maps them back to ``type_id``. Deprecated types are not offered, and
    ``Unknown`` is offered unless the mode is strict.
    """
    mode = cat.mode(kind)
    pool: list[Entity | Relation]
    if kind is Kind.ENTITY:
        pool = [
            e
            for e in cat.active_entities()
            if not (e.type_id == UNKNOWN_ID and mode is Mode.STRICT)
        ]
    else:
        pool = list(cat.active_relations())
    if include is not None:
        keep = set(include)
        pool = [t for t in pool if t.type_id in keep or t.type_id == UNKNOWN_ID]
    unknown = [t for t in pool if kind is Kind.ENTITY and t.type_id == UNKNOWN_ID]
    rest = [t for t in pool if t not in unknown]
    shown = [*unknown, *rest[: max(max_types - len(unknown), 0)]]
    rows: list[dict[str, object]] = []
    for i, t in enumerate(shown):
        row = _row(t, i, max_desc_chars)
        if isinstance(t, Relation):
            _relation_extras(cat, t, row)
        rows.append(row)
    what = "entity types" if kind is Kind.ENTITY else "relation types"
    text = (
        _WORDING[mode].format(what=what) + "\n" + json.dumps(rows, ensure_ascii=False)
    )
    view = RenderedView(
        catalog_version=cat.version,
        content_hash=cat.content_hash,
        kind=kind,
        mode=mode,
        shown=[t.type_id for t in shown],
    )
    if now is not None:
        view = view.model_copy(update={"rendered_at": now})
    return text, view
