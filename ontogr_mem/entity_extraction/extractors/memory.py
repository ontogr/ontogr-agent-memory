"""Memory extractor: base entities, then a small-model pass for attributes."""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ontogr_mem.data_models import Episode, ExtractedEntity, TypeCatalog, Usage
from ontogr_mem.entity_extraction.extractors.base import EntityExtractor, Extracted
from ontogr_mem.entity_extraction.text import find_spans, text_of
from ontogr_mem.llm import LLMError, LLMGate, LLMRequest, ModelSize


log = logging.getLogger(__name__)

DEFAULT_ATTRIBUTES: Mapping[str, tuple[str, ...]] = {
    "Person": ("role", "relationship_to_user", "stance"),
    "Preference": ("polarity", "strength", "scope"),
    "Event": ("when", "status"),
    "Organization": ("relationship_to_user",),
}

ENRICH_SYSTEM = (
    "You annotate entities found in a user's message. Reply with JSON only: "
    '{"entities": [{"id": "e0", "description": "...", "attributes": '
    '{"<name>": {"value": "...", "evidence": "<exact quote from the text>"}}}]}. '
    "Use only facts stated in the text. Only use the allowed attribute names "
    "listed for each entity. Omit an attribute when the text does not state it."
)

_MAX_DESCRIPTION = 300
_Scalar = str | int | float | bool


class MemoryEntityExtractor(EntityExtractor):
    """Wrap an extractor and fill `description` and `attributes`.

    One small-model call per batch of entities. The model never changes names,
    types or spans. An attribute is kept only if it is allowed for the entity's
    type, has a scalar value, and its `evidence` quote is found in the episode
    text. A failed or malformed reply leaves the plain result.
    """

    def __init__(
        self,
        base: EntityExtractor,
        gate: LLMGate,
        *,
        allowed_attributes: Mapping[str, tuple[str, ...]] | None = None,
        max_entities_per_call: int = 25,
        model_size: ModelSize = ModelSize.SMALL,
    ) -> None:
        """Create the extractor around `base`."""
        if max_entities_per_call < 1:
            msg = "max_entities_per_call must be >= 1"
            raise ValueError(msg)
        self._base = base
        self._gate = gate
        self._allowed = dict(allowed_attributes or DEFAULT_ATTRIBUTES)
        self._max = max_entities_per_call
        self._size = model_size
        self.languages = base.languages

    async def extract(
        self,
        episode: Episode,
        types: TypeCatalog,
        previous: Sequence[Episode] = (),
    ) -> Extracted:
        """Run the base extractor, then enrich its entities in batches."""
        base = await self._base.extract(episode, types, previous)
        if not base.entities:
            return base
        text = text_of(episode)
        enriched: list[ExtractedEntity] = []
        for i in range(0, len(base.entities), self._max):
            batch = base.entities[i : i + self._max]
            enriched.extend(await self._enrich(text, batch, types, base.usage))
        return Extracted(enriched, base.usage, base.triggers)

    async def _enrich(
        self,
        text: str,
        batch: list[ExtractedEntity],
        types: TypeCatalog,
        usage: Usage,
    ) -> list[ExtractedEntity]:
        labels = {e.local_id: types.entity_label(e.type_id) for e in batch}
        listing = [
            {
                "id": e.local_id,
                "name": e.name,
                "type": labels[e.local_id],
                "allowed_attributes": self._allowed.get(labels[e.local_id], ()),
            }
            for e in batch
        ]
        try:
            response = await self._gate.complete(
                LLMRequest(
                    system=ENRICH_SYSTEM,
                    prompt=f"TEXT:\n{text}\n\nENTITIES:\n{json.dumps(listing)}",
                    model_size=self._size,
                )
            )
        except LLMError:
            log.warning("enrichment failed; keeping the plain result", exc_info=True)
            return batch
        usage.add(
            "enrich", self._size.value, response.input_tokens, response.output_tokens
        )
        replies = _parse(response.text)
        return [
            self._apply(text, e, labels[e.local_id], replies.get(e.local_id))
            for e in batch
        ]

    def _apply(
        self,
        text: str,
        entity: ExtractedEntity,
        label: str,
        item: dict[str, Any] | None,
    ) -> ExtractedEntity:
        if item is None:
            return entity
        allowed = self._allowed.get(label, ())
        attributes: dict[str, _Scalar] = {}
        raw_attributes = item.get("attributes")
        for name, spec in (
            raw_attributes if isinstance(raw_attributes, dict) else {}
        ).items():
            if name not in allowed or not isinstance(spec, dict):
                continue
            value, evidence = spec.get("value"), spec.get("evidence")
            if (
                isinstance(value, _Scalar)
                and isinstance(evidence, str)
                and find_spans(text, evidence)
            ):
                attributes[name] = value
        description = item.get("description")
        return replace(
            entity,
            description=(
                description.strip()[:_MAX_DESCRIPTION]
                if isinstance(description, str) and description.strip()
                else entity.description
            ),
            attributes={**entity.attributes, **attributes},
        )


def _parse(raw: str) -> dict[str, dict[str, Any]]:
    """Map entity id to its reply item, or return `{}` for a malformed reply."""
    try:
        items = json.loads(raw)["entities"]
        return {it["id"]: it for it in items if isinstance(it, dict)}
    except (ValueError, KeyError, TypeError):
        return {}
