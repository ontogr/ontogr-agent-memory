"""Memory extractor: base entities, then role-model enrichment."""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace

from ontogr_mem.data_models import Episode, ExtractedEntity, TypeCatalog, Usage
from ontogr_mem.entity_extraction.extractors.base import EntityExtractor, Extracted
from ontogr_mem.entity_extraction.extractors.memory_agent import (
    EntityAnnotation,
    memory_agent,
)
from ontogr_mem.entity_extraction.text import find_spans, text_of
from ontogr_mem.llm import ModelFailure, ModelRole, RoleModels, run_agent


log = logging.getLogger(__name__)

DEFAULT_ATTRIBUTES: Mapping[str, tuple[str, ...]] = {
    "Person": ("role", "relationship_to_user", "stance"),
    "Preference": ("polarity", "strength", "scope"),
    "Event": ("when", "status"),
    "Organization": ("relationship_to_user",),
}

_MAX_DESCRIPTION = 300


class MemoryEntityExtractor(EntityExtractor):
    """Wrap an extractor and fill `description` and `attributes`.

    One role-model call enriches each batch of entities. The model never changes
    names, types or spans. An attribute is kept only if it is allowed for the
    entity's type and its evidence quote is found in the Episode. A failed run
    leaves the base result unchanged.
    """

    def __init__(
        self,
        base: EntityExtractor,
        models: RoleModels,
        *,
        allowed_attributes: Mapping[str, tuple[str, ...]] | None = None,
        max_entities_per_call: int = 25,
    ) -> None:
        """Create the extractor around the base extractor and role models."""
        if max_entities_per_call < 1:
            raise ValueError("max_entities_per_call must be >= 1")
        self._base = base
        self._models = models
        self._allowed = dict(allowed_attributes or DEFAULT_ATTRIBUTES)
        self._max = max_entities_per_call
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
            result = await run_agent(
                memory_agent,
                ModelRole.ENTITY_EXTRACTION,
                f"TEXT:\n{text}\n\nENTITIES:\n{json.dumps(listing)}",
                models=self._models,
                deps=None,
            )
        except ModelFailure as exc:
            if exc.usage is not None and exc.usage.requests:
                usage.add(
                    "enrich",
                    ModelRole.ENTITY_EXTRACTION.value,
                    exc.usage.input_tokens,
                    exc.usage.output_tokens,
                )
                usage.buckets[f"enrich/{ModelRole.ENTITY_EXTRACTION.value}"].calls += (
                    exc.usage.requests - 1
                )
            log.warning("enrichment failed; keeping the plain result", exc_info=True)
            return batch
        tokens = result.usage
        if tokens.requests:
            usage.add(
                "enrich",
                ModelRole.ENTITY_EXTRACTION.value,
                tokens.input_tokens,
                tokens.output_tokens,
            )
            usage.buckets[f"enrich/{ModelRole.ENTITY_EXTRACTION.value}"].calls += (
                tokens.requests - 1
            )
        replies = {item.id: item for item in result.output.entities}
        return [
            self._apply(text, e, labels[e.local_id], replies.get(e.local_id))
            for e in batch
        ]

    def _apply(
        self,
        text: str,
        entity: ExtractedEntity,
        label: str,
        item: EntityAnnotation | None,
    ) -> ExtractedEntity:
        if item is None:
            return entity
        allowed = self._allowed.get(label, ())
        attributes: dict[str, str | int | float | bool] = {}
        for name, spec in item.attributes.items():
            if name not in allowed:
                continue
            if find_spans(text, spec.evidence):
                attributes[name] = spec.value
        return replace(
            entity,
            description=(
                item.description.strip()[:_MAX_DESCRIPTION]
                if item.description.strip()
                else entity.description
            ),
            attributes={**entity.attributes, **attributes},
        )
