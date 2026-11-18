"""Entity extraction with a language model through the shared gate."""

import logging
from collections.abc import Sequence
from dataclasses import replace

from ontogr_mem.data_models import (
    UNKNOWN_TYPE_ID,
    Dropped,
    DropReason,
    Episode,
    ExtractedEntity,
    Method,
    TypeCatalog,
    Usage,
)
from ontogr_mem.data_models.extraction import TokenCount
from ontogr_mem.data_models.text import normalize_name
from ontogr_mem.entity_extraction.agent import entity_agent
from ontogr_mem.entity_extraction.extractors.base import EntityExtractor, Extracted
from ontogr_mem.entity_extraction.parsing import RawEntity
from ontogr_mem.entity_extraction.prompts import build_ner_prompt
from ontogr_mem.entity_extraction.settings import EntityExtractionSettings
from ontogr_mem.entity_extraction.text import find_spans, text_of
from ontogr_mem.llm import ModelRole, RoleModels, run_agent


log = logging.getLogger(__name__)


class LLMExtractor(EntityExtractor):
    """Entity extraction with a language model.

    The model names entities and numeric type ids. Offsets are never trusted: the
    spans are found in the episode text, and an entity that cannot be found there
    is discarded. Earlier episodes go into the prompt as context only. The reply
    shape is validated and re-asked by the shared model run.
    """

    def __init__(
        self,
        models: RoleModels,
        settings: EntityExtractionSettings | None = None,
        *,
        languages: frozenset[str] | None = None,
    ) -> None:
        """Configure model access, confidence thresholds and supported languages."""
        self._models = models
        self._settings = settings or EntityExtractionSettings()
        self.languages = languages

    async def extract(
        self,
        episode: Episode,
        types: TypeCatalog,
        previous: Sequence[Episode] = (),
    ) -> Extracted:
        """Extract typed entities, then bind names and types to the episode.

        Raises:
            ModelFailure: If the model is unavailable or its reply is unusable.
        """
        if not text_of(episode).strip():
            return Extracted([], Usage())
        run = await run_agent(
            entity_agent,
            ModelRole.ENTITY_EXTRACTION,
            build_ner_prompt(episode, types, previous),
            models=self._models,
            deps=None,
        )
        raw = [RawEntity(**item.model_dump()) for item in run.output.entities]
        usage = Usage(
            buckets={
                "entities/entity_extraction": TokenCount(
                    run.usage.input_tokens,
                    run.usage.output_tokens,
                    run.usage.requests,
                ),
            }
        )
        return Extracted(bind_entities(episode, types, raw, self._settings)[0], usage)


def bind_entities(
    episode: Episode,
    types: TypeCatalog,
    raw: Sequence[RawEntity],
    settings: EntityExtractionSettings,
) -> tuple[list[ExtractedEntity], list[str | None], list[Dropped]]:
    """Bind model entities to the episode text and the type catalog.

    Names that are not in the episode text and ids outside the catalog are
    discarded and reported as drops. Entities with the same normalized name and
    effective type collapse into the first, keeping the higher confidence.

    Returns:
        The bound entities, for each raw item the local id of the entity it
        became, and every rejected item with its reason. A discarded item has
        None as its local id. A duplicate has the id of the entity it collapsed
        into and is not reported as a drop.
    """
    text = text_of(episode)
    out: list[ExtractedEntity] = []
    ids: list[str | None] = []
    dropped: list[Dropped] = []
    seen: dict[tuple[str, int], int] = {}
    for item in raw:
        name = item.name.strip()
        info = {"name": item.name, "type_id": item.type_id}
        if not name:
            log.debug("discarding entity with a blank name")
            dropped.append(Dropped(info, DropReason.MISSING_FIELD, "blank name"))
            ids.append(None)
            continue
        if not types.has_entity_id(item.type_id):
            log.debug("discarding entity %r with type id %s", name, item.type_id)
            dropped.append(
                Dropped(info, DropReason.TYPE_OUT_OF_RANGE, f"type id {item.type_id}")
            )
            ids.append(None)
            continue
        spans = find_spans(text, name)
        if not spans:
            log.debug("discarding entity %r: not in the episode", name)
            dropped.append(
                Dropped(info, DropReason.MISSING_FIELD, "name not in the episode")
            )
            ids.append(None)
            continue
        confidence = (
            item.confidence
            if item.confidence is not None
            else settings.default_confidence
        )
        type_id = item.type_id if confidence >= settings.threshold else UNKNOWN_TYPE_ID
        proposed = (item.type_label or "").strip()
        name_key = normalize_name(name)
        if (index := seen.get((name_key, type_id))) is not None:
            prior = out[index]
            out[index] = replace(prior, confidence=max(prior.confidence, confidence))
            ids.append(out[index].local_id)
            continue
        seen[(name_key, type_id)] = len(out)
        local_id = f"e{len(out)}"
        out.append(
            ExtractedEntity(
                local_id=local_id,
                name=name,
                type_id=type_id,
                confidence=confidence,
                method=Method.LLM,
                episode_id=episode.id,
                proposed_type=proposed if item.type_id == UNKNOWN_TYPE_ID else "",
                description=item.description,
                spans=tuple(spans),
            )
        )
        ids.append(local_id)
    return out, ids, dropped
