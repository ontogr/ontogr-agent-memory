"""Entity extraction with a language model through the shared gate."""

import logging
from collections.abc import Sequence
from dataclasses import replace

from ontogr_mem.data_models import (
    UNKNOWN_TYPE_ID,
    Episode,
    ExtractedEntity,
    Method,
    TypeCatalog,
    Usage,
)
from ontogr_mem.data_models.text import normalize_name
from ontogr_mem.entity_extraction.errors import ParseFailedError
from ontogr_mem.entity_extraction.extractors.base import EntityExtractor, Extracted
from ontogr_mem.entity_extraction.parsing import (
    RawEntity,
    SchemaError,
    parse_entities,
)
from ontogr_mem.entity_extraction.prompts import (
    NER_SYSTEM,
    build_ner_prompt,
    ner_schema,
)
from ontogr_mem.entity_extraction.settings import EntityExtractionSettings
from ontogr_mem.entity_extraction.text import find_spans, text_of
from ontogr_mem.llm import InputTooLargeError, LLMGate, LLMRequest, ModelSize


log = logging.getLogger(__name__)


class LLMExtractor(EntityExtractor):
    """Entity extraction with a language model.

    The model names entities and numeric type ids. Offsets are never trusted: the
    spans are found in the episode text, and an entity that cannot be found there
    is discarded. Earlier episodes go into the prompt as context only. When the
    prompt is too large, the oldest earlier episode is dropped first and the
    current episode is never cut.
    """

    def __init__(
        self,
        gate: LLMGate,
        settings: EntityExtractionSettings | None = None,
        *,
        model_size: ModelSize = ModelSize.SMALL,
        languages: frozenset[str] | None = None,
    ) -> None:
        """Create the extractor.

        Args:
            gate: The shared gate that every model call passes through.
            settings: Threshold, repair budget and default confidence.
            model_size: Model tier to call.
            languages: Languages handled, or None for any.
        """
        self._gate = gate
        self._settings = settings or EntityExtractionSettings()
        self._size = model_size
        self.languages = languages

    async def extract(
        self,
        episode: Episode,
        types: TypeCatalog,
        previous: Sequence[Episode] = (),
    ) -> Extracted:
        """Run the entity prompt with a repair loop.

        Raises:
            ParseFailedError: If the reply stays invalid after the repair budget.
            InputTooLargeError: If the episode alone exceeds the input cap.
        """
        usage = Usage()
        if not text_of(episode).strip():
            return Extracted([], usage)
        context = list(previous)
        repair: tuple[str, list[str]] | None = None
        errors: list[str] = []
        attempt = 0
        while attempt <= self._settings.max_repairs:
            request = LLMRequest(
                system=NER_SYSTEM,
                prompt=build_ner_prompt(episode, types, context, repair),
                model_size=self._size,
                schema=ner_schema() if attempt == 0 else None,
            )
            try:
                response = await self._gate.complete(request)
            except InputTooLargeError:
                if not context:
                    raise
                context.pop(0)
                continue
            usage.add(
                "entities",
                self._size.value,
                response.input_tokens,
                response.output_tokens,
            )
            try:
                raw = parse_entities(response.text)
            except SchemaError as exc:
                errors, repair = exc.errors, (response.text, exc.errors)
                attempt += 1
                continue
            return Extracted(self._bind(episode, types, raw), usage)
        raise ParseFailedError("; ".join(errors))

    def _bind(
        self, episode: Episode, types: TypeCatalog, raw: list[RawEntity]
    ) -> list[ExtractedEntity]:
        text = text_of(episode)
        out: list[ExtractedEntity] = []
        seen: dict[str, int] = {}
        for item in raw:
            name = item.name.strip()
            if not name or not types.has_entity_id(item.type_id):
                log.debug("discarding entity %r with type id %s", name, item.type_id)
                continue
            spans = find_spans(text, name)
            if not spans:
                log.debug("discarding entity %r: not in the episode", name)
                continue
            confidence = (
                item.confidence
                if item.confidence is not None
                else self._settings.default_confidence
            )
            type_id = (
                item.type_id
                if confidence >= self._settings.threshold
                else UNKNOWN_TYPE_ID
            )
            label = (item.type_label or "").strip() or None
            name_key = normalize_name(name)
            if (index := seen.get(name_key)) is not None:
                prior = out[index]
                out[index] = replace(
                    prior, confidence=max(prior.confidence, confidence)
                )
                continue
            seen[name_key] = len(out)
            out.append(
                ExtractedEntity(
                    local_id=f"e{len(out)}",
                    name=name,
                    type_id=type_id,
                    confidence=confidence,
                    method=Method.LLM,
                    episode_id=episode.id,
                    type_label=label if item.type_id == UNKNOWN_TYPE_ID else None,
                    description=item.description,
                    spans=tuple(spans),
                )
            )
        return out
