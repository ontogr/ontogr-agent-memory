"""Extract entities and facts from an episode in one model call."""

from ontogr_mem.combined_extraction.agent import combined_agent
from ontogr_mem.combined_extraction.prompts import build_combined_prompt
from ontogr_mem.data_models import (
    Episode,
    ExtractionResult,
    PreviousEpisodes,
    Status,
    TokenCount,
    TypeView,
    Usage,
)
from ontogr_mem.entity_extraction.errors import UnknownKindError
from ontogr_mem.entity_extraction.extractors.llm import bind_entities
from ontogr_mem.entity_extraction.parsing import RawEntity
from ontogr_mem.entity_extraction.pipeline import TEXT_KINDS, add_speaker
from ontogr_mem.entity_extraction.settings import EntityExtractionSettings
from ontogr_mem.entity_extraction.text import language_of, text_of
from ontogr_mem.fact_extraction.binding import bind
from ontogr_mem.fact_extraction.candidates import ACTOR_ID, with_actor
from ontogr_mem.fact_extraction.models import FactResult
from ontogr_mem.fact_extraction.parsing import RawFact
from ontogr_mem.fact_extraction.settings import FactExtractionSettings
from ontogr_mem.llm import (
    ModelFailure,
    ModelOutputError,
    ModelRole,
    RoleModels,
    run_agent,
)


MISSING_REF = "?"


class CombinedExtractor:
    """Extract entities and facts from one text episode with one model call."""

    def __init__(
        self,
        models: RoleModels,
        entity_settings: EntityExtractionSettings | None = None,
        fact_settings: FactExtractionSettings | None = None,
    ) -> None:
        """Create an extractor with model and extraction settings."""
        self._models = models
        self._entity_settings = entity_settings or EntityExtractionSettings()
        self._fact_settings = fact_settings or FactExtractionSettings()

    async def extract(
        self, episode: Episode, view: TypeView, previous: PreviousEpisodes | None
    ) -> tuple[ExtractionResult, FactResult]:
        """Extract entities and facts, using provider-selected earlier episodes."""
        entities_result = ExtractionResult(episode.id, Status.EMPTY)
        fact_result = FactResult(episode.id, Status.EMPTY)
        if episode.kind not in TEXT_KINDS:
            raise UnknownKindError(episode.kind)
        language = language_of(episode)
        if (
            self._fact_settings.languages is not None
            and language not in self._fact_settings.languages
        ):
            entities_result.status = fact_result.status = Status.SKIPPED
            entities_result.detail = fact_result.detail = "unsupported language"
            return entities_result, fact_result
        if not text_of(episode).strip():
            return entities_result, fact_result
        previous_list = (
            await previous.before(episode, self._fact_settings.max_previous)
            if previous and self._fact_settings.max_previous > 0
            else []
        )
        try:
            run = await run_agent(
                combined_agent,
                ModelRole.FACT_EXTRACTION,
                build_combined_prompt(episode, view, previous_list),
                models=self._models,
                deps=None,
            )
        except ModelFailure as exc:
            status = (
                Status.PARSE_FAILED
                if isinstance(exc, ModelOutputError)
                else Status.ERROR
            )
            entities_result.status = fact_result.status = status
            entities_result.detail = fact_result.detail = str(exc)
            if exc.usage is not None:
                fact_result.usage = Usage(
                    buckets={
                        "combined/fact_extraction": TokenCount(
                            exc.usage.input_tokens,
                            exc.usage.output_tokens,
                            exc.usage.requests,
                        )
                    }
                )
            return entities_result, fact_result

        output = run.output
        raws = [
            RawEntity(**item.model_dump(exclude={"ref"})) for item in output.entities
        ]
        entities, local_ids = bind_entities(
            episode, view.catalog, raws, self._entity_settings
        )
        refs = {
            item.ref: local
            for item, local in zip(output.entities, local_ids, strict=True)
            if local is not None
        }
        entities = add_speaker(episode, entities, view.catalog)
        entities_result.entities = entities
        entities_result.status = (
            Status.OK
            if local_ids and any(local is not None for local in local_ids)
            else Status.EMPTY
        )
        candidates = with_actor(episode, entities, view)
        raw_facts = [
            RawFact(
                **item.model_dump(exclude={"entity_refs"}),
                entity_refs=tuple(
                    ref if ref == ACTOR_ID else refs.get(ref, MISSING_REF)
                    for ref in item.entity_refs
                ),
            )
            for item in output.facts
        ]
        bound = bind(
            episode,
            candidates,
            raw_facts,
            view,
            default_confidence=self._fact_settings.default_confidence,
            previous=previous_list,
        )
        fact_result.facts, fact_result.dropped = bound.facts, bound.dropped
        fact_result.status = Status.OK if bound.facts else Status.EMPTY
        fact_result.usage = Usage(
            buckets={
                "combined/fact_extraction": TokenCount(
                    run.usage.input_tokens, run.usage.output_tokens, run.usage.requests
                )
            }
        )
        return entities_result, fact_result
