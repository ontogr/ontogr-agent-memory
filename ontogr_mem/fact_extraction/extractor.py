"""Fact extractors: the LLM pass, the record path and the kind router.

Three pieces, all behind the `FactExtractor` protocol:

* `LLMFactExtractor`: messages, documents and tool output. One agent run per
  episode. The model reply is validated against a typed shape (the library
  re-asks the model up to twice for malformed output), then binding drops items
  that fail domain checks.
* `RecordFactMapper` (record_facts.py): records and events, no model.
* `RoutingFactExtractor`: picks between them by `Episode.kind`. An unknown kind
  raises; it never falls back to the model.

Failures come back as a `FactResult` status (error, parse_failed, skipped); an
unknown kind raises.

The candidates are prepared in `candidates.py` and the prompt is rendered in
`prompts.py`.
"""

import logging
from collections.abc import Sequence
from typing import Protocol

from ontogr_mem.data_models import (
    Episode,
    ExtractedEntity,
    Kind,
    Status,
    TypeView,
    Usage,
)
from ontogr_mem.data_models.extraction import TokenCount
from ontogr_mem.fact_extraction.agent import fact_agent
from ontogr_mem.fact_extraction.binding import bind
from ontogr_mem.fact_extraction.candidates import with_actor
from ontogr_mem.fact_extraction.errors import UnknownKindError
from ontogr_mem.fact_extraction.models import FactResult, SkipReason, skipped
from ontogr_mem.fact_extraction.parsing import RawFact
from ontogr_mem.fact_extraction.prompts import build_prompt
from ontogr_mem.fact_extraction.record_facts import RecordFactMapper
from ontogr_mem.fact_extraction.settings import FactExtractionSettings
from ontogr_mem.fact_extraction.text import language_of
from ontogr_mem.llm import (
    ModelFailure,
    ModelOutputError,
    ModelRole,
    RoleModels,
    run_agent,
)


log = logging.getLogger(__name__)

# Kinds whose content is prose and goes to the model.
TEXT_KINDS = frozenset({Kind.MESSAGE, Kind.DOCUMENT, Kind.TOOL_OUTPUT})
# Kinds whose content is a dict (records, provider events): mapped without a model.
STRUCTURED_KINDS = frozenset({Kind.RECORD, Kind.EVENT})


class FactExtractor(Protocol):
    """Extracts the facts of one episode."""

    version: str

    async def extract(
        self,
        episode: Episode,
        entities: Sequence[ExtractedEntity],
        types: TypeView | None,
        previous: Sequence[Episode] = (),
        *,
        force: bool = False,
    ) -> FactResult:
        """Return the episode's facts; failures come back as a status."""
        ...


class LLMFactExtractor:
    """Runs the fact pass as one typed agent run per episode.

    The model reply is validated against a typed shape; the library re-asks the
    model up to twice for malformed output. A run whose reply never validates or
    is cut off at the output limit ends the episode as `PARSE_FAILED`. Other
    model failures end it as `ERROR`. Usage spent before a failure is kept.
    """

    def __init__(
        self,
        models: RoleModels,
        settings: FactExtractionSettings | None = None,
    ) -> None:
        """Configure model access and fact extraction rules."""
        self._models = models
        self._settings = settings or FactExtractionSettings()
        self.languages = self._settings.languages
        self.version = self._settings.version

    async def extract(
        self,
        episode: Episode,
        entities: Sequence[ExtractedEntity],
        types: TypeView | None,
        previous: Sequence[Episode] = (),
        *,
        force: bool = False,
    ) -> FactResult:
        """Run the fact pass for one episode."""
        result = FactResult(episode.id, Status.EMPTY)
        if episode.kind not in TEXT_KINDS:
            msg = f"no LLM fact path for kind {episode.kind!r}"
            raise UnknownKindError(msg)
        language = language_of(episode)
        if self.languages is not None and language not in self.languages:
            return skipped(
                result, SkipReason.UNSUPPORTED_LANGUAGE, f"language {language!r}"
            )
        is_message = episode.kind is Kind.MESSAGE
        if not (force or is_message or len(entities) >= self._settings.min_entities):
            return skipped(
                result,
                SkipReason.NO_CANDIDATES,
                f"fewer than {self._settings.min_entities} candidate entities",
            )

        candidates = with_actor(episode, entities, types)
        try:
            run = await run_agent(
                fact_agent,
                ModelRole.FACT_EXTRACTION,
                build_prompt(
                    episode,
                    candidates,
                    types,
                    previous,
                    instructions=self._settings.instructions,
                ),
                models=self._models,
                deps=None,
            )
        except ModelFailure as exc:
            log.warning("fact call failed for %s: %r", episode.id, exc)
            if exc.usage is not None:
                result.usage = Usage(
                    buckets={
                        "facts/fact_extraction": TokenCount(
                            exc.usage.input_tokens,
                            exc.usage.output_tokens,
                            exc.usage.requests,
                        ),
                    }
                )
            result.status = (
                Status.PARSE_FAILED
                if isinstance(exc, ModelOutputError)
                else Status.ERROR
            )
            result.detail = str(exc)
            return result
        result.usage = Usage(
            buckets={
                "facts/fact_extraction": TokenCount(
                    run.usage.input_tokens,
                    run.usage.output_tokens,
                    run.usage.requests,
                ),
            }
        )
        raws = [
            RawFact(
                **item.model_dump(exclude={"entity_refs"}),
                entity_refs=tuple(item.entity_refs),
            )
            for item in run.output.facts
        ]

        bound = bind(
            episode,
            candidates,
            raws,
            types,
            default_confidence=self._settings.default_confidence,
            previous=previous,
        )
        result.facts = bound.facts
        result.dropped = bound.dropped
        result.status = Status.OK if bound.facts else Status.EMPTY
        return result


class RoutingFactExtractor:
    """Selects the path by kind: records without an LLM, the rest with one."""

    def __init__(self, llm: LLMFactExtractor, records: RecordFactMapper) -> None:
        """Wire the two paths."""
        self._llm = llm
        self._records = records
        self.version = f"{llm.version}+{records.version}"

    async def extract(
        self,
        episode: Episode,
        entities: Sequence[ExtractedEntity],
        types: TypeView | None,
        previous: Sequence[Episode] = (),
        *,
        force: bool = False,
    ) -> FactResult:
        """Route by `Episode.kind`; an unknown kind raises."""
        kind = episode.kind
        if kind in STRUCTURED_KINDS:
            return self._records.map(episode, entities, types)
        if kind in TEXT_KINDS:
            return await self._llm.extract(
                episode, entities, types, previous, force=force
            )
        msg = f"no fact extraction path for kind {kind!r}"
        raise UnknownKindError(msg)
