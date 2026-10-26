"""Fact extractors: the LLM pass, the record path and the kind router.

Three pieces, all behind the `FactExtractor` protocol:

* `LLMFactExtractor`: messages, documents and tool output. One model call per
  episode (with a fallback chain), then binding validates the reply.
* `RecordFactMapper` (record_facts.py): records and events, no model.
* `RoutingFactExtractor`: picks between them by `Episode.kind`. An unknown kind
  raises; it never falls back to the model.

Failures come back as a `FactResult` status (error, parse_failed, skipped); an
unknown kind raises.

The candidates are prepared in `candidates.py`, the prompt and the request in
`prompting.py`.
"""

import logging
from collections.abc import Callable, Sequence
from typing import Protocol

from ontogr_mem.data_models import (
    Episode,
    ExtractedEntity,
    Kind,
    Status,
    TypeView,
    Usage,
)
from ontogr_mem.fact_extraction.binding import bind
from ontogr_mem.fact_extraction.candidates import with_actor
from ontogr_mem.fact_extraction.errors import UnknownKindError
from ontogr_mem.fact_extraction.models import (
    FactResult,
    SkipReason,
    skipped,
    unusable,
)
from ontogr_mem.fact_extraction.parsing import RawFact, SchemaError, parse_facts
from ontogr_mem.fact_extraction.prompting import build_request, fit_prompt
from ontogr_mem.fact_extraction.record_facts import RecordFactMapper
from ontogr_mem.fact_extraction.settings import FactExtractionSettings
from ontogr_mem.fact_extraction.text import language_of
from ontogr_mem.llm import (
    InputTooLargeError,
    LLMCallError,
    LLMError,
    LLMGate,
    ModelSize,
    estimate_tokens,
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
    """Runs the fact pass over an `LLMGate`, with up to three attempts per episode.

    Attempt 1 sends the schema strictly, attempt 2 non-strictly (providers that
    reject strict schemas), attempt 3 sends no schema and feeds the validation
    errors back. A provider failure moves to the next, looser attempt; a timeout
    ends the episode as `ERROR`.
    """

    def __init__(
        self,
        gate: LLMGate,
        settings: FactExtractionSettings | None = None,
        *,
        model_size: ModelSize = ModelSize.MEDIUM,
        max_output_tokens: int = 4096,
        count_tokens: Callable[[str], int] = estimate_tokens,
    ) -> None:
        """Configure model size, extraction settings and the output cap."""
        self._gate = gate
        self._settings = settings or FactExtractionSettings()
        self._size = model_size
        self._max_output = max_output_tokens
        self._count = count_tokens
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
        usage = Usage()
        raws: list[RawFact] | None = None
        last_reply = ""
        errors: list[str] = []
        # `failure` is set by provider or timeout problems; an invalid reply ends
        # as parse_failed instead.
        failure: Exception | None = None
        for attempt in range(3):
            repair = (last_reply, errors) if attempt == 2 and errors else None
            try:
                prompt = fit_prompt(
                    episode,
                    candidates,
                    types,
                    previous,
                    repair,
                    max_input_tokens=self._gate.max_input_tokens,
                    instructions=self._settings.instructions,
                    count_tokens=self._count,
                )
                request = build_request(
                    attempt,
                    prompt,
                    model_size=self._size,
                    max_output_tokens=self._max_output,
                    with_types=types is not None,
                )
                response = await self._gate.complete(request)
            except InputTooLargeError as exc:
                # Raised by the fit check or by the gate's input cap.
                if repair is not None:
                    # Only the repair prompt is too large: the episode ends as
                    # parse_failed with the earlier errors.
                    break
                result.usage = usage
                return skipped(result, SkipReason.OVERSIZE, str(exc))
            except LLMCallError as exc:
                failure = exc
                continue
            except (LLMError, TimeoutError) as exc:
                failure = exc
                break
            # A reply arrived, so any earlier provider failure no longer applies.
            failure = None
            usage.add(
                "facts", self._size.value, response.input_tokens, response.output_tokens
            )
            try:
                raws = parse_facts(response.text)
                break
            except SchemaError as exc:
                last_reply, errors = response.text, exc.errors

        result.usage = usage
        if raws is None:
            if failure is not None:
                log.warning("fact call failed for %s: %r", episode.id, failure)
            return unusable(result, failure, errors)

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
