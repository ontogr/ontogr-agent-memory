"""Fallback: a cheap extractor first, a model only for episodes it handled badly."""

from collections.abc import Sequence

from ontogr_mem.data_models import Episode, TypeCatalog
from ontogr_mem.entity_extraction.extractors.base import (
    EntityExtractor,
    Extracted,
    Trigger,
)
from ontogr_mem.entity_extraction.extractors.overlaps import resolve_overlaps
from ontogr_mem.entity_extraction.extractors.triggers import (
    EscalationThresholds,
    evaluate_triggers,
)
from ontogr_mem.entity_extraction.text import text_of


class FallbackExtractor(EntityExtractor):
    """Run `cheap`, and also run `llm` when an escalation trigger fires.

    With `only_on` set, only those triggers cause the fallback. The model and
    the cheap pass are merged by the shared rule: entities of the same name are
    merged, and a longer span that adds a content word replaces the fragment it
    contains, whichever side it comes from. The model is merged first, so it
    keeps its type, method and confidence for entities of the same name. Cheap
    entities the model did not cover are kept, so the fallback adds recall
    instead of losing it. Model failures propagate so the episode is extracted
    again instead of being stored with the weak result. The result lists the
    cheap pass's triggers unless the model ran, in which case it lists none.
    """

    def __init__(
        self,
        cheap: EntityExtractor,
        llm: EntityExtractor,
        *,
        thresholds: EscalationThresholds | None = None,
        only_on: frozenset[Trigger] | None = None,
    ) -> None:
        """Create the extractor. `only_on=None` lets any trigger escalate."""
        self._cheap = cheap
        self._llm = llm
        self._thresholds = thresholds or EscalationThresholds()
        self._only_on = only_on
        self.languages = cheap.languages

    async def extract(
        self,
        episode: Episode,
        types: TypeCatalog,
        previous: Sequence[Episode] = (),
    ) -> Extracted:
        """Run the cheap pass, then the model when a trigger fires."""
        first = await self._cheap.extract(episode, types, previous)
        fired = evaluate_triggers(episode, first.entities, self._thresholds)
        escalate = bool(fired) and (
            self._only_on is None or bool(self._only_on & set(fired))
        )
        if not escalate:
            return Extracted(first.entities, first.usage, fired)
        second = await self._llm.extract(episode, types, previous)
        first.usage.merge(second.usage)
        entities = resolve_overlaps(
            [second.entities, first.entities], text=text_of(episode)
        )
        return Extracted(entities, first.usage)
