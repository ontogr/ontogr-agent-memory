"""Read temporal facts and add their observation and conflict context."""

from collections.abc import Sequence
from dataclasses import dataclass

from ontogr_mem.data_models.fact_versions import Fact
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.temporal.conflicts import annotate_views
from ontogr_mem.temporal.query import FactView, TemporalQuery, matches, view_of
from ontogr_mem.temporal.temporal_store import TemporalStore


@dataclass(frozen=True, slots=True)
class ObservationText:
    """Text and provenance carried by an observation.

    Attributes:
        statement: The observation's plain-language statement.
        quote: The supporting words from the source.
        episode_id: Episode that supplied the observation.
    """

    statement: str
    quote: str
    episode_id: str


class FactReader:
    """Apply temporal visibility and add evidence and conflict annotations.

    Args:
        store: Tenant-bound temporal store used for fact and evidence reads.
    """

    def __init__(self, store: TemporalStore) -> None:
        """Bind the reader to a temporal store."""
        self._store = store

    async def read(self, query: TemporalQuery) -> list[FactView]:
        """Return query-matching fact views with observation and conflict data."""
        return await self._annotate_views(await self._store.query(query))

    async def by_ids(self, ids: Sequence[str], query: TemporalQuery) -> list[FactView]:
        """Return visible versions of the requested fact ids, annotated."""
        facts = await self._store.facts_by_ids(ids)
        return await self.annotate([fact for fact in facts if matches(query, fact)])

    async def touching(
        self, entity_ids: Sequence[str], query: TemporalQuery
    ) -> list[FactView]:
        """Return visible fact versions that name a requested entity."""
        facts = await self._store.facts_of_entities(entity_ids)
        return await self.annotate([fact for fact in facts if matches(query, fact)])

    async def by_observations(
        self, obs_ids: Sequence[str], query: TemporalQuery
    ) -> list[FactView]:
        """Return visible versions supported by any requested observation."""
        observations = await self._store.observations_by_ids(obs_ids)
        wanted = {observation.obs_id for observation in observations}
        slots = sorted({observation.slot_key for observation in observations})
        facts = [
            fact
            for slot_key in slots
            for fact in (await self._store.load_slot(slot_key)).facts
            if wanted.intersection(fact.evidence) and matches(query, fact)
        ]
        facts.sort(key=lambda fact: (fact.fact_id, fact.version))
        return await self.annotate(facts)

    async def observation_texts(
        self, obs_ids: Sequence[str]
    ) -> dict[str, ObservationText]:
        """Return text and episode ids for stored observations, omitting misses."""
        observations = await self._store.observations_by_ids(obs_ids)
        return {
            observation.obs_id: ObservationText(
                statement=observation.statement,
                quote=observation.quote,
                episode_id=observation.episode_id,
            )
            for observation in observations
        }

    async def annotate(self, facts: Sequence[Fact]) -> list[FactView]:
        """Build views for facts and add their observation and conflict data."""
        return await self._annotate_views([view_of(fact) for fact in facts])

    async def _annotate_views(self, views: Sequence[FactView]) -> list[FactView]:
        """Annotate views with only the evidence and conflicts they reference."""
        if not views:
            return []

        slot_keys = sorted({view.slot_key for view in views})
        fact_ids = sorted({view.fact_id for view in views})
        obs_ids = sorted({obs_id for view in views for obs_id in view.evidence})
        rows: Sequence[Observation] = await self._store.observations_by_ids(obs_ids)
        observations = {observation.obs_id: observation for observation in rows}
        conflicts = await self._store.conflicts_for(slot_keys, fact_ids)
        decisions = await self._store.decisions_for(fact_ids)
        return list(annotate_views(views, observations, conflicts, decisions))
