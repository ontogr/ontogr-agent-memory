"""Derive pending slots through the temporal storage port."""

from ontogr_mem.temporal.handler import judge_pending_slots
from ontogr_mem.temporal.temporal_store import TemporalStore


async def pending_slots(store: TemporalStore) -> list[str]:
    """Return sorted slots with current facts awaiting a neighbour verdict."""
    facts = await store.current_facts()
    verdicts = await store.verdicts_for([fact.fact_id for fact in facts])
    return sorted(judge_pending_slots(facts, verdicts))
