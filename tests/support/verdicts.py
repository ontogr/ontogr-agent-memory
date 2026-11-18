"""Builders for verdict fixtures used by temporal tests."""

from collections.abc import Callable
from datetime import UTC, datetime

from ontogr_mem.data_models.fact_versions import Fact
from ontogr_mem.temporal.handler import fact_side
from ontogr_mem.temporal.temporal_store import TemporalStore
from ontogr_mem.temporal.trust import Trust
from ontogr_mem.temporal.verdicts import (
    FactVerdict,
    VerdictKind,
    VerdictSource,
    inputs_hash,
    pair_key,
)


async def verdict(
    store: TemporalStore,
    a: Fact,
    b: Fact,
    kind: VerdictKind,
    now: Callable[[], datetime],
) -> FactVerdict:
    """Build a verdict from the facts' stored first observations."""
    observations_a = {
        observation.obs_id: observation
        for observation in await store.observations_of_slot(a.slot_key)
    }
    observations_b = {
        observation.obs_id: observation
        for observation in await store.observations_of_slot(b.slot_key)
    }
    first = fact_side(a, observations_a, Trust())
    second = fact_side(b, observations_b, Trust())
    return FactVerdict(
        pair_key=pair_key(a.fact_id, b.fact_id),
        fact_a_id=a.fact_id,
        fact_b_id=b.fact_id,
        verdict=kind,
        inputs_hash=inputs_hash(first.fp, second.fp),
        prompt_version="closure-1",
        source=VerdictSource.LLM,
        decided_at=now().astimezone(UTC),
    )
