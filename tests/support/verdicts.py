"""Builders for verdict fixtures used by temporal tests."""

from datetime import UTC, datetime

from ontogr_mem.data_models.fact_versions import Fact
from ontogr_mem.temporal.verdicts import (
    FactFingerprint,
    FactVerdict,
    VerdictKind,
    VerdictSource,
    inputs_hash,
    pair_key,
)


def verdict(a: Fact, b: Fact, kind: VerdictKind) -> FactVerdict:
    """Build an LLM verdict for the current fingerprints of two facts."""
    return FactVerdict(
        pair_key=pair_key(a.fact_id, b.fact_id),
        fact_a_id=a.fact_id,
        fact_b_id=b.fact_id,
        verdict=kind,
        inputs_hash=inputs_hash(
            FactFingerprint(
                slot_key=a.slot_key,
                subject_id=a.subject_id,
                relation_type_id=a.relation_type_id,
                object_id=a.object_id,
                valid_from=a.valid_from.isoformat(),
            ),
            FactFingerprint(
                slot_key=b.slot_key,
                subject_id=b.subject_id,
                relation_type_id=b.relation_type_id,
                object_id=b.object_id,
                valid_from=b.valid_from.isoformat(),
            ),
        ),
        prompt_version="closure-1",
        source=VerdictSource.LLM,
        decided_at=datetime.now(UTC),
    )
