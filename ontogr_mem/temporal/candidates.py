"""Which pairs of facts are worth asking about.

The choice is structural: two facts in different slots that share an entity.
The order is total and deterministic, the number of pairs is capped, and the
number of pairs cut is reported.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ontogr_mem.data_models.fact_versions import Fact
from ontogr_mem.temporal.cardinality import ConflictPolicy, DictRelationPolicy
from ontogr_mem.temporal.verdicts import pair_key
from ontogr_mem.temporal.visibility import in_current_view


@dataclass(frozen=True, slots=True)
class Candidate:
    """Two facts that share an entity, with the numbers that ranked them.

    Attributes:
        a: The fact that starts first, or the one with the earlier fact id.
        b: The other fact.
        pair_key: Order-independent key of the pair.
        shared: Number of entities both facts name.
        proximity: Seconds between the two starts.
    """

    a: Fact
    b: Fact
    pair_key: str
    shared: int
    proximity: float


def _ends(fact: Fact) -> frozenset[str]:
    return frozenset({fact.subject_id, fact.object_id})


def generate_candidates(
    facts: Sequence[Fact],
    policy: DictRelationPolicy,
    *,
    cap: int = 200,
) -> tuple[list[Candidate], int]:
    """Return the ranked candidate pairs, at most ``cap``, and how many were cut.

    Args:
        facts: Current fact versions to pair up. Facts that are out of the
            current view are left out.
        policy: Relation rules; relations with the ignore policy are left out.
        cap: Largest number of pairs to return.

    Returns:
        The pairs in rank order, and the number of pairs the cap removed.
    """
    usable = [
        fact
        for fact in facts
        if in_current_view(fact.status)
        and policy.rules(fact.relation_type_id).conflict_policy
        is not ConflictPolicy.IGNORE
    ]
    by_entity: dict[str, list[Fact]] = {}
    for fact in sorted(usable, key=lambda f: f.fact_id):
        for entity in sorted(_ends(fact)):
            by_entity.setdefault(entity, []).append(fact)
    seen: dict[str, Candidate] = {}
    for bucket in by_entity.values():
        for i, x in enumerate(bucket):
            for y in bucket[i + 1 :]:
                if x.slot_key == y.slot_key:
                    continue
                key = pair_key(x.fact_id, y.fact_id)
                if key in seen:
                    continue
                a, b = sorted((x, y), key=lambda f: (f.valid_from, f.fact_id))
                seen[key] = Candidate(
                    a=a,
                    b=b,
                    pair_key=key,
                    shared=len(_ends(a) & _ends(b)),
                    proximity=abs((b.valid_from - a.valid_from).total_seconds()),
                )
    ordered = sorted(
        seen.values(),
        key=lambda c: (-c.shared, c.proximity, c.pair_key),
    )
    return ordered[:cap], max(0, len(ordered) - cap)
