"""Collect the candidate entities of one group.

Two sources propose candidates. The blocking index finds entities whose key
resembles the group's key, and the vector search finds entities whose
embedding resembles the group's. Each candidate keeps the signals that
proposed it, so the ranking can weigh each signal against its own floor.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ontogr_mem.resolution.blocking import STRIPPED_SUFFIX, BlockIndex
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.grouping import Group, group_text
from ontogr_mem.resolution.keys import stripped_key
from ontogr_mem.resolution.policy import ResolutionPolicy
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.resolution.scoring import Candidate, PairSignals, rank_key
from ontogr_mem.vectorstore import VectorKind, VectorStore
from ontogr_mem.vectorstore.text import normalise_text


CANDIDATE_CAP = 10
"""Most candidates one group keeps after ranking."""


@dataclass(slots=True)
class _Cand:
    entity_id: str
    jaccard: float | None = None
    cosine: float | None = None


async def candidate_pass(
    groups: Sequence[Group],
    *,
    store: ResolverStore,
    vectors: VectorStore,
    policy: ResolutionPolicy,
    settings: ResolverSettings,
) -> tuple[list[list[Candidate]], bool]:
    """Return candidates per group and whether some group could not be embedded."""
    rows = await store.blocking_rows(include_archived=True)
    index = BlockIndex.from_rows(rows)
    if not rows:
        return [[] for _ in groups], False
    texts = [normalise_text(group_text(group)) for group in groups]
    outcomes = await vectors.gateway.embed(texts)
    incomplete = any(outcome.vector is None for outcome in outcomes)
    out: list[list[Candidate]] = []
    for group, outcome in zip(groups, outcomes, strict=True):
        found: dict[str, _Cand] = {}
        keys = [group.loose]
        if policy.strip_suffixes(group.type_label):
            keys.append(stripped_key(group.name))
        for key in keys:
            for item in index.candidates(key):
                entity_id = item.split("#")[0]
                candidate = found.setdefault(entity_id, _Cand(entity_id))
                if not item.endswith(STRIPPED_SUFFIX):
                    candidate.jaccard = max(
                        candidate.jaccard or 0.0,
                        index.similarity(item, group.loose),
                    )
        for candidate in found.values():
            candidate.jaccard = max(
                candidate.jaccard or 0.0,
                index.similarity(candidate.entity_id, group.loose),
            )
        if outcome.vector is not None:
            hits = await vectors.search(
                VectorKind.NODE,
                outcome.vector,
                k=settings.ann_top_k,
                min_score=settings.ann_floor,
            )
            for hit in hits:
                candidate = found.setdefault(hit.id, _Cand(hit.id))
                candidate.cosine = max(candidate.cosine or 0.0, hit.score)
        ranked = [
            Candidate(
                entity_id=candidate.entity_id,
                signals=PairSignals(
                    cosine=candidate.cosine,
                    jaccard=candidate.jaccard,
                ),
            )
            for candidate in found.values()
        ]
        ranked.sort(key=lambda candidate: rank_key(candidate, settings))
        out.append(ranked[:CANDIDATE_CAP])
    return out, incomplete
