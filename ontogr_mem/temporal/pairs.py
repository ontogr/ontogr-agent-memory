"""The judge's view of a pair of facts.

A pair is two facts that may disagree. The judge sees the statement each fact
rests on, the interval it holds in, and a hash of both facts as they were when
they were shown. That hash is the cache key: a stored verdict is reused only
while both facts still hash the same.
"""

from dataclasses import dataclass

from ontogr_mem.data_models.fact_versions import Fact
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.temporal.handler import Partner, fact_side
from ontogr_mem.temporal.trust import Trust
from ontogr_mem.temporal.verdicts import inputs_hash


@dataclass(frozen=True, slots=True)
class JudgeFact:
    """One side of a pair as the prompt shows it.

    Attributes:
        fact_id: Stable id of the fact.
        statement: The first statement that supports the fact.
        interval: The valid interval as text.
        source: Source system of that statement.
    """

    fact_id: str
    statement: str
    interval: str
    source: str


@dataclass(frozen=True, slots=True)
class PairInput:
    """Everything the judge needs about one candidate pair.

    Attributes:
        pair_key: Order-independent key of the pair.
        inputs_hash: Hash of both facts as shown; the verdict cache key.
        a: First fact as shown.
        b: Second fact as shown.
        episode_ids: Episodes the two facts came from.
    """

    pair_key: str
    inputs_hash: str
    a: JudgeFact
    b: JudgeFact
    episode_ids: tuple[str, ...] = ()

    def swapped(self) -> "PairInput":
        """Return the same pair with the sides exchanged."""
        return PairInput(
            self.pair_key, self.inputs_hash, self.b, self.a, self.episode_ids
        )


def _interval(fact: Fact) -> str:
    """Return the valid interval as the judge reads it, e.g. ``2010-01-01 to open``."""
    end = fact.valid_to.date().isoformat() if fact.valid_to else "open"
    return f"{fact.valid_from.date().isoformat()} to {end}"


def pair_input(
    pair_key: str,
    left: Partner,
    right: Partner,
    trust: Trust,
) -> PairInput:
    """Build the judge's view of a pair: statements, intervals and the cache key.

    The statement shown is the fact's first observation. When it has none, the
    id triple stands in. ``inputs_hash`` is the cache key: a stored verdict is
    reused only while both facts still hash the same.
    """
    sides = []
    views = []
    episodes: set[str] = set()
    for partner in (left, right):
        fact, observations = partner.fact, partner.observations
        side = fact_side(fact, observations, trust)
        first: Observation | None = observations.get(side.first_obs_id)
        sides.append(side)
        views.append(
            JudgeFact(
                fact_id=fact.fact_id,
                statement=first.statement
                if first and first.statement
                else f"{fact.subject_id} {fact.relation_type_id} {fact.object_id}",
                interval=_interval(fact),
                source=first.source_system if first else "",
            )
        )
        episodes.update(
            observations[i].episode_id for i in side.obs_ids if i in observations
        )
    return PairInput(
        pair_key,
        inputs_hash(sides[0].fp, sides[1].fp),
        views[0],
        views[1],
        tuple(sorted(episodes)),
    )
