"""Source trust: a ranked rule set resolved from configuration at reconcile time.

Observations carry their source attributes (system, connector, actor role), and
the rank is looked up here. Changing the rules or the margin re-ranks on the next
reconcile without rewriting data. Trust never orders facts in time. With no
rules every rank is 0, so every concurrent disagreement stays a dispute.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from ontogr_mem.data_models.observation import Observation


@dataclass(frozen=True, slots=True)
class TrustRule:
    """Matches observations by the fields it names.

    An empty field matches every observation.

    Attributes:
        rank: Trust rank the rule grants.
        system: Source system to match, or empty.
        connector: Connector to match, or empty.
        role: Actor role to match, or empty.
    """

    rank: int
    system: str = ""
    connector: str = ""
    role: str = ""

    def matches(self, observation: Observation) -> bool:
        """Return whether every field the rule names equals the observation's."""
        return (
            (not self.system or self.system == observation.source_system)
            and (not self.connector or self.connector == observation.connector)
            and (not self.role or self.role == observation.actor_role)
        )


@dataclass(frozen=True, slots=True)
class Trust:
    """Trust rules and the smallest rank gap that permits a preference.

    Attributes:
        rules: Rules that grant ranks.
        margin: Smallest gap between two ranks that lets the higher one win
            automatically. At least 1.
    """

    rules: tuple[TrustRule, ...] = ()
    margin: int = 1

    def __post_init__(self) -> None:
        """Reject a margin that would let a tie count as a preference."""
        if self.margin < 1:
            msg = "margin must be at least 1"
            raise ValueError(msg)

    def rank_of(self, observation: Observation) -> int:
        """Return the highest rank among matching rules, or 0 when none match."""
        return max(
            (rule.rank for rule in self.rules if rule.matches(observation)),
            default=0,
        )

    def fact_rank(self, observations: Iterable[Observation]) -> int:
        """Return a fact's trust: its best observation, so corroboration counts."""
        return max((self.rank_of(o) for o in observations), default=0)
