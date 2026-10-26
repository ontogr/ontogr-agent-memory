"""Tests for the contradiction handler on one slot, with no store.

Each test builds the slot's observations, the stored facts and any verdicts or
decisions, then checks the directives, conflicts and rule verdicts it returns.
"""

from datetime import UTC, datetime

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.data_models.facts import Precision
from ontogr_mem.data_models.observation import ObservationKind
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    ConflictPolicy,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.classify import ConflictClass, Disposition
from ontogr_mem.temporal.handler import (
    HandlerInputs,
    HandlerResult,
    Partner,
    judge_pending_slots,
    slot_directives,
)
from ontogr_mem.temporal.judge import pair_input
from ontogr_mem.temporal.reconcile import Close
from ontogr_mem.temporal.trust import Trust
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    DecisionAction,
    FactVerdict,
    VerdictKind,
    VerdictSource,
    pair_key,
)
from tests.unit.temporal.helpers import at, fact_of, obs


NOW = datetime(2026, 4, 1, tzinfo=UTC)
ONE_VALUED = RelationRules(Cardinality.ONE, ConflictPolicy.REVIEW)
MANY_VALUED = RelationRules(Cardinality.MANY, ConflictPolicy.REVIEW)
POLICY = DictRelationPolicy({1: ONE_VALUED, 2: MANY_VALUED, 3: MANY_VALUED})


def run_handler(
    observations,
    facts=(),
    *,
    partners=None,
    verdicts=(),
    decisions=(),
    policy=POLICY,
) -> HandlerResult:
    """Run the handler on slot ``slot`` with the given inputs."""
    return slot_directives(
        HandlerInputs(
            slot_key="slot",
            observations=list(observations),
            facts=list(facts),
            policy=policy,
            trust=Trust(),
            now=NOW,
            verdicts=list(verdicts),
            decisions=list(decisions),
            partners=partners or {},
        )
    )


def contradiction(
    fact_a: str, fact_b: str, verdict_inputs: str, *, decided_at=None
) -> FactVerdict:
    """A stored verdict that two facts contradict each other."""
    return FactVerdict(
        pair_key=pair_key(fact_a, fact_b),
        fact_a_id=fact_a,
        fact_b_id=fact_b,
        verdict=VerdictKind.CONTRADICTS,
        inputs_hash=verdict_inputs,
        prompt_version="judge-1",
        source=VerdictSource.LLM,
        decided_at=decided_at,
    )


class TestSlotDirectives:
    """What the handler tells the walk about one slot."""

    def test_later_value_in_a_single_valued_slot_is_an_update_not_a_conflict(self):
        """A newer value from an equally trusted source closes the old one."""
        acme = obs("o1", "acme", at(2020))
        globex = obs("o2", "globex", at(2022))

        result = run_handler([acme, globex])

        assert result.conflicts == ()
        assert result.directives.excluded == {}
        assert [v.source for v in result.rule_verdicts] == [VerdictSource.RULE]

    def test_equal_start_values_are_kept_side_by_side_without_a_second_conflict(self):
        """A tie coexists; the walk reports it, so the handler does not repeat it."""
        acme = obs("o1", "acme", at(2020))
        globex = obs("o2", "globex", at(2020))

        result = run_handler([acme, globex])

        assert result.directives.no_cut == (frozenset({"o1", "o2"}),)
        assert result.conflicts == ()

    def test_a_correction_excludes_the_wrong_fact_and_reports_the_pair(self):
        """The corrected statement is excluded with status CORRECTED."""
        acme = obs("o1", "acme", at(2020))
        correction = obs("o2", "globex", at(2022), kind=ObservationKind.CORRECTION)

        result = run_handler([acme, correction])

        assert result.directives.excluded == {"o1": FactStatus.CORRECTED}
        assert [(c.cls, c.disposition) for c in result.conflicts] == [
            (ConflictClass.CORRECTION, Disposition.CORRECT)
        ]

    def test_a_judged_contradiction_closes_the_earlier_fact_at_the_later_start(self):
        """A verdict across slots ends the local fact where the other one starts."""
        mine = obs("m1", "acme", at(2010))
        other = obs("p1", "globex", at(2015), slot="other", rel=2)
        partner = Partner(fact_of("f-p", [other], slot="other"), {"p1": other})
        stored = pair_input(
            pair_key("f-m", "f-p"),
            Partner(fact_of("f-m", [mine]), {"m1": mine}),
            partner,
            Trust(),
        )
        verdict = contradiction("f-m", "f-p", stored.inputs_hash)

        result = run_handler(
            [mine],
            [fact_of("f-m", [mine])],
            partners={"f-p": partner},
            verdicts=[verdict],
        )

        assert result.directives.closes == (
            Close(
                anchor_obs_id="m1",
                at=at(2015),
                at_precision=Precision.YEAR,
                cause_id=verdict.verdict_id,
            ),
        )

    def test_a_person_choice_overrides_the_judged_contradiction(self):
        """A CHOOSE decision keeps the chosen fact and closes nothing."""
        mine = obs("m1", "acme", at(2010))
        other = obs("p1", "globex", at(2015), slot="other", rel=2)
        partner = Partner(fact_of("f-p", [other], slot="other"), {"p1": other})
        stored = pair_input(
            pair_key("f-m", "f-p"),
            Partner(fact_of("f-m", [mine]), {"m1": mine}),
            partner,
            Trust(),
        )
        choice = ConflictDecision(
            decision_id="d1",
            conflict_id="c1",
            fact_ids=("f-m", "f-p"),
            action=DecisionAction.CHOOSE,
            decided_by=VerdictSource.USER,
            at=at(2024),
            chosen_fact_id="f-m",
        )

        result = run_handler(
            [mine],
            [fact_of("f-m", [mine])],
            partners={"f-p": partner},
            verdicts=[contradiction("f-m", "f-p", stored.inputs_hash)],
            decisions=[choice],
        )

        assert result.directives.closes == ()
        assert [(c.cls, c.disposition) for c in result.conflicts] == [
            (ConflictClass.DISPUTE, Disposition.PREFER)
        ]

    def test_a_decision_about_a_larger_fact_set_covers_the_pair(self):
        """A decision naming both facts still applies when it names another fact."""
        mine = obs("m1", "acme", at(2010))
        other = obs("p1", "globex", at(2015), slot="other", rel=2)
        partner = Partner(fact_of("f-p", [other], slot="other"), {"p1": other})
        stored = pair_input(
            pair_key("f-m", "f-p"),
            Partner(fact_of("f-m", [mine]), {"m1": mine}),
            partner,
            Trust(),
        )
        choice = ConflictDecision(
            decision_id="d1",
            conflict_id="c1",
            fact_ids=("f-m", "f-p", "f-extra"),
            action=DecisionAction.CHOOSE,
            decided_by=VerdictSource.USER,
            at=at(2024),
            chosen_fact_id="f-m",
        )

        result = run_handler(
            [mine],
            [fact_of("f-m", [mine])],
            partners={"f-p": partner},
            verdicts=[contradiction("f-m", "f-p", stored.inputs_hash)],
            decisions=[choice],
        )

        assert result.directives.closes == ()
        assert [(item.cls, item.disposition) for item in result.conflicts] == [
            (ConflictClass.DISPUTE, Disposition.PREFER)
        ]

    def test_a_verdict_for_other_inputs_is_ignored(self):
        """A stored verdict whose inputs hash changed says nothing about today."""
        mine = obs("m1", "acme", at(2010))
        other = obs("p1", "globex", at(2015), slot="other", rel=2)
        partner = Partner(fact_of("f-p", [other], slot="other"), {"p1": other})

        result = run_handler(
            [mine],
            [fact_of("f-m", [mine])],
            partners={"f-p": partner},
            verdicts=[contradiction("f-m", "f-p", "changed-inputs")],
        )

        assert result.directives.closes == ()
        assert result.conflicts == ()

    def test_a_partner_without_a_verdict_leaves_the_slot_judge_pending(self):
        """A slot with an unjudged neighbour is judge-pending."""
        mine = fact_of("f-m", [obs("m1", "acme", at(2010))])
        other = fact_of(
            "f-p", [obs("p1", "globex", at(2015), slot="other", rel=2)], slot="other"
        )

        pending = judge_pending_slots([mine, other], [])
        judged = judge_pending_slots(
            [mine, other], [contradiction("f-m", "f-p", "any")]
        )

        assert pending == {"slot", "other"}
        assert judged == set()

    def test_facts_of_one_slot_do_not_make_each_other_pending(self):
        """Two facts of one slot are settled by the rules of that slot."""
        first = fact_of("f-1", [obs("m1", "acme", at(2010))])
        second = fact_of("f-2", [obs("m2", "globex", at(2012))])

        assert judge_pending_slots([first, second], []) == set()
