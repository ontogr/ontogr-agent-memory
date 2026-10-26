"""The memory of the judge and of people: fact verdicts and conflict decisions.

Both are inputs to reconciliation and are keyed on stable fact ids. A
``FactVerdict`` is valid only for the exact facts it saw: a change in either fact
changes ``inputs_hash``, so a stored answer goes stale instead of being frozen,
and an unchanged pair is never asked twice.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ontogr_mem.data_models.ids import stable_id


class VerdictKind(StrEnum):
    """One answer per pair, so a pair can never be both duplicate and contradiction."""

    CONTRADICTS = "contradicts"
    COMPATIBLE = "compatible"
    REFINES = "refines"
    DUPLICATE = "duplicate"
    UNRELATED = "unrelated"
    UNCERTAIN = "uncertain"


NEGATIVE_VERDICTS = frozenset(
    {
        VerdictKind.COMPATIBLE,
        VerdictKind.REFINES,
        VerdictKind.DUPLICATE,
        VerdictKind.UNRELATED,
    }
)
"""Answers that are stored so the pair is not asked again."""


class VerdictSource(StrEnum):
    """Who produced a verdict or a decision."""

    RULE = "rule"
    LLM = "llm"
    USER = "user"


RULE_PROMPT_VERSION = "rule-1"
"""Stands in for a prompt version on verdicts the rules produce.

Rule verdicts and judge verdicts for the same pair then have different ids and
can both be stored.
"""


@dataclass(frozen=True, slots=True)
class FactFingerprint:
    """What a verdict about a fact depends on.

    A verdict is valid only for the facts it saw. The fingerprint names the
    identity of a fact, the slot it lives in and the statement it rests on. It
    leaves out the end of the interval. The end is the effect of a verdict, so
    including it would make the verdict invalidate itself.

    Attributes:
        slot_key: Slot the fact belongs to.
        subject_id: Entity on the subject side, empty when the fact has no
            observation to name it.
        relation_type_id: Relation type, ``None`` when the fact has no
            observation to name it.
        object_id: Entity on the object side, empty when the fact has no
            observation to name it.
        valid_from: Start of the fact's valid interval, as text.
        statement: The first statement that supports the fact, empty when the
            fact has no observation to name it.
    """

    slot_key: str
    subject_id: str = ""
    relation_type_id: int | None = None
    object_id: str = ""
    valid_from: str = ""
    statement: str = ""


def fingerprint_fields(fingerprint: FactFingerprint) -> tuple[str, ...]:
    """Return the fields of a fingerprint as text, in a fixed order.

    The order is the hash's, so two fingerprints of the same facts always give
    the same text.
    """
    relation = (
        ""
        if fingerprint.relation_type_id is None
        else str(fingerprint.relation_type_id)
    )
    return (
        fingerprint.slot_key,
        fingerprint.subject_id,
        relation,
        fingerprint.object_id,
        fingerprint.valid_from,
        fingerprint.statement,
    )


def inputs_hash(a: FactFingerprint, b: FactFingerprint) -> str:
    """Hash everything a verdict depended on, independent of side order.

    When either fact's interval or statement moves, the stored verdict no longer
    describes these facts.
    """
    first, second = sorted((a, b), key=fingerprint_fields)
    return stable_id(
        "inputs", *fingerprint_fields(first), "|", *fingerprint_fields(second)
    )


def pair_key(fact_a_id: str, fact_b_id: str) -> str:
    """Return the order-independent key of a pair of facts."""
    first, second = sorted((fact_a_id, fact_b_id))
    return stable_id("pair", first, second)


@dataclass(frozen=True, slots=True)
class FactVerdict:
    """What was decided about two facts, by whom, from what, and how sure.

    Attributes:
        pair_key: Order-independent key of the two facts.
        fact_a_id: One fact of the pair.
        fact_b_id: The other fact of the pair.
        verdict: The answer.
        inputs_hash: Hash of the facts as they were when judged.
        prompt_version: Version of the prompt or rule set that produced it.
        source: Rule, model or person.
        confidence: Model confidence, 1.0 for rules and people.
        kind: Kind of disagreement the model named.
        reason_code: Short rule or reason label.
        reason_text: Plain explanation.
        quote_a: Words of the first statement that support the answer.
        quote_b: Words of the second statement that support the answer.
        model: Model that answered, if any.
        votes: ``(pass, answer)`` for each pass of the judge.
        decided_at: When the verdict was made.
        source_episode_ids: Episodes the two facts came from.
    """

    pair_key: str
    fact_a_id: str
    fact_b_id: str
    verdict: VerdictKind
    inputs_hash: str
    prompt_version: str
    source: VerdictSource
    confidence: float = 1.0
    kind: str = ""
    reason_code: str = ""
    reason_text: str = ""
    quote_a: str = ""
    quote_b: str = ""
    model: str = ""
    votes: tuple[tuple[str, str], ...] = ()
    decided_at: datetime | None = None
    source_episode_ids: tuple[str, ...] = ()

    @property
    def verdict_id(self) -> str:
        """Identity: the pair, its inputs and the prompt that judged it."""
        return stable_id(
            "verdict", self.pair_key, self.inputs_hash, self.prompt_version
        )

    def valid_for(self, inputs: str, prompt_version: str) -> bool:
        """Return whether the verdict still describes these facts and prompt."""
        return self.inputs_hash == inputs and self.prompt_version == prompt_version


class DecisionAction(StrEnum):
    """What a person, or a rule, decided about a conflict."""

    CHOOSE = "choose"
    MARK_WRONG = "mark_wrong"
    MARK_BOTH_VALID = "mark_both_valid"
    MARK_COMPATIBLE = "mark_compatible"


@dataclass(frozen=True, slots=True)
class ConflictDecision:
    """A durable decision about a conflict.

    Commits never delete a decision, and reconciliation reads it on every run.

    Attributes:
        decision_id: Stable hash of the decision's inputs; a rerun upserts it.
        conflict_id: Conflict the decision answers.
        fact_ids: The facts the decision is about.
        action: What was decided.
        decided_by: Who decided.
        at: When the decision was made.
        chosen_fact_id: Fact kept by ``CHOOSE``.
        wrong_fact_id: Fact marked wrong by ``MARK_WRONG``.
        actor: Person or system that made the decision.
        reason: Plain explanation.
        trust_inputs: Trust ranks seen when the decision was made.
        carried_from: Retired fact id this decision was carried over from.
    """

    decision_id: str
    conflict_id: str
    fact_ids: tuple[str, ...]
    action: DecisionAction
    decided_by: VerdictSource
    at: datetime
    chosen_fact_id: str = ""
    wrong_fact_id: str = ""
    actor: str = ""
    reason: str = ""
    trust_inputs: tuple[tuple[str, int], ...] = ()
    carried_from: str = ""


def latest_decision(
    decisions: Iterable[ConflictDecision],
    fact_ids: Iterable[str],
) -> ConflictDecision | None:
    """Return the decision that applies to a set of facts.

    A decision applies when it names every one of ``fact_ids``. A decision
    about more facts still applies to these: a person who settled three facts
    at once settled these as well. The newest decision wins, and a tie is
    broken by the decision id.

    Args:
        decisions: The decisions to choose from.
        fact_ids: The facts the decision has to cover.

    Returns:
        The decision that applies, or ``None`` when no decision covers them.
    """
    wanted = set(fact_ids)
    if not wanted:
        return None
    matching = [decision for decision in decisions if wanted <= set(decision.fact_ids)]
    if not matching:
        return None
    return max(matching, key=lambda decision: (decision.at, decision.decision_id))


def conflict_id_for(key: str) -> str:
    """Return the deterministic id of the conflict over a pair key."""
    return stable_id("conflict", key)


def decision_id_for(
    conflict_id: str,
    action: DecisionAction,
    decided_by: VerdictSource,
    at: datetime,
    *,
    chosen_fact_id: str = "",
    wrong_fact_id: str = "",
    actor: str = "",
) -> str:
    """Return the stable id of a decision built from its inputs."""
    return stable_id(
        "decision",
        conflict_id,
        action.value,
        decided_by.value,
        at.isoformat(),
        chosen_fact_id,
        wrong_fact_id,
        actor,
    )
