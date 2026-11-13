"""The typed model judge and its two-pass contradiction check.

The judge gives one answer per pair: contradicts, compatible, refines,
duplicate, unrelated or uncertain. A contradiction that would change state needs
a second pass with the sides swapped, which must agree with a confidence at or
above the floor. Otherwise the pair is stored as uncertain, a visible review
item and not a dropped one. A failed pair is counted, never recorded as "no
contradiction".

The prompt lives in ``ontogr_mem.temporal.prompt`` and the typed output in
``ontogr_mem.temporal.judge_agent``. Pair membership remains a domain check here.
"""

import asyncio
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from ontogr_mem.llm import (
    ModelFailure,
    ModelOutputError,
    ModelRole,
    RoleModels,
    TokenUsage,
    run_agent,
)
from ontogr_mem.llm.tokens import estimate_tokens
from ontogr_mem.temporal.answers import Answer, unswap
from ontogr_mem.temporal.judge_agent import judge_agent
from ontogr_mem.temporal.pairs import (  # noqa: F401
    JudgeFact,
    PairInput,
    pair_input,
)
from ontogr_mem.temporal.prompt import build_prompt
from ontogr_mem.temporal.report import Outcome
from ontogr_mem.temporal.settings import JudgeSettings
from ontogr_mem.temporal.verdicts import (
    FactVerdict,
    VerdictKind,
    VerdictSource,
)


@dataclass(slots=True)
class JudgeOutcome:
    """What a judging call produced. Failures are counted, never swallowed.

    Attributes:
        verdicts: One verdict per answered pair.
        counts: Failure and invalid-answer counts, by ``Outcome`` value.
        tokens_in: Input tokens the calls used.
        tokens_out: Output tokens the calls used.
        requests: Model requests, including output-validation retries.
    """

    verdicts: list[FactVerdict] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)
    tokens_in: int = 0
    tokens_out: int = 0
    requests: int = 0


class PairJudge(Protocol):
    """Anything that turns pairs into verdicts."""

    prompt_version: str

    async def judge_pairs(
        self, pairs: Sequence[PairInput], *, now: datetime
    ) -> JudgeOutcome:
        """Judge ``pairs``. Only answered pairs produce a verdict."""
        ...


class NullJudge:
    """Judges nothing: pairs stay unjudged and many-valued slots never close."""

    prompt_version = "null"

    async def judge_pairs(
        self, pairs: Sequence[PairInput], *, now: datetime
    ) -> JudgeOutcome:
        """Return an empty outcome."""
        return JudgeOutcome()


class LLMPairJudge:
    """The judge over the configured model role.

    Args:
        models: Shared role models and concurrency limit.
        settings: Contradiction floor, prompt version, and batch limits.
    """

    def __init__(self, models: RoleModels, settings: JudgeSettings) -> None:
        """Bind the judge to its model pool and settings."""
        self._models = models
        self._settings = settings

    @property
    def prompt_version(self) -> str:
        """Version recorded with every verdict this judge produces."""
        return self._settings.prompt_version

    def _batches(self, pairs: Sequence[PairInput]) -> list[list[PairInput]]:
        """Split pairs into calls bounded by pair count and estimated tokens."""
        out: list[list[PairInput]] = []
        current: list[PairInput] = []
        size = 0
        for pair in pairs:
            cost = estimate_tokens(build_prompt([pair]))
            if current and (
                len(current) >= self._settings.max_pairs
                or size + cost > self._settings.token_budget
            ):
                out.append(current)
                current, size = [], 0
            current.append(pair)
            size += cost
        if current:
            out.append(current)
        return out

    async def _ask(
        self, pairs: Sequence[PairInput], swapped: bool, outcome: JudgeOutcome
    ) -> dict[str, Answer]:
        """Ask one pass. Return the answers by pair key, for answered pairs."""
        answered: dict[str, Answer] = {}

        def collect_usage(usage: TokenUsage | None) -> None:
            if usage is not None:
                outcome.tokens_in += usage.input_tokens
                outcome.tokens_out += usage.output_tokens
                outcome.requests += usage.requests

        async def ask_batch(batch: list[PairInput]) -> None:
            """Ask one independent batch through the shared model pool."""
            shown = [p.swapped() if swapped else p for p in batch]
            try:
                result = await run_agent(
                    judge_agent,
                    ModelRole.JUDGE,
                    build_prompt(shown),
                    models=self._models,
                    deps=None,
                )
            except ModelOutputError as exc:
                collect_usage(exc.usage)
                outcome.counts[Outcome.JUDGE_PARSE_FAILED.value] += len(batch)
                return
            except ModelFailure as exc:
                collect_usage(exc.usage)
                outcome.counts[Outcome.JUDGE_LLM_ERROR.value] += len(batch)
                return
            collect_usage(result.usage)
            found: dict[int, Answer] = {}
            for item in result.output.pairs:
                match = re.fullmatch(r"P([0-9]+)", item.id)
                index = int(match.group(1)) - 1 if match else -1
                if not 0 <= index < len(batch):
                    outcome.counts[Outcome.JUDGE_INVALID_ID.value] += 1
                    continue
                found[index] = Answer(
                    verdict=item.verdict,
                    confidence=item.confidence,
                    kind=item.kind,
                    rule=item.rule,
                    reason=item.reason,
                    quote_a=item.quote_a,
                    quote_b=item.quote_b,
                )
            for index, answer in found.items():
                answered[batch[index].pair_key] = unswap(answer) if swapped else answer

        await asyncio.gather(*(ask_batch(batch) for batch in self._batches(pairs)))
        return answered

    async def judge_pairs(
        self, pairs: Sequence[PairInput], *, now: datetime
    ) -> JudgeOutcome:
        """Judge the pairs and confirm above-floor contradictions.

        Returns:
            One verdict per answered pair. A pair left unanswered or unconfirmed
            is counted as failed and is not stored.
        """
        outcome = JudgeOutcome()
        first = await self._ask(pairs, False, outcome)
        outcome.counts[Outcome.JUDGE_FAILED.value] += sum(
            1 for p in pairs if p.pair_key not in first
        )
        floor = self._settings.floor
        confirm = [
            p
            for p in pairs
            if p.pair_key in first
            and first[p.pair_key].verdict is VerdictKind.CONTRADICTS
            and first[p.pair_key].confidence >= floor
        ]
        second = await self._ask(confirm, True, outcome) if confirm else {}
        for pair in pairs:
            one = first.get(pair.pair_key)
            if one is None:
                continue
            verdict, confidence = one.verdict, one.confidence
            votes = [("p1", one.verdict.value)]
            if one.verdict is VerdictKind.CONTRADICTS:
                if one.confidence < floor:
                    verdict = VerdictKind.UNCERTAIN
                else:
                    two = second.get(pair.pair_key)
                    if two is None:
                        outcome.counts[Outcome.JUDGE_FAILED.value] += 1
                        continue
                    votes.append(("p2", two.verdict.value))
                    if (
                        two.verdict is VerdictKind.CONTRADICTS
                        and two.confidence >= floor
                    ):
                        confidence = min(one.confidence, two.confidence)
                    else:
                        verdict = VerdictKind.UNCERTAIN
            outcome.verdicts.append(
                FactVerdict(
                    pair_key=pair.pair_key,
                    fact_a_id=pair.a.fact_id,
                    fact_b_id=pair.b.fact_id,
                    verdict=verdict,
                    inputs_hash=pair.inputs_hash,
                    prompt_version=self.prompt_version,
                    source=VerdictSource.LLM,
                    confidence=confidence,
                    kind=one.kind,
                    reason_code=one.rule,
                    reason_text=one.reason,
                    quote_a=one.quote_a,
                    quote_b=one.quote_b,
                    votes=tuple(votes),
                    decided_at=now,
                    source_episode_ids=pair.episode_ids,
                )
            )
        return outcome
