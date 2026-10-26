"""Batched, vote-checked verification of entity pairs.

A pair that needs two votes is settled by a YES only when a second vote, with
the sides swapped, agrees with it. A pair that no rule can settle stays open,
and a reply that cannot be read is never remembered as a verdict.
"""

import asyncio
from collections.abc import Sequence
from enum import StrEnum

from ontogr_mem.data_models.entities import (
    VerdictKind,
    VerifyStatus,
)
from ontogr_mem.data_models.extraction import Usage
from ontogr_mem.llm import (
    LLMClient,
    LLMError,
    LLMRequest,
    ModelSize,
    estimate_tokens,
)
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.pairs import PairInput, PairOutcome, VerifyStats
from ontogr_mem.resolution.verifier_prompt import (
    SYSTEM,
    flatten,  # noqa: F401
    pair_block,
)
from ontogr_mem.resolution.verifier_reply import parse_reply


class PassResult(StrEnum):
    """What one pair answered in one verification pass."""

    YES = "yes"
    NO = "no"
    NO_ANSWER = "no_answer"
    NO_VERDICT = "no_verdict"
    ERROR = "error"


class Verifier:
    """Batched, vote-checked pair verification over an injected ``LLMClient``."""

    def __init__(
        self, client: LLMClient, settings: ResolverSettings | None = None
    ) -> None:
        """Create a verifier. Wrap ``client`` in an ``LLMGate`` to cap calls."""
        self._client = client
        self._settings = settings or ResolverSettings()

    def _windows(self, pairs: Sequence[PairInput], *, swap: bool) -> list[list[int]]:
        windows: list[list[int]] = []
        cur: list[int] = []
        used = estimate_tokens(SYSTEM)
        for idx, p in enumerate(pairs):
            cost = estimate_tokens(pair_block(len(cur) + 1, p, swap))
            full = len(cur) >= self._settings.pairs_per_call or (
                cur and used + cost > self._settings.pair_token_budget
            )
            if full:
                windows.append(cur)
                cur, used = [], estimate_tokens(SYSTEM)
            cur.append(idx)
            used += cost
        if cur:
            windows.append(cur)
        return windows

    async def _ask(
        self,
        pairs: Sequence[PairInput],
        indices: list[int],
        swap: bool,
        usage: Usage,
        stats: VerifyStats,
    ) -> dict[int, PassResult] | None:
        """Ask one window. ``None`` means the call failed."""
        body = "\n".join(
            pair_block(i + 1, pairs[j], swap) for i, j in enumerate(indices)
        )
        req = LLMRequest(
            system=SYSTEM,
            prompt=body,
            model_size=ModelSize.MEDIUM,
            max_output_tokens=120 * len(indices) + 64,
        )
        stats.calls += 1
        try:
            resp = await self._client.complete(req)
        except (LLMError, TimeoutError):
            return None
        usage.add(
            "resolve_verify",
            ModelSize.MEDIUM.value,
            resp.input_tokens,
            resp.output_tokens,
        )
        parsed, invalid = parse_reply(resp.text, len(indices))
        stats.invalid_ids += invalid
        return {indices[k]: _result(v) for k, v in parsed.items()}

    async def _pass(
        self,
        pairs: Sequence[PairInput],
        swap: bool,
        usage: Usage,
        stats: VerifyStats,
    ) -> dict[int, PassResult]:
        """Run one full pass and answer every pair it names."""
        windows = self._windows(pairs, swap=swap)
        results = await asyncio.gather(
            *(self._ask(pairs, w, swap, usage, stats) for w in windows)
        )
        out: dict[int, PassResult] = {}
        missing: list[int] = []
        for w, res in zip(windows, results, strict=True):
            if res is None:
                out.update(dict.fromkeys(w, PassResult.ERROR))
                continue
            for j in w:
                if j in res:
                    out[j] = res[j]
                else:
                    missing.append(j)
        for j in missing:
            stats.reasked += 1
            res = await self._ask(pairs, [j], swap, usage, stats)
            if res is None:
                out[j] = PassResult.ERROR
            else:
                out[j] = res.get(j, PassResult.NO_VERDICT)
        return out

    async def verify(
        self, pairs: Sequence[PairInput], usage: Usage
    ) -> tuple[list[PairOutcome], VerifyStats]:
        """Verify ``pairs``; outcomes are in input order."""
        stats = VerifyStats()
        outcomes = [
            PairOutcome(p.a.entity_id, p.b.entity_id, None, VerifyStatus.NO_VERDICT)
            for p in pairs
        ]
        if not pairs:
            return outcomes, stats
        first = await self._pass(pairs, False, usage, stats)
        need_vote = self._first_pass(pairs, first, outcomes)
        if need_vote:
            await self._second_pass(pairs, need_vote, outcomes, usage, stats)
        return outcomes, stats

    def _first_pass(
        self,
        pairs: Sequence[PairInput],
        first: dict[int, PassResult],
        outcomes: list[PairOutcome],
    ) -> list[int]:
        """Apply the first pass and return the positions that need a second vote."""
        need_vote: list[int] = []
        for j, result in first.items():
            o = outcomes[j]
            if result is PassResult.YES:
                if pairs[j].two_votes and self._settings.verification_votes >= 2:
                    o.passes = ["YES"]
                    need_vote.append(j)
                else:
                    o.verdict, o.status, o.passes = (
                        VerdictKind.SAME,
                        VerifyStatus.OK,
                        ["YES"],
                    )
            elif result is PassResult.NO:
                o.verdict, o.status, o.passes = (
                    VerdictKind.DISTINCT,
                    VerifyStatus.OK,
                    ["NO"],
                )
            elif result is PassResult.NO_ANSWER:
                o.status = VerifyStatus.PARSE_FAILED
            elif result is PassResult.ERROR:
                o.status = VerifyStatus.ERROR
            else:
                o.status = VerifyStatus.NO_VERDICT
        return need_vote

    async def _second_pass(
        self,
        pairs: Sequence[PairInput],
        need_vote: list[int],
        outcomes: list[PairOutcome],
        usage: Usage,
        stats: VerifyStats,
    ) -> None:
        """Run the swapped vote for the pairs that need it and settle each one."""
        sub = [pairs[j] for j in need_vote]
        second = await self._pass(sub, True, usage, stats)
        for k, j in enumerate(need_vote):
            o, result = outcomes[j], second[k]
            if result is PassResult.YES:
                o.verdict, o.status, o.passes = (
                    VerdictKind.SAME,
                    VerifyStatus.OK,
                    ["YES", "YES"],
                )
            elif result is PassResult.NO:
                o.verdict, o.status = (
                    VerdictKind.UNCERTAIN,
                    VerifyStatus.VOTE_DISAGREED,
                )
                o.passes = ["YES", "NO"]
            elif result is PassResult.NO_ANSWER:
                o.status, o.passes = VerifyStatus.PARSE_FAILED, []
            elif result is PassResult.ERROR:
                o.status, o.passes = VerifyStatus.ERROR, []
            else:
                o.status, o.passes = VerifyStatus.NO_VERDICT, []


def _result(vote: bool | None) -> PassResult:
    """Return the pass result of one parsed answer."""
    if vote is None:
        return PassResult.NO_ANSWER
    return PassResult.YES if vote else PassResult.NO
