"""Run the judge over the labelled set and report what it gets wrong and costs.

The report gives precision and recall for ``contradicts``, the false-supersession
rate (a non-contradiction judged as one, the costly error), the uncertain and
unanswered counts, calls and tokens, and the pairs answered wrongly.

Live run, from the repository root (needs ``OLLAMA_API_KEY`` in the environment
of this one command):

    OLLAMA_API_KEY=... uv run python -m tests.eval.runner
"""

import asyncio
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ontogr_mem.llm import LLMClient, LLMGate, LLMRequest, LLMResponse
from ontogr_mem.temporal.judge import JudgeFact, LLMPairJudge, PairInput
from ontogr_mem.temporal.settings import JudgeSettings
from tests.eval.labelled import LABELLED, Labelled
from tests.support.ollama import OllamaLLMClient, available


GATE_INPUT_TOKENS = 50_000


@dataclass(slots=True)
class EvalReport:
    """Counts and rates for one run."""

    n: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    uncertain: int = 0
    unanswered: int = 0
    uncertain_negatives: int = 0
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    wrong: list[tuple[str, str, str]] = field(default_factory=list)
    by_category: Counter[str] = field(default_factory=Counter)
    category_total: Counter[str] = field(default_factory=Counter)

    @property
    def precision(self) -> float:
        """Share of judged contradictions that were real."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        """Share of real contradictions that were found."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def false_supersession_rate(self) -> float:
        """Non-contradictions judged as contradictions, over all non-contradictions."""
        negatives = self.fp + self.tn + self.uncertain_negatives
        return self.fp / negatives if negatives else 0.0

    def render(self) -> str:
        """Return a short text table."""
        lines = [
            f"pairs {self.n}  calls {self.calls}  tokens in/out "
            f"{self.tokens_in}/{self.tokens_out}",
            f"precision {self.precision:.2f}  recall {self.recall:.2f}  "
            f"false-supersession {self.false_supersession_rate:.2f}",
            f"uncertain {self.uncertain}  unanswered {self.unanswered}",
        ]
        for cat in sorted(self.category_total):
            lines.append(
                f"  {cat:<20} {self.by_category[cat]}/{self.category_total[cat]}"
            )
        lines += [
            f"  WRONG {pair_id}: label {label}, got {got}"
            for pair_id, label, got in self.wrong
        ]
        return "\n".join(lines)


def to_input(item: Labelled) -> PairInput:
    """Return a labelled pair as the judge sees it; the pair id is the key."""
    left = JudgeFact(f"{item.id}-a", item.a[0], item.a[1], item.a[2])
    right = JudgeFact(f"{item.id}-b", item.b[0], item.b[1], item.b[2])
    return PairInput(item.id, f"hash-{item.id}", left, right)


class _Counting:
    """An ``LLMClient`` that counts the calls it passes on."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Count the call and pass it to the wrapped client."""
        self.calls += 1
        return await self._inner.complete(request)


async def run_eval(
    llm: LLMClient,
    items: tuple[Labelled, ...] = LABELLED,
    settings: JudgeSettings | None = None,
) -> EvalReport:
    """Judge every labelled pair once and score the answers.

    Args:
        llm: Model client the judge calls through a gate.
        items: Labelled pairs to judge.
        settings: Judge settings. Defaults to the environment's values.

    Returns:
        Counts, rates and the pairs answered wrongly.
    """
    counted = _Counting(llm)
    gate = LLMGate(counted, max_input_tokens=GATE_INPUT_TOKENS)
    judge = LLMPairJudge(gate, settings or JudgeSettings())
    outcome = await judge.judge_pairs(
        [to_input(item) for item in items], now=datetime.now(UTC)
    )
    said_by_id = {v.pair_key: v.verdict.value for v in outcome.verdicts}
    report = EvalReport(
        n=len(items),
        calls=counted.calls,
        tokens_in=outcome.tokens_in,
        tokens_out=outcome.tokens_out,
    )
    for item in items:
        said = said_by_id.get(item.id)
        report.category_total[item.category] += 1
        if said is None:
            report.unanswered += 1
        positive = item.label == "contradicts"
        if said == "uncertain":
            report.uncertain += 1
            if not positive:
                report.uncertain_negatives += 1
        if said == "contradicts" and positive:
            report.tp += 1
        elif said == "contradicts":
            report.fp += 1
        elif positive:
            report.fn += 1
        else:
            report.tn += 1
        if said == item.label or (not positive and said not in (None, "contradicts")):
            report.by_category[item.category] += 1
        else:
            report.wrong.append((item.id, item.label, str(said)))
    return report


def main() -> None:
    """Run the set against Ollama Cloud and print the report."""
    if not available():
        sys.exit("set OLLAMA_API_KEY to run the live evaluation")
    llm = OllamaLLMClient()
    print(f"model {llm.model}")
    print(asyncio.run(run_eval(llm)).render())


if __name__ == "__main__":
    main()
