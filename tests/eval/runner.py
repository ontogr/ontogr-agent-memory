"""Run the judge over the labelled set and report what it gets wrong and costs.

The report gives precision and recall for ``contradicts``, unanswered counts,
model requests and tokens, and the pairs answered wrongly.

Run against the configured judge model from the repository root:

    uv run python -m tests.eval.runner
"""

import asyncio
from collections import Counter
from dataclasses import dataclass, field

from ontogr_mem.llm import LLMConfig, ModelRole, RoleModels, build_models
from ontogr_mem.temporal.closure import ClosureJudge
from ontogr_mem.temporal.settings import ClosureSettings
from tests.eval.labelled import LABELLED, Labelled


@dataclass(slots=True)
class EvalReport:
    """Counts and rates for one run."""

    n: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    unanswered: int = 0
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

    def render(self) -> str:
        """Return a short text table."""
        lines = [
            f"pairs {self.n}  calls {self.calls}  tokens in/out "
            f"{self.tokens_in}/{self.tokens_out}",
            f"precision {self.precision:.2f}  recall {self.recall:.2f}",
            f"unanswered {self.unanswered}",
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


async def run_eval(
    models: RoleModels,
    items: tuple[Labelled, ...] = LABELLED,
    settings: ClosureSettings | None = None,
) -> EvalReport:
    """Judge statement text and report outcomes and token usage.

    The judge receives statement text only. Intervals and sources remain
    metadata in the labelled set.

    Args:
        models: Shared model pool used by the judge role.
        items: Labelled pairs to judge.
        settings: Closure settings for the prompt version and candidate count.

    Returns:
        Counts, rates and the pairs answered wrongly.
    """
    judge = ClosureJudge(models, settings or ClosureSettings())
    report = EvalReport(n=len(items))
    for item in items:
        outcome = await judge.judge(item.b[0], [(item.id, item.a[0])])
        report.calls += outcome.requests
        report.tokens_in += outcome.input_tokens
        report.tokens_out += outcome.output_tokens
        said = "contradicts" if item.id in outcome.contradicted else None
        if not outcome.failed and said is None:
            said = "compatible"
        report.category_total[item.category] += 1
        if outcome.failed:
            report.unanswered += 1
        positive = item.label == "contradicts"
        if said == "contradicts" and positive:
            report.tp += 1
        elif said == "contradicts":
            report.fp += 1
        elif positive:
            report.fn += 1
        else:
            report.tn += 1
        if said == item.label or (not positive and said == "compatible"):
            report.by_category[item.category] += 1
        else:
            report.wrong.append((item.id, item.label, str(said)))
    return report


async def _run_live() -> EvalReport:
    """Open configured models for one explicit live evaluation."""
    models = build_models(LLMConfig())
    try:
        if not models.configured(ModelRole.JUDGE):
            raise SystemExit("configure the judge model before running evaluation")
        return await run_eval(models)
    finally:
        await models.aclose()


def main() -> None:
    """Print a labelled evaluation using the configured judge model."""
    print(asyncio.run(_run_live()).render())


if __name__ == "__main__":
    main()
