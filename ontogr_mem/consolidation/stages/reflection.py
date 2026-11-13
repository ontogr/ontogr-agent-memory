"""Stage 6: reflection. Distil session entries into lesson proposals.

A curator call per batch proposes lessons. A writer call per lesson returns a
``LessonVerdict``. An accepted lesson is stored as a PENDING proposal, or as
ACCEPTED when ``auto_accept`` is set. Any other verdict is stored as REJECTED.
A session's watermark moves only when none of its model calls failed.

Writer calls in a batch run concurrently, and sessions run concurrently too. The
consolidation role's concurrency limit caps parallel model calls. The lessons a
writer knows about are the ones stored when the stage started plus the ones the
same session accepted earlier, so a lesson accepted in one batch is known in the
next.
"""

import asyncio
from collections.abc import Sequence

from ontogr_mem.consolidation.agents import (
    LessonDraft,
    curator_agent,
    writer_agent,
)
from ontogr_mem.consolidation.models import (
    LessonVerdict,
    Proposal,
    ProposalStatus,
    SessionEntry,
    StageResult,
    StageStatus,
    Watermark,
)
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.consolidation.text import quote, short_hash
from ontogr_mem.data_models.extraction import Usage
from ontogr_mem.data_models.text import normalize_name
from ontogr_mem.llm.failures import ModelFailure
from ontogr_mem.llm.models import RoleModels
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.llm.run import TokenUsage, run_agent


STAGE = "reflection"


def proposal_id(session_id: str, text: str) -> str:
    """Return the id of a lesson in a session.

    The id ignores case and extra whitespace, so the same lesson from the same
    session is decided once.
    """
    return "pr-" + short_hash(session_id, normalize_name(text))


def _watermark_key(session_id: str) -> str:
    return f"reflection|{session_id}"


async def _eligible(ctx: StageContext) -> dict[str, list[SessionEntry]]:
    """Return the entries to distil, by session and in order, past each watermark.

    Entries below the helpfulness or confidence floor are never distilled.
    """
    cfg = ctx.settings.reflection
    by_session: dict[str, list[SessionEntry]] = {}
    for entry in await ctx.store.list(SessionEntry):
        if entry.helpfulness < cfg.min_helpfulness:
            continue
        if entry.confidence < cfg.min_confidence:
            continue
        by_session.setdefault(entry.session_id, []).append(entry)
    eligible: dict[str, list[SessionEntry]] = {}
    for session_id, entries in by_session.items():
        mark = await ctx.store.get(Watermark, _watermark_key(session_id))
        seen = int(mark.value) if mark is not None else 0
        due = sorted((e for e in entries if e.seq > seen), key=lambda e: e.seq)
        if due:
            eligible[session_id] = due
    return eligible


def _status(verdict: LessonVerdict, *, auto_accept: bool) -> ProposalStatus:
    if verdict is not LessonVerdict.ACCEPT:
        return ProposalStatus.REJECTED
    return ProposalStatus.ACCEPTED if auto_accept else ProposalStatus.PENDING


def _record_usage(usage: Usage, measured: TokenUsage | None) -> None:
    """Add all request usage for one consolidation run."""
    if measured is None or measured.requests < 1:
        return
    bucket_key = f"{STAGE}/{ModelRole.CONSOLIDATION.value}"
    usage.add(
        STAGE,
        ModelRole.CONSOLIDATION.value,
        measured.input_tokens,
        measured.output_tokens,
    )
    # The token totals include all output-validation re-asks.
    usage.buckets[bucket_key].calls += measured.requests - 1


async def _curate(
    models: RoleModels,
    usage: Usage,
    batch: Sequence[SessionEntry],
) -> list[LessonDraft]:
    """Return the lessons the curator proposes for one batch of entries."""
    entries = [{"id": e.entry_id, "text": e.text} for e in batch]
    try:
        result = await run_agent(
            curator_agent,
            ModelRole.CONSOLIDATION,
            f"Entries: {quote(entries)}",
            models=models,
            deps=None,
        )
    except ModelFailure as exc:
        _record_usage(usage, exc.usage)
        raise
    _record_usage(usage, result.usage)
    return result.output.lessons


async def _judge(
    models: RoleModels,
    usage: Usage,
    *,
    lesson: str,
    sources: list[str],
    known: list[str],
) -> LessonVerdict:
    """Return the writer's verdict on one lesson, given its source texts."""
    prompt = (
        f"Lesson: {quote(lesson)}\n"
        f"Source entries: {quote(sources)}\n"
        f"Known lessons: {quote(known)}"
    )
    try:
        result = await run_agent(
            writer_agent,
            ModelRole.CONSOLIDATION,
            prompt,
            models=models,
            deps=None,
        )
    except ModelFailure as exc:
        _record_usage(usage, exc.usage)
        raise
    _record_usage(usage, result.usage)
    return result.output.verdict


async def _decide_batch(
    ctx: StageContext,
    session_id: str,
    batch: Sequence[SessionEntry],
    lessons: Sequence[LessonDraft],
    known: list[str],
    *,
    models: RoleModels,
    usage: Usage,
) -> tuple[bool, dict[str, int], list[str]]:
    """Decide every lesson of one batch.

    A lesson with no source entry from its batch is rejected as unsupported
    without a model call. A lesson that was already proposed is counted as a
    duplicate and costs no model call. The remaining lessons are judged at the
    same time, each seeing the lessons that were known when the batch started.
    The accepted lessons are merged into ``known`` in the order the curator
    proposed them.

    Returns:
        Whether the batch is free of errors, the counts of the batch and the
        error texts.
    """
    counts = {"proposed": 0, "rejected": 0, "duplicates": 0}
    errors: list[str] = []
    by_id = {e.entry_id: e.text for e in batch}
    pending: list[tuple[LessonDraft, str, list[str]]] = []
    seen: set[str] = set()
    for lesson in lessons:
        proposal_identifier = proposal_id(session_id, lesson.text)
        if (
            proposal_identifier in seen
            or await ctx.store.get(Proposal, proposal_identifier) is not None
        ):
            counts["duplicates"] += 1
            continue
        seen.add(proposal_identifier)
        sources = [i for i in lesson.entry_ids if i in by_id]
        pending.append((lesson, proposal_identifier, sources))

    verdicts: list[LessonVerdict | None] = [None] * len(pending)

    async def judge(index: int, lesson: LessonDraft, sources: list[str]) -> str | None:
        """Return the verdict on one lesson, or the error text of its call."""
        if not sources:
            verdicts[index] = LessonVerdict.UNSUPPORTED
            return None
        try:
            verdict = await _judge(
                models,
                usage,
                lesson=lesson.text,
                sources=[by_id[i] for i in sources],
                known=known,
            )
        except ModelFailure as exc:
            return str(exc)
        verdicts[index] = verdict
        return None

    failures = await asyncio.gather(
        *(
            judge(index, lesson, sources)
            for index, (lesson, _, sources) in enumerate(pending)
        )
    )

    ok = True
    for (lesson, proposal_identifier, sources), verdict, failure in zip(
        pending, verdicts, failures, strict=True
    ):
        if failure is not None:
            ok = False
            errors.append(f"writer: {failure}")
            continue
        if verdict is None:
            ok = False
            errors.append(f"writer: no verdict for {proposal_identifier}")
            continue
        await ctx.store.put(
            Proposal(
                proposal_id=proposal_identifier,
                session_id=session_id,
                text=lesson.text,
                source_entry_ids=sources,
                status=_status(
                    verdict, auto_accept=ctx.settings.reflection.auto_accept
                ),
                reason="" if verdict is LessonVerdict.ACCEPT else verdict.value,
                created_at=ctx.clock(),
            )
        )
        if verdict is LessonVerdict.ACCEPT:
            known.append(lesson.text)
            counts["proposed"] += 1
        else:
            counts["rejected"] += 1
    return ok, counts, errors


async def _distil_session(
    ctx: StageContext,
    session_id: str,
    entries: Sequence[SessionEntry],
    known: list[str],
    *,
    models: RoleModels,
    usage: Usage,
) -> tuple[bool, dict[str, int], list[str]]:
    """Distil one session and return whether it is complete, its counts and errors.

    The curator calls of the batches run at the same time. The lessons of a
    batch are decided in order, so a lesson accepted in one batch is known in
    the next.
    """
    cfg = ctx.settings.reflection
    counts = {"proposed": 0, "rejected": 0, "duplicates": 0}
    errors: list[str] = []
    batches = [
        entries[i : i + cfg.batch_size] for i in range(0, len(entries), cfg.batch_size)
    ]
    curated = await asyncio.gather(
        *(_curate(models, usage, batch) for batch in batches),
        return_exceptions=True,
    )
    ok = True
    for batch, result in zip(batches, curated, strict=True):
        if isinstance(result, BaseException):
            if not isinstance(result, ModelFailure):
                raise result
            ok = False
            errors.append(f"curator: {result}")
            continue
        batch_ok, batch_counts, batch_errors = await _decide_batch(
            ctx,
            session_id,
            batch,
            result,
            known,
            models=models,
            usage=usage,
        )
        ok = ok and batch_ok
        errors.extend(batch_errors)
        for key, value in batch_counts.items():
            counts[key] += value
    return ok, counts, errors


class ReflectionStage:
    """Distil session entries into lesson proposals."""

    name = STAGE
    fatal = False

    async def gate(self, ctx: StageContext) -> str | None:
        """Skip without a model, or when no session has entries left to distil."""
        if not ctx.models.configured(ModelRole.CONSOLIDATION):
            return "no model"
        if not await _eligible(ctx):
            return "no new eligible session entries"
        return None

    async def run(self, ctx: StageContext) -> StageResult:
        """Curate and write, every session at the same time.

        A failed model call leaves its lessons undecided and holds the session's
        watermark, so the next run retries them. Other exceptions propagate.
        """
        models = ctx.models
        if not models.configured(ModelRole.CONSOLIDATION):
            return StageResult(
                stage=self.name, status=StageStatus.SKIPPED, notes=["no model"]
            )
        usage = Usage()
        counts = {"sessions": 0, "proposed": 0, "rejected": 0, "duplicates": 0}
        errors: list[str] = []
        eligible = sorted((await _eligible(ctx)).items())
        known_lessons = [
            p.text
            for p in await ctx.store.list(Proposal)
            if p.status is not ProposalStatus.REJECTED
        ]

        sessions = await asyncio.gather(
            *(
                _distil_session(
                    ctx,
                    session_id,
                    entries,
                    list(known_lessons),
                    models=models,
                    usage=usage,
                )
                for session_id, entries in eligible
            )
        )

        for (session_id, entries), session in zip(eligible, sessions, strict=True):
            ok, session_counts, session_errors = session
            counts["sessions"] += 1
            for key, value in session_counts.items():
                counts[key] += value
            errors.extend(f"{session_id}: {error}" for error in session_errors)
            if ok:
                await ctx.set_watermark(
                    _watermark_key(session_id), str(entries[-1].seq)
                )

        return StageResult(
            stage=self.name,
            status=StageStatus.PARTIAL if errors else StageStatus.OK,
            counts=counts,
            errors=errors,
            usage=usage,
        )
