"""Runs every stage of the tenant under a lease, and reruns when asked."""

import logging
import uuid
from collections.abc import Callable, Sequence

from ontogr_mem.consolidation.config import ConsolidationSettings
from ontogr_mem.consolidation.models import (
    RunResult,
    RunStatus,
    StageResult,
    StageStatus,
)
from ontogr_mem.consolidation.ports import ResolutionPort, TemporalPort
from ontogr_mem.consolidation.registry import validate_stages
from ontogr_mem.consolidation.stage import Clock, Stage, StageContext
from ontogr_mem.consolidation.store import ConsStore
from ontogr_mem.llm.gate import LLMGate


log = logging.getLogger(__name__)


def _random_run_id() -> str:
    return uuid.uuid4().hex


def _skipped(stage: Stage, note: str) -> StageResult:
    return StageResult(stage=stage.name, status=StageStatus.SKIPPED, notes=[note])


class ConsolidationRunner:
    """Acquire the tenant's lease, run every stage, release, and rerun if asked.

    Args:
        stages: Stages in run order, with unique names.
        store: Consolidation store of the tenant.
        clock: Time source of the run.
        settings: Lease and rerun limits.
        resolution: Entity resolution the stages act on.
        temporal: Temporal engine the stages act on.
        llm: The gate every model call of the stages passes through.
        run_ids: Source of run identities, which are also lease holder ids.
            Defaults to random hex strings.
    """

    def __init__(
        self,
        stages: Sequence[Stage],
        store: ConsStore,
        clock: Clock,
        settings: ConsolidationSettings,
        *,
        resolution: ResolutionPort | None = None,
        temporal: TemporalPort | None = None,
        llm: LLMGate | None = None,
        run_ids: Callable[[], str] | None = None,
    ) -> None:
        """Wire the runner to its stages and ports."""
        self._stages = validate_stages(stages)
        self._store = store
        self._clock = clock
        self._settings = settings
        self._resolution = resolution
        self._temporal = temporal
        self._llm = llm
        self._run_ids = run_ids or _random_run_id

    async def run(self) -> RunResult:
        """Run the stages once, plus the reruns that were requested meanwhile.

        A run that cannot take the lease returns ``LOCKED`` and leaves a rerun
        request for the holder. The results of every pass are kept, so a
        failure of an earlier pass stays visible.
        """
        run_id = self._run_ids()
        ttl = self._settings.runner.lease_ttl_s
        if not await self._store.acquire(run_id, self._clock(), ttl):
            await self._store.request_rerun()
            return RunResult(run_id=run_id, status=RunStatus.LOCKED)
        reruns = 0
        try:
            result = await self._pass(run_id)
            stages = list(result.stages)
            while result.status is not RunStatus.LEASE_LOST:
                if not await self._store.release(run_id):
                    break
                if reruns >= self._settings.runner.max_reruns:
                    # The request was consumed by release; keep it for the next run.
                    await self._store.request_rerun()
                    result.rerun_pending = True
                    break
                reruns += 1
                if not await self._store.acquire(run_id, self._clock(), ttl):
                    await self._store.request_rerun()
                    result.rerun_pending = True
                    break
                result = await self._pass(run_id)
                stages.extend(result.stages)
            result.stages = stages
            result.reruns = reruns
            return result
        finally:
            await self._store.release(run_id)

    async def _pass(self, run_id: str) -> RunResult:
        ctx = StageContext(
            run_id=run_id,
            clock=self._clock,
            store=self._store,
            settings=self._settings,
            resolution=self._resolution,
            temporal=self._temporal,
            llm=self._llm,
        )
        results: list[StageResult] = []
        aborted = False
        lost = False
        ttl = self._settings.runner.lease_ttl_s
        for stage in self._stages:
            # A long run renews its lease before each stage. If another holder
            # took it over, the rest of the pass is skipped.
            if not lost and not await self._store.acquire(run_id, self._clock(), ttl):
                lost = True
            if lost:
                results.append(_skipped(stage, "lease lost to another worker"))
                continue
            if aborted:
                results.append(_skipped(stage, "aborted: the fatal stage failed"))
                continue
            res = await self._run_stage(stage, ctx)
            results.append(res)
            ctx.results.append(res)
            if res.status is StageStatus.FAILED and stage.fatal:
                aborted = True
        status = RunStatus.OK
        if lost:
            status = RunStatus.LEASE_LOST
        elif aborted:
            status = RunStatus.FAILED
        elif any(
            r.status in (StageStatus.FAILED, StageStatus.PARTIAL) for r in results
        ):
            status = RunStatus.PARTIAL
        return RunResult(run_id=run_id, status=status, stages=results)

    async def _run_stage(self, stage: Stage, ctx: StageContext) -> StageResult:
        # A failing stage is reported as FAILED so that one bad stage cannot stop
        # the others.
        try:
            reason = await stage.gate(ctx)
            if reason is not None:
                return _skipped(stage, reason)
            res = await stage.run(ctx)
        except Exception as exc:
            log.warning("stage %s failed: %r", stage.name, exc)
            return StageResult(
                stage=stage.name,
                status=StageStatus.FAILED,
                errors=[f"{type(exc).__name__}: {exc}"],
            )
        if res.status is StageStatus.OK and res.watermark is not None:
            await ctx.set_watermark(stage.name, res.watermark)
        return res
