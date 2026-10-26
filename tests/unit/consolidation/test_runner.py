"""Runner behavior: stage order, failures, watermarks, the lease and reruns."""

import pytest

from ontogr_mem.consolidation.config import ConsolidationSettings
from ontogr_mem.consolidation.errors import RegistryError
from ontogr_mem.consolidation.models import (
    RunStatus,
    StageResult,
    StageStatus,
    Watermark,
)
from ontogr_mem.consolidation.runner import ConsolidationRunner
from ontogr_mem.consolidation.stage import Stage, StageContext
from ontogr_mem.graphstore.memory import MemoryConsStore
from tests.support.consolidation import ManualClock


class FakeStage:
    """A stage with scripted gate and run behavior that counts its runs."""

    def __init__(
        self,
        name: str,
        *,
        fatal: bool = False,
        gate_reason: str | None = None,
        gate_raises: Exception | None = None,
        status: StageStatus = StageStatus.OK,
        raises: Exception | None = None,
        watermark: str | None = None,
    ) -> None:
        """Script the stage name, flags and behavior."""
        self.name = name
        self.fatal = fatal
        self._gate_reason = gate_reason
        self._gate_raises = gate_raises
        self._status = status
        self._raises = raises
        self._watermark = watermark
        self.runs = 0

    async def gate(self, ctx: StageContext) -> str | None:
        """Return the scripted gate result, or raise the scripted error."""
        if self._gate_raises is not None:
            raise self._gate_raises
        return self._gate_reason

    async def run(self, ctx: StageContext) -> StageResult:
        """Run the scripted behavior and count the run."""
        self.runs += 1
        if self._raises is not None:
            raise self._raises
        return StageResult(
            stage=self.name, status=self._status, watermark=self._watermark
        )


def make_runner(
    stages: list[Stage], clock: ManualClock, store: MemoryConsStore | None = None
) -> tuple[ConsolidationRunner, MemoryConsStore]:
    """Build a runner over scripted stages, with a fresh store unless given."""
    store = store or MemoryConsStore()
    runner = ConsolidationRunner(stages, store, clock, ConsolidationSettings())
    return runner, store


async def test_the_registry_takes_any_number_of_fatal_stages() -> None:
    """The registry does not police fatal stages; a failing one still aborts."""
    after = FakeStage("after")
    runner, _ = make_runner(
        [
            FakeStage("one", fatal=True, raises=RuntimeError("boom")),
            FakeStage("two", fatal=True),
            after,
        ],
        ManualClock(),
    )

    result = await runner.run()

    assert result.status is RunStatus.FAILED
    assert result.stage("one").status is StageStatus.FAILED
    assert after.runs == 0
    assert result.stage("after").status is StageStatus.SKIPPED


def test_registry_refuses_duplicate_names() -> None:
    """Duplicate stage names are refused."""
    with pytest.raises(RegistryError):
        make_runner([FakeStage("a", fatal=True), FakeStage("a")], ManualClock())


async def test_stages_run_in_order_and_report_each() -> None:
    """Stages run in order and each reports a result."""
    runner, _ = make_runner(
        [FakeStage("one", fatal=True), FakeStage("two"), FakeStage("three")],
        ManualClock(),
    )

    result = await runner.run()

    assert [s.stage for s in result.stages] == ["one", "two", "three"]
    assert result.status is RunStatus.OK


async def test_gated_stage_is_skipped_without_running() -> None:
    """A gated stage is skipped, with its reason as the note."""
    quiet = FakeStage("quiet", gate_reason="nothing new")
    runner, _ = make_runner([FakeStage("a", fatal=True), quiet], ManualClock())

    result = await runner.run()

    assert quiet.runs == 0
    assert result.stage("quiet").status is StageStatus.SKIPPED
    assert result.stage("quiet").notes == ["nothing new"]


async def test_gate_exception_fails_the_stage_not_the_run() -> None:
    """An exception in a gate fails that stage and the run goes on."""
    after = FakeStage("after")
    broken = FakeStage("broken", gate_raises=ValueError("bad gate"))
    runner, _ = make_runner([FakeStage("a", fatal=True), broken, after], ManualClock())

    result = await runner.run()

    assert result.stage("broken").status is StageStatus.FAILED
    assert result.stage("broken").errors == ["ValueError: bad gate"]
    assert broken.runs == 0
    assert after.runs == 1
    assert result.status is RunStatus.PARTIAL


async def test_non_fatal_failure_does_not_stop_later_stages() -> None:
    """A non-fatal failure is reported and later stages still run."""
    after = FakeStage("after")
    runner, _ = make_runner(
        [
            FakeStage("a", fatal=True),
            FakeStage("bad", raises=ValueError("boom")),
            after,
        ],
        ManualClock(),
    )

    result = await runner.run()

    assert result.stage("bad").status is StageStatus.FAILED
    assert result.stage("bad").errors == ["ValueError: boom"]
    assert after.runs == 1
    assert result.status is RunStatus.PARTIAL


async def test_fatal_failure_aborts_the_rest() -> None:
    """A fatal failure skips every stage after it."""
    after = FakeStage("after")
    runner, _ = make_runner(
        [FakeStage("core", fatal=True, raises=RuntimeError("x")), after],
        ManualClock(),
    )

    result = await runner.run()

    assert result.status is RunStatus.FAILED
    assert after.runs == 0
    assert result.stage("after").status is StageStatus.SKIPPED


async def test_watermark_is_stored_only_when_a_stage_is_ok() -> None:
    """A watermark is stored only for a stage that finished OK."""
    good = FakeStage("good", fatal=True, watermark="7")
    half = FakeStage("half", status=StageStatus.PARTIAL, watermark="9")
    runner, store = make_runner([good, half], ManualClock())

    await runner.run()

    stored = await store.get(Watermark, "good")
    assert stored is not None and stored.value == "7"
    assert await store.get(Watermark, "half") is None


async def test_a_second_runner_is_locked_and_its_rerun_is_honored_once() -> None:
    """A second runner is locked while the first runs, and its request reruns once."""
    clock = ManualClock()
    store = MemoryConsStore()

    class Rival(FakeStage):
        async def run(self, ctx: StageContext) -> StageResult:
            if self.runs == 0:
                rival, _ = make_runner([FakeStage("x", fatal=True)], clock, store)
                locked = await rival.run()
                assert locked.status is RunStatus.LOCKED
            return await super().run(ctx)

    runner, _ = make_runner([Rival("a", fatal=True)], clock, store)

    result = await runner.run()

    assert result.status is RunStatus.OK
    assert result.reruns == 1


async def test_expired_lease_can_be_taken() -> None:
    """A run takes over a lease that has expired."""
    clock = ManualClock()
    runner, store = make_runner([FakeStage("a", fatal=True)], clock)
    await store.acquire("dead-worker", clock(), 10)
    clock.advance(seconds=11)

    result = await runner.run()

    assert result.status is RunStatus.OK


async def test_rerun_requested_during_a_run_is_claimed_after_release() -> None:
    """A rerun requested during a pass runs one more pass."""

    class AskingStage(FakeStage):
        async def run(self, ctx: StageContext) -> StageResult:
            if self.runs == 0:
                await ctx.store.request_rerun()
            return await super().run(ctx)

    stage = AskingStage("a", fatal=True)
    runner, _ = make_runner([stage], ManualClock())

    result = await runner.run()

    assert stage.runs == 2
    assert result.reruns == 1


async def test_reruns_are_capped() -> None:
    """Reruns stop at the configured maximum."""

    class AlwaysAsking(FakeStage):
        async def run(self, ctx: StageContext) -> StageResult:
            await ctx.store.request_rerun()
            return await super().run(ctx)

    stage = AlwaysAsking("a", fatal=True)
    runner, _ = make_runner([stage], ManualClock())

    result = await runner.run()

    assert stage.runs == 4  # the first pass plus max_reruns (3)
    assert result.reruns == 3


async def test_lease_is_released_after_a_run() -> None:
    """The lease is released when a run ends."""
    clock = ManualClock()
    runner, store = make_runner([FakeStage("a", fatal=True)], clock)

    await runner.run()

    assert await store.acquire("next", clock(), 60)


async def test_a_rerun_request_at_the_cap_is_kept_for_the_next_run() -> None:
    """A rerun request left at the cap is kept for the next holder."""

    class AlwaysAsking(FakeStage):
        async def run(self, ctx: StageContext) -> StageResult:
            await ctx.store.request_rerun()
            return await super().run(ctx)

    clock = ManualClock()
    runner, store = make_runner([AlwaysAsking("a", fatal=True)], clock)

    result = await runner.run()

    assert result.reruns == 3
    assert result.rerun_pending is True
    assert await store.acquire("next", clock(), 60)
    assert await store.release("next") is True


async def test_no_pending_flag_when_nothing_was_asked() -> None:
    """No rerun is pending when none was requested."""
    runner, _ = make_runner([FakeStage("a", fatal=True)], ManualClock())

    result = await runner.run()

    assert result.rerun_pending is False
    assert result.reruns == 0


async def test_the_lease_is_renewed_between_stages_of_a_long_run() -> None:
    """The lease is renewed before each stage, so a rival cannot take it mid-run."""
    clock = ManualClock()
    seen: list[bool] = []

    class Slow(FakeStage):
        async def run(self, ctx: StageContext) -> StageResult:
            clock.advance(seconds=500)  # two of these exceed the 600 s lease
            seen.append(await ctx.store.acquire("other", clock(), 60))
            return await super().run(ctx)

    runner, _ = make_runner([Slow("a", fatal=True), Slow("b"), Slow("c")], clock)

    result = await runner.run()

    assert result.status is RunStatus.OK
    assert seen == [False, False, False]


async def test_a_stolen_lease_stops_the_remaining_stages() -> None:
    """A run whose lease was taken stops and reports LEASE_LOST."""
    clock = ManualClock()

    class Thief(FakeStage):
        async def run(self, ctx: StageContext) -> StageResult:
            clock.advance(seconds=601)  # our lease ran out during this stage
            assert await ctx.store.acquire("thief", clock(), 600)
            return await super().run(ctx)

    after = FakeStage("after")
    runner, store = make_runner([Thief("a", fatal=True), after], clock)

    result = await runner.run()

    assert result.status is RunStatus.LEASE_LOST
    assert after.runs == 0
    assert result.stage("after").notes == ["lease lost to another worker"]
    # The thief still holds the lease, so the cleanup must not have released it.
    assert not await store.acquire("third", clock(), 60)


async def test_run_ids_do_not_depend_on_the_clock() -> None:
    """Two runs on one fixed clock get different run ids."""
    clock = ManualClock()
    runner, _ = make_runner([FakeStage("a", fatal=True)], clock)

    first = await runner.run()
    second = await runner.run()

    assert first.run_id != second.run_id
