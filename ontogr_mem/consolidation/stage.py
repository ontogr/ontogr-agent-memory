"""The stage protocol and the context every stage runs in."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from ontogr_mem.consolidation.graph import EntityGraph

from ontogr_mem.consolidation.config import ConsolidationSettings
from ontogr_mem.consolidation.models import StageResult, Watermark
from ontogr_mem.consolidation.ports import ResolutionPort, TemporalPort
from ontogr_mem.consolidation.store import ConsStore
from ontogr_mem.llm.gate import LLMGate


Clock = Callable[[], datetime]


@dataclass(slots=True)
class StageContext:
    """Everything a stage may touch during one pass of a run.

    Attributes:
        run_id: Identity of the run, used as the lease holder.
        clock: Time source of the run.
        store: Consolidation document store of the tenant.
        settings: Settings of the run.
        resolution: Entity resolution the stages act on, if wired.
        temporal: Temporal engine the stages act on, if wired.
        llm: The gate every model call of the stages passes through, if wired.
        results: Results of the stages that already ran in this pass.
        entity_graph: The graph read by stages in this pass, when built.
    """

    run_id: str
    clock: Clock
    store: ConsStore
    settings: ConsolidationSettings
    resolution: ResolutionPort | None = None
    temporal: TemporalPort | None = None
    llm: LLMGate | None = None
    results: list[StageResult] = field(default_factory=list)
    entity_graph: EntityGraph | None = field(default=None, init=False, repr=False)

    async def set_watermark(self, stage: str, value: str) -> None:
        """Store the watermark of ``stage``."""
        await self.store.put(Watermark(key=stage, value=value, updated_at=self.clock()))

    def invalidate_entity_graph(self) -> None:
        """Discard the cached graph after a stage changes entities or facts."""
        self.entity_graph = None


class Stage(Protocol):
    """One consolidation step.

    ``gate`` returns a reason to skip, or None to run. A skipped stage makes no
    model call. A ``fatal`` stage aborts the run when it fails. The other stages
    fail open: a failure is reported and the run goes on.
    """

    name: str
    fatal: bool

    async def gate(self, ctx: StageContext) -> str | None:
        """Return why there is nothing to do, or None."""
        ...

    async def run(self, ctx: StageContext) -> StageResult:
        """Do the work. Running it again must converge to the same state."""
        ...
