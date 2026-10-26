"""Consolidation: the scheduled job that runs stages over one tenant's memory.

Build a runner from the stages, the tenant's store and the ports the stages act on::

    runner = ConsolidationRunner(
        default_stages(),
        store,
        clock,
        ConsolidationSettings(),
        resolution,
        temporal,
    )
    result = await runner.run()
"""

from ontogr_mem.consolidation.config import ConsolidationSettings
from ontogr_mem.consolidation.models import (
    RunResult,
    RunStatus,
    StageResult,
    StageStatus,
)
from ontogr_mem.consolidation.runner import ConsolidationRunner
from ontogr_mem.consolidation.stages import default_stages
from ontogr_mem.consolidation.store import ConsStore


__all__ = [
    "ConsStore",
    "ConsolidationRunner",
    "ConsolidationSettings",
    "RunResult",
    "RunStatus",
    "StageResult",
    "StageStatus",
    "default_stages",
]
