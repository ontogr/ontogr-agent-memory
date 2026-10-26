"""The consolidation stages, in run order."""

from ontogr_mem.consolidation.stage import Stage
from ontogr_mem.consolidation.stages.communities import CommunitiesStage
from ontogr_mem.consolidation.stages.deduplication import DeduplicationStage
from ontogr_mem.consolidation.stages.descriptions import DescriptionsStage
from ontogr_mem.consolidation.stages.feedback_weights import FeedbackWeightsStage
from ontogr_mem.consolidation.stages.ingest_cleanup import IngestCleanupStage
from ontogr_mem.consolidation.stages.prune_archive import PruneArchiveStage
from ontogr_mem.consolidation.stages.reflection import ReflectionStage
from ontogr_mem.consolidation.stages.report import ReportStage
from ontogr_mem.consolidation.stages.scoring import ScoringStage


def default_stages() -> list[Stage]:
    """Return the nine stages in run order."""
    return [
        IngestCleanupStage(),
        DeduplicationStage(),
        FeedbackWeightsStage(),
        DescriptionsStage(),
        CommunitiesStage(),
        ReflectionStage(),
        ScoringStage(),
        PruneArchiveStage(),
        ReportStage(),
    ]


__all__ = [
    "CommunitiesStage",
    "DeduplicationStage",
    "DescriptionsStage",
    "FeedbackWeightsStage",
    "IngestCleanupStage",
    "PruneArchiveStage",
    "ReflectionStage",
    "ReportStage",
    "ScoringStage",
    "default_stages",
]
