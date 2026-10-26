"""Settings of the consolidation runner and stages.

Environment variables use the prefix ``MEMORY_CONSOLIDATION_`` and the delimiter
``__`` between a section and its field, for example
``MEMORY_CONSOLIDATION_RUNNER__LEASE_TTL_S``. Every default below is an unmeasured
starting point and should be tuned on real data.
"""

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ontogr_mem.llm.base import ModelSize


class LLMSettings(BaseModel):
    """Model calls of the consolidation stages.

    Attributes:
        model_size: Model tier the stages call.
        max_output_tokens: Cap on generated tokens. A curator that answers with
            several lessons needs room for all of them. Unmeasured.
    """

    model_config = ConfigDict(frozen=True)

    model_size: ModelSize = ModelSize.SMALL
    max_output_tokens: int = Field(default=2048, ge=1)


class RunnerSettings(BaseModel):
    """Lease and rerun limits of the runner.

    Defaults are unmeasured starting points.

    Attributes:
        lease_ttl_s: Seconds a lease lasts without renewal. A holder that
            crashes blocks the scope for this long. Unmeasured.
        max_reruns: Extra passes a run makes when reruns were requested while it
            held the lease. Unmeasured.
    """

    model_config = ConfigDict(frozen=True)

    lease_ttl_s: float = Field(default=600.0, gt=0)
    max_reruns: int = Field(default=3, ge=0)


class FeedbackSettings(BaseModel):
    """Streaming feedback weights and their decay on read.

    Defaults are unmeasured starting points.

    Attributes:
        alpha: Share of the gap to the rating that one explicit rating moves the
            weight by. Unmeasured.
        implicit_factor: Multiplier of ``alpha`` for implicit feedback. Unmeasured.
        max_attempts: Times an event is tried before it is left unapplied.
            Unmeasured.
        decay_beta: Fraction of the distance to neutral lost per idle day.
            Unmeasured.
        neutral_epsilon: Weights closer than this to neutral are dropped. Unmeasured.
    """

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(default=0.1, ge=0.0, le=1.0)
    implicit_factor: float = Field(default=0.5, ge=0.0, le=1.0)
    max_attempts: int = Field(default=3, ge=1)
    decay_beta: float = Field(default=0.02, ge=0.0, le=1.0)
    neutral_epsilon: float = Field(default=0.01, ge=0.0)


class ScoringSettings(BaseModel):
    """Importance as a weighted sum of recency, frequency and salience.

    Defaults are unmeasured starting points.

    Attributes:
        w_recency: Weight of the recency term. Unmeasured.
        w_frequency: Weight of the frequency term. Unmeasured.
        w_salience: Weight of the salience term. Unmeasured.
        half_life_days: Days after which recency halves. Unmeasured.
        frequency_cap: Access count at which frequency reaches 1.0. Unmeasured.
        degree_cap: Graph degree at which the degree part of salience caps. Unmeasured.
    """

    model_config = ConfigDict(frozen=True)

    w_recency: float = 0.4
    w_frequency: float = 0.3
    w_salience: float = 0.3
    half_life_days: float = Field(default=30.0, gt=0)
    frequency_cap: int = Field(default=20, ge=1)
    degree_cap: int = Field(default=10, ge=1)


class PruneSettings(BaseModel):
    """Archive and retention. Nothing changes unless ``dry_run`` is turned off.

    Defaults are unmeasured starting points.

    Attributes:
        dry_run: Report what would be archived without changing anything.
        archive_below: Importance under which an item is archived. Unmeasured.
        min_age_days: Minimum age before an item can be archived. Unmeasured.
        retention_days: Days an archived item is kept before deletion. Unmeasured.
        delete_after_retention: Delete archived items after retention ends.
    """

    model_config = ConfigDict(frozen=True)

    dry_run: bool = True
    archive_below: float = Field(default=0.15, ge=0.0, le=1.0)
    min_age_days: float = Field(default=30.0, ge=0)
    retention_days: float = Field(default=180.0, ge=0)
    delete_after_retention: bool = False


class CommunitySettings(BaseModel):
    """Community detection and the summary hierarchy above it.

    Defaults are unmeasured starting points.

    Attributes:
        max_iterations: Label propagation rounds. Unmeasured.
        min_size: Smallest group that counts as a community. Unmeasured.
        max_bucket_size: Largest number of children in one bucket. Unmeasured.
        summary_max_chars: Length cap of a community summary. Unmeasured.
    """

    model_config = ConfigDict(frozen=True)

    max_iterations: int = Field(default=20, ge=0)
    min_size: int = Field(default=2, ge=1)
    max_bucket_size: int = Field(default=20, ge=2)
    summary_max_chars: int = Field(default=600, ge=1)


class DescriptionSettings(BaseModel):
    """Entity summary regeneration.

    Defaults are unmeasured starting points.

    Attributes:
        max_neighbours: Neighbours shown to the model for one entity. Unmeasured.
        max_batch: Entities regenerated in one run. Unmeasured.
    """

    model_config = ConfigDict(frozen=True)

    max_neighbours: int = Field(default=20, ge=0)
    max_batch: int = Field(default=50, ge=1)


class ReflectionSettings(BaseModel):
    """Session distillation.

    Defaults are unmeasured starting points.

    Attributes:
        min_confidence: Confidence an entry needs to be distilled. Unmeasured.
        min_helpfulness: Helpfulness an entry needs to be distilled. Unmeasured.
        batch_size: Entries per model call. Unmeasured.
        auto_accept: Apply proposals without a review step.
    """

    model_config = ConfigDict(frozen=True)

    min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    min_helpfulness: int = 0
    batch_size: int = Field(default=6, ge=1)
    auto_accept: bool = False


class ConsolidationSettings(BaseSettings):
    """All consolidation settings in one place.

    Attributes:
        runner: Lease and rerun limits.
        feedback: Feedback weight update and decay.
        scoring: Importance terms.
        prune: Archive and retention.
        community: Community detection and summaries.
        description: Entity summary regeneration.
        reflection: Session distillation.
        llm: Model calls of the stages.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_CONSOLIDATION_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    runner: RunnerSettings = Field(default_factory=RunnerSettings)
    feedback: FeedbackSettings = Field(default_factory=FeedbackSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    prune: PruneSettings = Field(default_factory=PruneSettings)
    community: CommunitySettings = Field(default_factory=CommunitySettings)
    description: DescriptionSettings = Field(default_factory=DescriptionSettings)
    reflection: ReflectionSettings = Field(default_factory=ReflectionSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
