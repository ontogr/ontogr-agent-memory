"""A memory client over a canned service, for tests of the MCP tools.

``Memory`` is the real client, so tool schemas come from its real signatures. Only
the service behind it is replaced by ``FakeMemoryService``, which records the
requests it receives and returns the receipts, jobs and contexts a test sets.
"""

from datetime import UTC, datetime

from ontogr_mem.client.memory import Memory
from ontogr_mem.data_models.jobs import JobState
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.service.contracts import (
    AddMemory,
    EntityOut,
    EntityView,
    ErrorInfo,
    GetEntity,
    InvalidateMemory,
    JobView,
    MemoryContext,
    SearchMemory,
    WriteReceipt,
)
from tests.support.scripted_model import scripted_model, scripted_role_models


TENANT = "acme-tenant"

JOB_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def job_view(
    status: JobState = JobState.QUEUED, error: ErrorInfo | None = None
) -> JobView:
    """Build a job view for an add_memory job with the given state and error."""
    return JobView(
        job_id="job-9",
        tenant=TENANT,
        op="add_memory",
        status=status,
        error=error,
        created_at=JOB_TIME,
        updated_at=JOB_TIME,
    )


class FakeMemoryService:
    """A service that returns canned results or raises a configured failure.

    Attributes:
        context: Result of every search.
        entity: Result of every entity lookup.
        receipt: Result of every write and invalidation.
        job: Result of every job lookup.
        failure: Exception raised by every operation, or None for no failure.
        add_requests: Every add request received, in order.
        search_requests: Every search request received, in order.
        invalidate_requests: Every invalidation request received, in order.
    """

    def __init__(self) -> None:
        """Create a service that succeeds with empty, queued results."""
        self.context = MemoryContext(state="none")
        self.entity = EntityView(
            entity=EntityOut(entity_id="entity-1", tenant=TENANT, name="Ada")
        )
        self.receipt = WriteReceipt(
            job_id="job-1", status=JobState.QUEUED, tenant=TENANT
        )
        self.job = job_view()
        self.failure: Exception | None = None
        self.add_requests: list[AddMemory] = []
        self.search_requests: list[SearchMemory] = []
        self.invalidate_requests: list[InvalidateMemory] = []

    async def add_memory(self, principal: object, request: AddMemory) -> WriteReceipt:
        """Record the request, then return the canned receipt or raise."""
        self.add_requests.append(request)
        if self.failure is not None:
            raise self.failure
        return self.receipt

    async def search_memory(
        self, principal: object, request: SearchMemory
    ) -> MemoryContext:
        """Record the request, then return the canned context or raise."""
        self.search_requests.append(request)
        if self.failure is not None:
            raise self.failure
        return self.context

    async def get_entity(self, principal: object, request: GetEntity) -> EntityView:
        """Return the canned entity view or raise."""
        if self.failure is not None:
            raise self.failure
        return self.entity

    async def invalidate(
        self, principal: object, request: InvalidateMemory
    ) -> WriteReceipt:
        """Record the request, then return the canned receipt or raise."""
        self.invalidate_requests.append(request)
        if self.failure is not None:
            raise self.failure
        return self.receipt

    async def get_job(self, principal: object, job_id: str) -> JobView:
        """Return the canned job or raise."""
        if self.failure is not None:
            raise self.failure
        return self.job


def build_memory(service: FakeMemoryService, *roles: ModelRole) -> Memory:
    """Build a real memory client over the fake service.

    Args:
        service: The service behind the client.
        *roles: Model roles to configure with a scripted model. With none, no
            role has a model.

    Returns:
        A client for the tenant ``TENANT``.
    """
    model, _ = scripted_model()
    models = scripted_role_models(**{role.value: model for role in roles})
    return Memory(service, tenant=TENANT, models=models)
