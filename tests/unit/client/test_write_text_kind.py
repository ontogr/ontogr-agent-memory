"""Unit tests for how Memory.write_text chooses the memory kind of its request.

The service is replaced by a recording fake, so the tests read the request the
client builds and send no traffic.
"""

from ontogr_mem.client.memory import Memory
from ontogr_mem.data_models import Role
from ontogr_mem.data_models.jobs import JobState
from ontogr_mem.service.contracts import AddMemory, MemoryKind, WriteReceipt
from tests.support.scripted_model import scripted_role_models


class RecordingService:
    """Fake memory service that keeps every add_memory request it receives."""

    def __init__(self) -> None:
        """Create a service with no recorded requests."""
        self.requests: list[AddMemory] = []

    async def add_memory(self, principal: object, request: AddMemory) -> WriteReceipt:
        """Record the request and return a queued receipt.

        Args:
            principal: Caller identity. The client passes it through unchanged.
            request: The add request built by the client.

        Returns:
            A receipt for a queued job in the request's tenant.
        """
        self.requests.append(request)
        return WriteReceipt(
            job_id="job-1",
            status=JobState.QUEUED,
            tenant=request.tenant or "acme-tenant",
        )


class TestWriteTextKind:
    """Memory.write_text builds a message request when given a role."""

    async def test_a_role_makes_a_message_episode_request(self) -> None:
        """A role on write_text produces a message request with that role."""
        service = RecordingService()
        memory = Memory(service, tenant="acme-tenant", models=scripted_role_models())

        await memory.write_text(
            "I work at Acme.", role=Role.ASSISTANT, speaker="helper"
        )

        request = service.requests[0]
        assert request.kind == MemoryKind.MESSAGE
        assert request.role == Role.ASSISTANT
        assert request.speaker == "helper"
        assert request.content == "I work at Acme."

    async def test_no_role_makes_a_text_request(self) -> None:
        """Omitting the role produces a text request."""
        service = RecordingService()
        memory = Memory(service, tenant="acme-tenant", models=scripted_role_models())

        await memory.write_text("Quarterly report text.")

        assert service.requests[0].kind == MemoryKind.TEXT
