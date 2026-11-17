"""Polling helper for background jobs in service tests."""

import asyncio

from ontogr_mem.service.contracts import JobView
from ontogr_mem.service.memory_service import MemoryService
from ontogr_mem.service.principal import Principal


async def wait_for_job(
    service: MemoryService,
    principal: Principal,
    job_id: str,
    *,
    timeout: float = 30.0,
) -> JobView:
    """Return a job's final view after polling until its state is terminal.

    Args:
        service: The public memory service that owns the job.
        principal: Caller authorized to read the job.
        job_id: Identifier of the submitted job.
        timeout: Maximum seconds to wait for a terminal state.

    Returns:
        The job view in a done or failed state.

    Raises:
        AssertionError: If the job has no terminal state within the timeout.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        job = await service.get_job(principal, job_id)
        if job.status.is_terminal:
            return job
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise AssertionError(
                f"job {job_id} did not finish within {timeout} seconds"
            )
        await asyncio.sleep(min(0.1, remaining))
