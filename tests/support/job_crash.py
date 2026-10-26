"""Run a sqlite job that either blocks for a crash test or finishes recovery."""

import asyncio
import sys
from pathlib import Path
from typing import Any

from ontogr_mem.data_models.clock import SystemClock
from ontogr_mem.data_models.jobs import Job
from ontogr_mem.relational_store.jobs.runner import JobRunner
from ontogr_mem.relational_store.sqlite.jobs import SqliteJobStore


async def main(database: str, marker: Path, mode: str) -> None:
    """Run the crash worker in hang or finish mode.

    Args:
        database: SQLite file shared by the killed and recovery processes.
        marker: File written only after the handler starts its first attempt.
        mode: Either ``hang`` to block or ``finish`` to recover and print status.
    """
    if mode not in ("hang", "finish"):
        raise ValueError("mode must be hang or finish")
    store = SqliteJobStore(database)

    async def index(job: Job) -> dict[str, Any]:
        if mode == "hang":
            pending = marker.with_suffix(".pending")
            pending.write_text(job.job_id)
            pending.replace(marker)
            await asyncio.Event().wait()
        return {"ok": True}

    runner = JobRunner(store, {"index": index}, SystemClock())
    try:
        await runner.start()
        job = await runner.submit("t", "index", {}, idempotency_key="crash-1")
        await runner.drain()
        if mode == "finish":
            finished = await runner.get(job.job_id)
            if finished is None:
                raise RuntimeError("recovered job is missing")
            print(finished.status.value)
    finally:
        await runner.stop()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2]), sys.argv[3]))
