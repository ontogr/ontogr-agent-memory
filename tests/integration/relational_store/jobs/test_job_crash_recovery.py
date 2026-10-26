"""Recover sqlite jobs after process death and enforce the interruption cap."""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

from ontogr_mem.data_models.jobs import Job, JobState
from ontogr_mem.relational_store.jobs.runner import JobRunner
from ontogr_mem.relational_store.sqlite.jobs import SqliteJobStore
from tests.support.clock import FixedClock


class TestJobCrashRecovery:
    """Use a real sqlite file as the recovery artifact across processes."""

    async def test_a_killed_lane_is_finished_by_a_new_process(
        self, tmp_path: Path
    ) -> None:
        """A killed process leaves a running row that resumes on its next attempt."""
        database = tmp_path / "crash.sqlite3"
        marker = tmp_path / "entered.txt"
        command = [
            sys.executable,
            "-m",
            "tests.support.job_crash",
            str(database),
            str(marker),
        ]
        root = Path(__file__).resolve().parents[3]
        process = subprocess.Popen(
            [*command, "hang"], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            async with asyncio.timeout(30):
                while not marker.exists():
                    if process.poll() is not None:
                        _, stderr = await asyncio.to_thread(process.communicate)
                        raise AssertionError(
                            "crash worker exited before entering handler: "
                            f"{stderr.decode()}"
                        )
                    await asyncio.sleep(0.05)
            process.kill()
            await asyncio.to_thread(process.communicate, timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                await asyncio.to_thread(process.communicate, timeout=10)

        job_id = marker.read_text()
        store = SqliteJobStore(str(database))
        try:
            running = await store.get_job(job_id)
            assert running is not None and running.status == JobState.RUNNING
            assert running.attempts == 1
        finally:
            await store.close()

        recovered = await asyncio.to_thread(
            subprocess.run,
            [*command, "finish"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert recovered.returncode == 0, recovered.stderr
        assert recovered.stdout.strip() == "done"
        store = SqliteJobStore(str(database))
        try:
            finished = await store.get_job(job_id)
            assert finished is not None and finished.status == JobState.DONE
            assert finished.attempts == 2
            assert finished.result == {"ok": True}
        finally:
            await store.close()

    async def test_a_job_interrupted_at_the_cap_does_not_run_again(
        self, tmp_path: Path
    ) -> None:
        """Recovery fails a capped job without invoking its handler again."""
        store = SqliteJobStore(str(tmp_path / "capped.sqlite3"))
        entered = asyncio.Event()
        calls = 0

        async def index(job: Job) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            entered.set()
            await asyncio.Event().wait()
            return {}

        runner = JobRunner(store, {"index": index}, FixedClock(), max_attempts=1)
        restarted = JobRunner(store, {"index": index}, FixedClock(), max_attempts=1)
        try:
            job = await runner.submit("t", "index", {})
            await asyncio.wait_for(entered.wait(), 5)
            await runner.stop()
            assert await restarted.start() == 1
            await asyncio.wait_for(restarted.drain(), 5)
            failed = await restarted.get(job.job_id)
            assert failed is not None and failed.status == JobState.FAILED
            assert failed.attempts == 1
            assert failed.error == {
                "type": "Interrupted",
                "message": "the job was interrupted before it finished",
            }
            assert calls == 1
        finally:
            await runner.stop()
            await restarted.stop()
            await store.close()
