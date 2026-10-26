"""Shared row encoding and decoding for job storage adapters."""

import json
from datetime import datetime
from typing import Any

from ontogr_mem.data_models.jobs import Job, JobProgress, JobState
from ontogr_mem.data_models.timestamps import from_micros, to_micros
from ontogr_mem.relational_store.jobs.erase_journal import (
    ErasureRecord,
    ErasureStatus,
)


JOB_COLUMNS = (
    "job_id, tenant_id, op, payload, request_hash, idempotency_key, status, "
    "attempts, result, error, created_at, updated_at, trace_id, "
    "progress_stage, progress_done, progress_total"
)


def dumps(value: Any) -> str:
    """Encode structured job data as canonical JSON."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def datetime_from_micros(value: int) -> datetime:
    """Read a finite job timestamp, refusing an open-end sentinel."""
    result = from_micros(value)
    if result is None:
        raise ValueError("job timestamp cannot be an open end")
    return result


def job_from_row(row: Any) -> Job:
    """Decode a detached job from a row keyed by column name."""
    return Job(
        job_id=row["job_id"],
        tenant_id=row["tenant_id"],
        op=row["op"],
        payload=json.loads(row["payload"]),
        request_hash=row["request_hash"],
        idempotency_key=row["idempotency_key"],
        status=JobState(row["status"]),
        attempts=row["attempts"],
        result=None if row["result"] is None else json.loads(row["result"]),
        error=None if row["error"] is None else json.loads(row["error"]),
        created_at=datetime_from_micros(row["created_at"]),
        updated_at=datetime_from_micros(row["updated_at"]),
        trace_id=row["trace_id"],
        progress=(
            None
            if row["progress_stage"] is None
            else JobProgress(
                stage=row["progress_stage"],
                done=row["progress_done"],
                total=row["progress_total"],
            )
        ),
    )


def erasure_from_row(row: Any) -> ErasureRecord:
    """Decode one detached erase journal row."""
    return ErasureRecord(
        erasure_id=row["erasure_id"],
        tenant=row["tenant"],
        mode=row["mode"],
        principal_id=row["principal_id"],
        selector_kind=row["selector_kind"],
        target=row["target"],
        episode_ids=tuple(json.loads(row["episode_ids"])),
        fact_ids=tuple(json.loads(row["fact_ids"])),
        observation_ids=tuple(json.loads(row["observation_ids"])),
        entity_ids=tuple(json.loads(row["entity_ids"])),
        created_at=datetime_from_micros(row["created_at"]),
        status=ErasureStatus(row["status"]),
        last_step=row["last_step"],
        finished_at=(
            None
            if row["finished_at"] is None
            else datetime_from_micros(row["finished_at"])
        ),
        residue=tuple(json.loads(row["residue"])),
        counts=tuple(tuple(item) for item in json.loads(row["counts"])),
    )


def job_values(job: Job) -> tuple[object, ...]:
    """Return row values for one job in the shared column order."""
    return (
        job.job_id,
        job.tenant_id,
        job.op,
        dumps(job.payload),
        job.request_hash,
        job.idempotency_key,
        job.status.value,
        job.attempts,
        None if job.result is None else dumps(job.result),
        None if job.error is None else dumps(job.error),
        to_micros(job.created_at),
        to_micros(job.updated_at),
        job.trace_id,
        None if job.progress is None else job.progress.stage,
        None if job.progress is None else job.progress.done,
        None if job.progress is None else job.progress.total,
    )
