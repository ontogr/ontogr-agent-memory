---
title: ontogr_mem.service.contracts.JobView
sidebar_label: JobView
---

# `ontogr_mem.service.contracts.JobView` \{#ontogr_mem-service-contracts-JobView}

Bases: <code>[ContractModel](ContractModel.md)</code>

A job's state visible to an authorized caller.

**Attributes:**

- [**job_id**](#ontogr_mem-service-contracts-JobView-job_id) (<code>str</code>) – Unique identifier of the submission.
- [**tenant**](#ontogr_mem-service-contracts-JobView-tenant) (<code>[TenantId](TenantId.md)</code>) – Tenant that owns the job.
- [**op**](#ontogr_mem-service-contracts-JobView-op) (<code>str</code>) – Operation performed by the job.
- [**status**](#ontogr_mem-service-contracts-JobView-status) (<code>JobState</code>) – Current job lifecycle state.
- [**attempts**](#ontogr_mem-service-contracts-JobView-attempts) (<code>int</code>) – Number of attempts already started.
- [**result**](#ontogr_mem-service-contracts-JobView-result) (<code>dict\[str, Any\] | None</code>) – Result returned by completed work, if any.
- [**error**](#ontogr_mem-service-contracts-JobView-error) (<code>[ErrorInfo](ErrorInfo.md) | None</code>) – Failure details, if the job failed.
- [**trace_id**](#ontogr_mem-service-contracts-JobView-trace_id) (<code>str</code>) – Trace id recorded when the job was submitted.
- [**progress**](#ontogr_mem-service-contracts-JobView-progress) (<code>JobProgress | None</code>) – Latest persisted stage and count, if reported.
- [**created_at**](#ontogr_mem-service-contracts-JobView-created_at) (<code>[UtcDatetime](UtcDatetime.md)</code>) – Time the job was submitted.
- [**updated_at**](#ontogr_mem-service-contracts-JobView-updated_at) (<code>[UtcDatetime](UtcDatetime.md)</code>) – Time of the last job state change.

## `attempts` \{#ontogr_mem-service-contracts-JobView-attempts}

```python
attempts: int = 0
```

## `created_at` \{#ontogr_mem-service-contracts-JobView-created_at}

```python
created_at: UtcDatetime
```

## `error` \{#ontogr_mem-service-contracts-JobView-error}

```python
error: ErrorInfo | None = None
```

## `job_id` \{#ontogr_mem-service-contracts-JobView-job_id}

```python
job_id: str
```

## `model_config` \{#ontogr_mem-service-contracts-JobView-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `op` \{#ontogr_mem-service-contracts-JobView-op}

```python
op: str
```

## `progress` \{#ontogr_mem-service-contracts-JobView-progress}

```python
progress: JobProgress | None = None
```

## `result` \{#ontogr_mem-service-contracts-JobView-result}

```python
result: dict[str, Any] | None = Field(default=None, description='Payload of the verb named by op; each write verb fills its own keys once the job is done.')
```

## `status` \{#ontogr_mem-service-contracts-JobView-status}

```python
status: JobState
```

## `tenant` \{#ontogr_mem-service-contracts-JobView-tenant}

```python
tenant: TenantId
```

## `trace_id` \{#ontogr_mem-service-contracts-JobView-trace_id}

```python
trace_id: str = ''
```

## `updated_at` \{#ontogr_mem-service-contracts-JobView-updated_at}

```python
updated_at: UtcDatetime
```
