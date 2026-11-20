---
title: ontogr_mem.service.contracts.WriteReceipt
sidebar_label: WriteReceipt
---

# `ontogr_mem.service.contracts.WriteReceipt` \{#ontogr_mem-service-contracts-WriteReceipt}

Bases: <code>[ContractModel](ContractModel.md)</code>

What a write returns at once: a job to follow.

**Attributes:**

- [**error**](#ontogr_mem-service-contracts-WriteReceipt-error) (<code>[ErrorInfo](ErrorInfo.md) | None</code>) –
- [**idempotent_replay**](#ontogr_mem-service-contracts-WriteReceipt-idempotent_replay) (<code>bool</code>) –
- [**job_id**](#ontogr_mem-service-contracts-WriteReceipt-job_id) (<code>str</code>) –
- [**model_config**](#ontogr_mem-service-contracts-WriteReceipt-model_config) –
- [**result**](#ontogr_mem-service-contracts-WriteReceipt-result) (<code>dict\[str, Any\] | None</code>) –
- [**status**](#ontogr_mem-service-contracts-WriteReceipt-status) (<code>JobState</code>) –
- [**tenant**](#ontogr_mem-service-contracts-WriteReceipt-tenant) (<code>[TenantId](TenantId.md)</code>) –
- [**trace_id**](#ontogr_mem-service-contracts-WriteReceipt-trace_id) (<code>str</code>) –

## `error` \{#ontogr_mem-service-contracts-WriteReceipt-error}

```python
error: ErrorInfo | None = None
```

## `idempotent_replay` \{#ontogr_mem-service-contracts-WriteReceipt-idempotent_replay}

```python
idempotent_replay: bool = False
```

## `job_id` \{#ontogr_mem-service-contracts-WriteReceipt-job_id}

```python
job_id: str
```

## `model_config` \{#ontogr_mem-service-contracts-WriteReceipt-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `result` \{#ontogr_mem-service-contracts-WriteReceipt-result}

```python
result: dict[str, Any] | None = Field(default=None, description='Payload of the write that ran; add_memory, update_memory, forget_memory, consolidate_memory and clear_tenant fill their own keys.')
```

## `status` \{#ontogr_mem-service-contracts-WriteReceipt-status}

```python
status: JobState
```

## `tenant` \{#ontogr_mem-service-contracts-WriteReceipt-tenant}

```python
tenant: TenantId
```

## `trace_id` \{#ontogr_mem-service-contracts-WriteReceipt-trace_id}

```python
trace_id: str = ''
```
