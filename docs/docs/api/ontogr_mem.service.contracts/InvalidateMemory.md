---
title: ontogr_mem.service.contracts.InvalidateMemory
sidebar_label: InvalidateMemory
---

# `ontogr_mem.service.contracts.InvalidateMemory` \{#ontogr_mem-service-contracts-InvalidateMemory}

Bases: <code>[WriteRequest](WriteRequest.md)</code>

Close one current fact while retaining its version history.

**Attributes:**

- [**at**](#ontogr_mem-service-contracts-InvalidateMemory-at) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –
- [**fact_id**](#ontogr_mem-service-contracts-InvalidateMemory-fact_id) (<code>str</code>) –
- [**idempotency_key**](#ontogr_mem-service-contracts-InvalidateMemory-idempotency_key) (<code>str | None</code>) –
- [**model_config**](#ontogr_mem-service-contracts-InvalidateMemory-model_config) –
- [**reason**](#ontogr_mem-service-contracts-InvalidateMemory-reason) (<code>str</code>) –
- [**tenant**](#ontogr_mem-service-contracts-InvalidateMemory-tenant) (<code>[TenantId](TenantId.md) | None</code>) –
- [**wait_seconds**](#ontogr_mem-service-contracts-InvalidateMemory-wait_seconds) (<code>float</code>) –

## `at` \{#ontogr_mem-service-contracts-InvalidateMemory-at}

```python
at: UtcDatetime | None = None
```

## `fact_id` \{#ontogr_mem-service-contracts-InvalidateMemory-fact_id}

```python
fact_id: str
```

## `idempotency_key` \{#ontogr_mem-service-contracts-InvalidateMemory-idempotency_key}

```python
idempotency_key: str | None = Field(default=None, max_length=128)
```

## `model_config` \{#ontogr_mem-service-contracts-InvalidateMemory-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `reason` \{#ontogr_mem-service-contracts-InvalidateMemory-reason}

```python
reason: str = ''
```

## `tenant` \{#ontogr_mem-service-contracts-InvalidateMemory-tenant}

```python
tenant: TenantId | None = None
```

## `wait_seconds` \{#ontogr_mem-service-contracts-InvalidateMemory-wait_seconds}

```python
wait_seconds: float = Field(default=0.0, ge=0.0, le=60.0)
```
