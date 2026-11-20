---
title: ontogr_mem.service.contracts.UpdateMemory
sidebar_label: UpdateMemory
---

# `ontogr_mem.service.contracts.UpdateMemory` \{#ontogr_mem-service-contracts-UpdateMemory}

Bases: <code>[WriteRequest](WriteRequest.md)</code>

Replace a fact: close the old version and assert the new one.

**Attributes:**

- [**effective_at**](#ontogr_mem-service-contracts-UpdateMemory-effective_at) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –
- [**fact_id**](#ontogr_mem-service-contracts-UpdateMemory-fact_id) (<code>str</code>) –
- [**idempotency_key**](#ontogr_mem-service-contracts-UpdateMemory-idempotency_key) (<code>str | None</code>) –
- [**model_config**](#ontogr_mem-service-contracts-UpdateMemory-model_config) –
- [**object**](#ontogr_mem-service-contracts-UpdateMemory-object) (<code>str | None</code>) –
- [**statement**](#ontogr_mem-service-contracts-UpdateMemory-statement) (<code>str | None</code>) –
- [**tenant**](#ontogr_mem-service-contracts-UpdateMemory-tenant) (<code>[TenantId](TenantId.md) | None</code>) –
- [**wait_seconds**](#ontogr_mem-service-contracts-UpdateMemory-wait_seconds) (<code>float</code>) –

## `effective_at` \{#ontogr_mem-service-contracts-UpdateMemory-effective_at}

```python
effective_at: UtcDatetime | None = Field(default=None, description='When the change took effect. Defaults to now.')
```

## `fact_id` \{#ontogr_mem-service-contracts-UpdateMemory-fact_id}

```python
fact_id: str
```

## `idempotency_key` \{#ontogr_mem-service-contracts-UpdateMemory-idempotency_key}

```python
idempotency_key: str | None = Field(default=None, max_length=128)
```

## `model_config` \{#ontogr_mem-service-contracts-UpdateMemory-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `object` \{#ontogr_mem-service-contracts-UpdateMemory-object}

```python
object: str | None = Field(default=None, description='The new object name.')
```

## `statement` \{#ontogr_mem-service-contracts-UpdateMemory-statement}

```python
statement: str | None = None
```

## `tenant` \{#ontogr_mem-service-contracts-UpdateMemory-tenant}

```python
tenant: TenantId | None = None
```

## `wait_seconds` \{#ontogr_mem-service-contracts-UpdateMemory-wait_seconds}

```python
wait_seconds: float = Field(default=0.0, ge=0.0, le=60.0)
```
