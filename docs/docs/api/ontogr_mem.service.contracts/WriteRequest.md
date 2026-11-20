---
title: ontogr_mem.service.contracts.WriteRequest
sidebar_label: WriteRequest
---

# `ontogr_mem.service.contracts.WriteRequest` \{#ontogr_mem-service-contracts-WriteRequest}

Bases: <code>[ScopedRequest](ScopedRequest.md)</code>

A request that queues a job and may wait for it.

**Attributes:**

- [**model_config**](#ontogr_mem-service-contracts-WriteRequest-model_config) –
- [**tenant**](#ontogr_mem-service-contracts-WriteRequest-tenant) (<code>[TenantId](TenantId.md) | None</code>) –
- [**wait_seconds**](#ontogr_mem-service-contracts-WriteRequest-wait_seconds) (<code>float</code>) –

## `model_config` \{#ontogr_mem-service-contracts-WriteRequest-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `tenant` \{#ontogr_mem-service-contracts-WriteRequest-tenant}

```python
tenant: TenantId | None = None
```

## `wait_seconds` \{#ontogr_mem-service-contracts-WriteRequest-wait_seconds}

```python
wait_seconds: float = Field(default=0.0, ge=0.0, le=60.0)
```
