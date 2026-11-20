---
title: ontogr_mem.service.contracts.ScopedRequest
sidebar_label: ScopedRequest
---

# `ontogr_mem.service.contracts.ScopedRequest` \{#ontogr_mem-service-contracts-ScopedRequest}

Bases: <code>[ContractModel](ContractModel.md)</code>

A request within one tenant; `None` means the principal's tenant.

**Attributes:**

- [**model_config**](#ontogr_mem-service-contracts-ScopedRequest-model_config) –
- [**tenant**](#ontogr_mem-service-contracts-ScopedRequest-tenant) (<code>[TenantId](TenantId.md) | None</code>) –

## `model_config` \{#ontogr_mem-service-contracts-ScopedRequest-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `tenant` \{#ontogr_mem-service-contracts-ScopedRequest-tenant}

```python
tenant: TenantId | None = None
```
