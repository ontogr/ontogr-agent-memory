---
title: ontogr_mem.service.contracts.ClearTenant
sidebar_label: ClearTenant
---

# `ontogr_mem.service.contracts.ClearTenant` \{#ontogr_mem-service-contracts-ClearTenant}

Bases: <code>[WriteRequest](WriteRequest.md)</code>

Remove everything in a tenant. Needs a dry run, then its token.

**Attributes:**

- [**confirm_token**](#ontogr_mem-service-contracts-ClearTenant-confirm_token) (<code>str | None</code>) –
- [**dry_run**](#ontogr_mem-service-contracts-ClearTenant-dry_run) (<code>bool</code>) –
- [**model_config**](#ontogr_mem-service-contracts-ClearTenant-model_config) –
- [**tenant**](#ontogr_mem-service-contracts-ClearTenant-tenant) (<code>[TenantId](TenantId.md) | None</code>) –
- [**wait_seconds**](#ontogr_mem-service-contracts-ClearTenant-wait_seconds) (<code>float</code>) –

## `confirm_token` \{#ontogr_mem-service-contracts-ClearTenant-confirm_token}

```python
confirm_token: str | None = None
```

## `dry_run` \{#ontogr_mem-service-contracts-ClearTenant-dry_run}

```python
dry_run: bool = False
```

## `model_config` \{#ontogr_mem-service-contracts-ClearTenant-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `tenant` \{#ontogr_mem-service-contracts-ClearTenant-tenant}

```python
tenant: TenantId | None = None
```

## `wait_seconds` \{#ontogr_mem-service-contracts-ClearTenant-wait_seconds}

```python
wait_seconds: float = Field(default=0.0, ge=0.0, le=60.0)
```
