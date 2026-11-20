---
title: ontogr_mem.service.contracts.ForgetMemory
sidebar_label: ForgetMemory
---

# `ontogr_mem.service.contracts.ForgetMemory` \{#ontogr_mem-service-contracts-ForgetMemory}

Bases: <code>[WriteRequest](WriteRequest.md)</code>

Close or remove memory. Hard modes need a dry run, then the token.

**Attributes:**

- [**confirm_token**](#ontogr_mem-service-contracts-ForgetMemory-confirm_token) (<code>str | None</code>) –
- [**dry_run**](#ontogr_mem-service-contracts-ForgetMemory-dry_run) (<code>bool</code>) –
- [**episode_id**](#ontogr_mem-service-contracts-ForgetMemory-episode_id) (<code>str | None</code>) –
- [**fact_id**](#ontogr_mem-service-contracts-ForgetMemory-fact_id) (<code>str | None</code>) –
- [**mode**](#ontogr_mem-service-contracts-ForgetMemory-mode) (<code>[ForgetMode](ForgetMode.md)</code>) –
- [**model_config**](#ontogr_mem-service-contracts-ForgetMemory-model_config) –
- [**reason**](#ontogr_mem-service-contracts-ForgetMemory-reason) (<code>str</code>) –
- [**tenant**](#ontogr_mem-service-contracts-ForgetMemory-tenant) (<code>[TenantId](TenantId.md) | None</code>) –
- [**wait_seconds**](#ontogr_mem-service-contracts-ForgetMemory-wait_seconds) (<code>float</code>) –

## `confirm_token` \{#ontogr_mem-service-contracts-ForgetMemory-confirm_token}

```python
confirm_token: str | None = None
```

## `dry_run` \{#ontogr_mem-service-contracts-ForgetMemory-dry_run}

```python
dry_run: bool = False
```

## `episode_id` \{#ontogr_mem-service-contracts-ForgetMemory-episode_id}

```python
episode_id: str | None = None
```

## `fact_id` \{#ontogr_mem-service-contracts-ForgetMemory-fact_id}

```python
fact_id: str | None = None
```

## `mode` \{#ontogr_mem-service-contracts-ForgetMemory-mode}

```python
mode: ForgetMode = ForgetMode.INVALIDATE
```

## `model_config` \{#ontogr_mem-service-contracts-ForgetMemory-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `reason` \{#ontogr_mem-service-contracts-ForgetMemory-reason}

```python
reason: str = ''
```

## `tenant` \{#ontogr_mem-service-contracts-ForgetMemory-tenant}

```python
tenant: TenantId | None = None
```

## `wait_seconds` \{#ontogr_mem-service-contracts-ForgetMemory-wait_seconds}

```python
wait_seconds: float = Field(default=0.0, ge=0.0, le=60.0)
```
