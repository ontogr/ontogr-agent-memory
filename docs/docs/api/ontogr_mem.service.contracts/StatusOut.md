---
title: ontogr_mem.service.contracts.StatusOut
sidebar_label: StatusOut
---

# `ontogr_mem.service.contracts.StatusOut` \{#ontogr_mem-service-contracts-StatusOut}

Bases: <code>[ContractModel](ContractModel.md)</code>

Service health.

**Attributes:**

- [**backend**](#ontogr_mem-service-contracts-StatusOut-backend) (<code>str</code>) –
- [**detail**](#ontogr_mem-service-contracts-StatusOut-detail) (<code>dict\[str, Any\]</code>) –
- [**llm_available**](#ontogr_mem-service-contracts-StatusOut-llm_available) (<code>bool</code>) –
- [**model_config**](#ontogr_mem-service-contracts-StatusOut-model_config) –
- [**queued_jobs**](#ontogr_mem-service-contracts-StatusOut-queued_jobs) (<code>int</code>) –
- [**status**](#ontogr_mem-service-contracts-StatusOut-status) (<code>Literal['ok', 'degraded']</code>) –
- [**tenants**](#ontogr_mem-service-contracts-StatusOut-tenants) (<code>list\[str\]</code>) –

## `backend` \{#ontogr_mem-service-contracts-StatusOut-backend}

```python
backend: str
```

## `detail` \{#ontogr_mem-service-contracts-StatusOut-detail}

```python
detail: dict[str, Any] = Field(default_factory=dict, description='Extra health fields the status verb chooses to expose.')
```

## `llm_available` \{#ontogr_mem-service-contracts-StatusOut-llm_available}

```python
llm_available: bool
```

## `model_config` \{#ontogr_mem-service-contracts-StatusOut-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `queued_jobs` \{#ontogr_mem-service-contracts-StatusOut-queued_jobs}

```python
queued_jobs: int = 0
```

## `status` \{#ontogr_mem-service-contracts-StatusOut-status}

```python
status: Literal['ok', 'degraded']
```

## `tenants` \{#ontogr_mem-service-contracts-StatusOut-tenants}

```python
tenants: list[str] = Field(default_factory=list)
```
