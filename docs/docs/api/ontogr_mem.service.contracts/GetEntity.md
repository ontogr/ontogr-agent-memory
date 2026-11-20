---
title: ontogr_mem.service.contracts.GetEntity
sidebar_label: GetEntity
---

# `ontogr_mem.service.contracts.GetEntity` \{#ontogr_mem-service-contracts-GetEntity}

Bases: <code>[ScopedRequest](ScopedRequest.md)</code>

Look up one entity by id or name, with its facts.

**Attributes:**

- [**entity_id**](#ontogr_mem-service-contracts-GetEntity-entity_id) (<code>str | None</code>) –
- [**include_history**](#ontogr_mem-service-contracts-GetEntity-include_history) (<code>bool</code>) –
- [**max_facts**](#ontogr_mem-service-contracts-GetEntity-max_facts) (<code>int</code>) –
- [**model_config**](#ontogr_mem-service-contracts-GetEntity-model_config) –
- [**name**](#ontogr_mem-service-contracts-GetEntity-name) (<code>str | None</code>) –
- [**tenant**](#ontogr_mem-service-contracts-GetEntity-tenant) (<code>[TenantId](TenantId.md) | None</code>) –

## `entity_id` \{#ontogr_mem-service-contracts-GetEntity-entity_id}

```python
entity_id: str | None = None
```

## `include_history` \{#ontogr_mem-service-contracts-GetEntity-include_history}

```python
include_history: bool = False
```

## `max_facts` \{#ontogr_mem-service-contracts-GetEntity-max_facts}

```python
max_facts: int = Field(default=25, ge=1, le=MAX_RESULTS)
```

## `model_config` \{#ontogr_mem-service-contracts-GetEntity-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `name` \{#ontogr_mem-service-contracts-GetEntity-name}

```python
name: str | None = None
```

## `tenant` \{#ontogr_mem-service-contracts-GetEntity-tenant}

```python
tenant: TenantId | None = None
```
