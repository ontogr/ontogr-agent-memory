---
title: ontogr_mem.service.contracts.MemoryFeedback
sidebar_label: MemoryFeedback
---

# `ontogr_mem.service.contracts.MemoryFeedback` \{#ontogr_mem-service-contracts-MemoryFeedback}

Bases: <code>[ScopedRequest](ScopedRequest.md)</code>

A score on one entity or on a fact's two entities.

**Attributes:**

- [**entity_id**](#ontogr_mem-service-contracts-MemoryFeedback-entity_id) (<code>str | None</code>) –
- [**fact_id**](#ontogr_mem-service-contracts-MemoryFeedback-fact_id) (<code>str | None</code>) –
- [**model_config**](#ontogr_mem-service-contracts-MemoryFeedback-model_config) –
- [**note**](#ontogr_mem-service-contracts-MemoryFeedback-note) (<code>str</code>) –
- [**score**](#ontogr_mem-service-contracts-MemoryFeedback-score) (<code>float</code>) –
- [**tenant**](#ontogr_mem-service-contracts-MemoryFeedback-tenant) (<code>[TenantId](TenantId.md) | None</code>) –

## `entity_id` \{#ontogr_mem-service-contracts-MemoryFeedback-entity_id}

```python
entity_id: str | None = None
```

## `fact_id` \{#ontogr_mem-service-contracts-MemoryFeedback-fact_id}

```python
fact_id: str | None = None
```

## `model_config` \{#ontogr_mem-service-contracts-MemoryFeedback-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `note` \{#ontogr_mem-service-contracts-MemoryFeedback-note}

```python
note: str = ''
```

## `score` \{#ontogr_mem-service-contracts-MemoryFeedback-score}

```python
score: float = Field(ge=-1.0, le=1.0)
```

## `tenant` \{#ontogr_mem-service-contracts-MemoryFeedback-tenant}

```python
tenant: TenantId | None = None
```
