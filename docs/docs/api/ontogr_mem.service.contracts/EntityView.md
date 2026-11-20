---
title: ontogr_mem.service.contracts.EntityView
sidebar_label: EntityView
---

# `ontogr_mem.service.contracts.EntityView` \{#ontogr_mem-service-contracts-EntityView}

Bases: <code>[ContractModel](ContractModel.md)</code>

An entity and the facts that touch it.

**Attributes:**

- [**entity**](#ontogr_mem-service-contracts-EntityView-entity) (<code>[EntityOut](EntityOut.md)</code>) –
- [**facts**](#ontogr_mem-service-contracts-EntityView-facts) (<code>list\[[FactOut](FactOut.md)\]</code>) –
- [**model_config**](#ontogr_mem-service-contracts-EntityView-model_config) –

## `entity` \{#ontogr_mem-service-contracts-EntityView-entity}

```python
entity: EntityOut
```

## `facts` \{#ontogr_mem-service-contracts-EntityView-facts}

```python
facts: list[FactOut] = Field(default_factory=list)
```

## `model_config` \{#ontogr_mem-service-contracts-EntityView-model_config}

```python
model_config = ConfigDict(extra='forbid')
```
