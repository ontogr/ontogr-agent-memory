---
title: ontogr_mem.service.contracts.EntityPage
sidebar_label: EntityPage
---

# `ontogr_mem.service.contracts.EntityPage` \{#ontogr_mem-service-contracts-EntityPage}

Bases: <code>[ContractModel](ContractModel.md)</code>

One page of active entities.

**Attributes:**

- [**entities**](#ontogr_mem-service-contracts-EntityPage-entities) (<code>list\[[EntityOut](EntityOut.md)\]</code>) –
- [**model_config**](#ontogr_mem-service-contracts-EntityPage-model_config) –
- [**next_cursor**](#ontogr_mem-service-contracts-EntityPage-next_cursor) (<code>str | None</code>) –

## `entities` \{#ontogr_mem-service-contracts-EntityPage-entities}

```python
entities: list[EntityOut]
```

## `model_config` \{#ontogr_mem-service-contracts-EntityPage-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `next_cursor` \{#ontogr_mem-service-contracts-EntityPage-next_cursor}

```python
next_cursor: str | None = None
```
