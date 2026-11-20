---
title: ontogr_mem.client.EntityPage
sidebar_label: EntityPage
---

# `ontogr_mem.client.EntityPage` \{#ontogr_mem-client-EntityPage}

Bases: <code>[ContractModel](../ontogr_mem.service.contracts/ContractModel.md)</code>

One page of active entities.

**Attributes:**

- [**entities**](#ontogr_mem-client-EntityPage-entities) (<code>list\[[EntityOut](../ontogr_mem.service.contracts/EntityOut.md)\]</code>) –
- [**model_config**](#ontogr_mem-client-EntityPage-model_config) –
- [**next_cursor**](#ontogr_mem-client-EntityPage-next_cursor) (<code>str | None</code>) –

## `entities` \{#ontogr_mem-client-EntityPage-entities}

```python
entities: list[EntityOut]
```

## `model_config` \{#ontogr_mem-client-EntityPage-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `next_cursor` \{#ontogr_mem-client-EntityPage-next_cursor}

```python
next_cursor: str | None = None
```
