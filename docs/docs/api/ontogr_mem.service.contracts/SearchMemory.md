---
title: ontogr_mem.service.contracts.SearchMemory
sidebar_label: SearchMemory
---

# `ontogr_mem.service.contracts.SearchMemory` \{#ontogr_mem-service-contracts-SearchMemory}

Bases: <code>[ContractModel](ContractModel.md)</code>

Recall facts, entities and a ready-to-prompt context.

The time fields are the retrieval request's filters: `valid_at` and
`known_at` ask about one instant, `overlaps_start` with
`overlaps_end` about a window. `session_id` limits facts to episodes
associated with that session.

**Attributes:**

- [**budget_tokens**](#ontogr_mem-service-contracts-SearchMemory-budget_tokens) (<code>int</code>) –
- [**disputed**](#ontogr_mem-service-contracts-SearchMemory-disputed) (<code>Literal['flag', 'exclude']</code>) –
- [**entity_types**](#ontogr_mem-service-contracts-SearchMemory-entity_types) (<code>list\[Annotated\[str, Field(min_length=1, max_length=128)\]\]</code>) –
- [**exclude_entity_types**](#ontogr_mem-service-contracts-SearchMemory-exclude_entity_types) (<code>list\[Annotated\[str, Field(min_length=1, max_length=128)\]\]</code>) –
- [**exclude_relation_types**](#ontogr_mem-service-contracts-SearchMemory-exclude_relation_types) (<code>list\[Annotated\[str, Field(min_length=1, max_length=128)\]\]</code>) –
- [**format**](#ontogr_mem-service-contracts-SearchMemory-format) (<code>Literal['json', 'prompt', 'both']</code>) –
- [**graph_depth**](#ontogr_mem-service-contracts-SearchMemory-graph_depth) (<code>int</code>) –
- [**include_history**](#ontogr_mem-service-contracts-SearchMemory-include_history) (<code>bool</code>) –
- [**known_at**](#ontogr_mem-service-contracts-SearchMemory-known_at) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –
- [**limit**](#ontogr_mem-service-contracts-SearchMemory-limit) (<code>int</code>) –
- [**mentions**](#ontogr_mem-service-contracts-SearchMemory-mentions) (<code>list\[str\]</code>) –
- [**modalities**](#ontogr_mem-service-contracts-SearchMemory-modalities) (<code>list\[Literal['actual', 'planned', 'conditional', 'desired']\]</code>) –
- [**model_config**](#ontogr_mem-service-contracts-SearchMemory-model_config) –
- [**overlaps_end**](#ontogr_mem-service-contracts-SearchMemory-overlaps_end) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –
- [**overlaps_start**](#ontogr_mem-service-contracts-SearchMemory-overlaps_start) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –
- [**query**](#ontogr_mem-service-contracts-SearchMemory-query) (<code>str</code>) –
- [**relation_types**](#ontogr_mem-service-contracts-SearchMemory-relation_types) (<code>list\[Annotated\[str, Field(min_length=1, max_length=128)\]\]</code>) –
- [**session_id**](#ontogr_mem-service-contracts-SearchMemory-session_id) (<code>str | None</code>) –
- [**tenant**](#ontogr_mem-service-contracts-SearchMemory-tenant) (<code>[TenantId](TenantId.md) | None</code>) –
- [**tenants**](#ontogr_mem-service-contracts-SearchMemory-tenants) (<code>list\[[TenantId](TenantId.md)\]</code>) –
- [**valid_at**](#ontogr_mem-service-contracts-SearchMemory-valid_at) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –

## `budget_tokens` \{#ontogr_mem-service-contracts-SearchMemory-budget_tokens}

```python
budget_tokens: int = Field(default=2000, ge=100, le=32000)
```

## `disputed` \{#ontogr_mem-service-contracts-SearchMemory-disputed}

```python
disputed: Literal['flag', 'exclude'] = 'flag'
```

## `entity_types` \{#ontogr_mem-service-contracts-SearchMemory-entity_types}

```python
entity_types: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(default_factory=list, max_length=MAX_FILTER_LABELS)
```

## `exclude_entity_types` \{#ontogr_mem-service-contracts-SearchMemory-exclude_entity_types}

```python
exclude_entity_types: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(default_factory=list, max_length=MAX_FILTER_LABELS)
```

## `exclude_relation_types` \{#ontogr_mem-service-contracts-SearchMemory-exclude_relation_types}

```python
exclude_relation_types: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(default_factory=list, max_length=MAX_FILTER_LABELS)
```

## `format` \{#ontogr_mem-service-contracts-SearchMemory-format}

```python
format: Literal['json', 'prompt', 'both'] = 'both'
```

## `graph_depth` \{#ontogr_mem-service-contracts-SearchMemory-graph_depth}

```python
graph_depth: int = Field(default=1, ge=1)
```

## `include_history` \{#ontogr_mem-service-contracts-SearchMemory-include_history}

```python
include_history: bool = False
```

## `known_at` \{#ontogr_mem-service-contracts-SearchMemory-known_at}

```python
known_at: UtcDatetime | None = None
```

## `limit` \{#ontogr_mem-service-contracts-SearchMemory-limit}

```python
limit: int = Field(default=10, ge=1, le=MAX_RESULTS)
```

## `mentions` \{#ontogr_mem-service-contracts-SearchMemory-mentions}

```python
mentions: list[str] = Field(default_factory=list)
```

## `modalities` \{#ontogr_mem-service-contracts-SearchMemory-modalities}

```python
modalities: list[Literal['actual', 'planned', 'conditional', 'desired']] = Field(default_factory=lambda: list(DEFAULT_MODALITIES), min_length=1)
```

## `model_config` \{#ontogr_mem-service-contracts-SearchMemory-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `overlaps_end` \{#ontogr_mem-service-contracts-SearchMemory-overlaps_end}

```python
overlaps_end: UtcDatetime | None = None
```

## `overlaps_start` \{#ontogr_mem-service-contracts-SearchMemory-overlaps_start}

```python
overlaps_start: UtcDatetime | None = None
```

## `query` \{#ontogr_mem-service-contracts-SearchMemory-query}

```python
query: str = Field(min_length=1, max_length=2000)
```

## `relation_types` \{#ontogr_mem-service-contracts-SearchMemory-relation_types}

```python
relation_types: list[Annotated[str, Field(min_length=1, max_length=128)]] = Field(default_factory=list, max_length=MAX_FILTER_LABELS)
```

## `session_id` \{#ontogr_mem-service-contracts-SearchMemory-session_id}

```python
session_id: str | None = None
```

## `tenant` \{#ontogr_mem-service-contracts-SearchMemory-tenant}

```python
tenant: TenantId | None = None
```

## `tenants` \{#ontogr_mem-service-contracts-SearchMemory-tenants}

```python
tenants: list[TenantId] = Field(default_factory=list)
```

## `valid_at` \{#ontogr_mem-service-contracts-SearchMemory-valid_at}

```python
valid_at: UtcDatetime | None = None
```
