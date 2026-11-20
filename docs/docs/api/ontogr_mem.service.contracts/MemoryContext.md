---
title: ontogr_mem.service.contracts.MemoryContext
sidebar_label: MemoryContext
---

# `ontogr_mem.service.contracts.MemoryContext` \{#ontogr_mem-service-contracts-MemoryContext}

Bases: <code>[ContractModel](ContractModel.md)</code>

Search output. `state` separates "nothing stored" from "not ready".

**Attributes:**

- [**degraded**](#ontogr_mem-service-contracts-MemoryContext-degraded) (<code>list\[str\]</code>) –
- [**entities**](#ontogr_mem-service-contracts-MemoryContext-entities) (<code>list\[[EntityOut](EntityOut.md)\]</code>) –
- [**facts**](#ontogr_mem-service-contracts-MemoryContext-facts) (<code>list\[[FactOut](FactOut.md)\]</code>) –
- [**model_config**](#ontogr_mem-service-contracts-MemoryContext-model_config) –
- [**rendered**](#ontogr_mem-service-contracts-MemoryContext-rendered) (<code>str | None</code>) –
- [**state**](#ontogr_mem-service-contracts-MemoryContext-state) (<code>Literal['found', 'none', 'indexing', 'degraded']</code>) –
- [**tokens_used**](#ontogr_mem-service-contracts-MemoryContext-tokens_used) (<code>int</code>) –
- [**trace_id**](#ontogr_mem-service-contracts-MemoryContext-trace_id) (<code>str</code>) –

## `degraded` \{#ontogr_mem-service-contracts-MemoryContext-degraded}

```python
degraded: list[str] = Field(default_factory=list)
```

## `entities` \{#ontogr_mem-service-contracts-MemoryContext-entities}

```python
entities: list[EntityOut] = Field(default_factory=list)
```

## `facts` \{#ontogr_mem-service-contracts-MemoryContext-facts}

```python
facts: list[FactOut] = Field(default_factory=list)
```

## `model_config` \{#ontogr_mem-service-contracts-MemoryContext-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `rendered` \{#ontogr_mem-service-contracts-MemoryContext-rendered}

```python
rendered: str | None = None
```

## `state` \{#ontogr_mem-service-contracts-MemoryContext-state}

```python
state: Literal['found', 'none', 'indexing', 'degraded']
```

## `tokens_used` \{#ontogr_mem-service-contracts-MemoryContext-tokens_used}

```python
tokens_used: int = 0
```

## `trace_id` \{#ontogr_mem-service-contracts-MemoryContext-trace_id}

```python
trace_id: str = ''
```
