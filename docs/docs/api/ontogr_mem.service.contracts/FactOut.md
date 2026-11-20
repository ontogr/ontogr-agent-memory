---
title: ontogr_mem.service.contracts.FactOut
sidebar_label: FactOut
---

# `ontogr_mem.service.contracts.FactOut` \{#ontogr_mem-service-contracts-FactOut}

Bases: <code>[ContractModel](ContractModel.md)</code>

One fact with its validity window and provenance.

**Attributes:**

- [**counterpart_ids**](#ontogr_mem-service-contracts-FactOut-counterpart_ids) (<code>list\[str\]</code>) –
- [**disputed**](#ontogr_mem-service-contracts-FactOut-disputed) (<code>bool</code>) –
- [**episode_ids**](#ontogr_mem-service-contracts-FactOut-episode_ids) (<code>list\[str\]</code>) –
- [**evidence_count**](#ontogr_mem-service-contracts-FactOut-evidence_count) (<code>int</code>) –
- [**expired_at**](#ontogr_mem-service-contracts-FactOut-expired_at) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –
- [**fact_id**](#ontogr_mem-service-contracts-FactOut-fact_id) (<code>str</code>) –
- [**kind**](#ontogr_mem-service-contracts-FactOut-kind) (<code>str</code>) –
- [**modality**](#ontogr_mem-service-contracts-FactOut-modality) (<code>str</code>) –
- [**model_config**](#ontogr_mem-service-contracts-FactOut-model_config) –
- [**object**](#ontogr_mem-service-contracts-FactOut-object) (<code>[EntityRef](EntityRef.md)</code>) –
- [**recorded_at**](#ontogr_mem-service-contracts-FactOut-recorded_at) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –
- [**relation**](#ontogr_mem-service-contracts-FactOut-relation) (<code>str</code>) –
- [**score**](#ontogr_mem-service-contracts-FactOut-score) (<code>float</code>) –
- [**statement**](#ontogr_mem-service-contracts-FactOut-statement) (<code>str</code>) –
- [**status**](#ontogr_mem-service-contracts-FactOut-status) (<code>str</code>) –
- [**subject**](#ontogr_mem-service-contracts-FactOut-subject) (<code>[EntityRef](EntityRef.md)</code>) –
- [**tenant**](#ontogr_mem-service-contracts-FactOut-tenant) (<code>[TenantId](TenantId.md)</code>) –
- [**time_basis**](#ontogr_mem-service-contracts-FactOut-time_basis) (<code>str</code>) –
- [**valid_from**](#ontogr_mem-service-contracts-FactOut-valid_from) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –
- [**valid_to**](#ontogr_mem-service-contracts-FactOut-valid_to) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –

## `counterpart_ids` \{#ontogr_mem-service-contracts-FactOut-counterpart_ids}

```python
counterpart_ids: list[str] = Field(default_factory=list)
```

## `disputed` \{#ontogr_mem-service-contracts-FactOut-disputed}

```python
disputed: bool = False
```

## `episode_ids` \{#ontogr_mem-service-contracts-FactOut-episode_ids}

```python
episode_ids: list[str] = Field(default_factory=list)
```

## `evidence_count` \{#ontogr_mem-service-contracts-FactOut-evidence_count}

```python
evidence_count: int = 0
```

## `expired_at` \{#ontogr_mem-service-contracts-FactOut-expired_at}

```python
expired_at: UtcDatetime | None = None
```

## `fact_id` \{#ontogr_mem-service-contracts-FactOut-fact_id}

```python
fact_id: str
```

## `kind` \{#ontogr_mem-service-contracts-FactOut-kind}

```python
kind: str = 'state'
```

## `modality` \{#ontogr_mem-service-contracts-FactOut-modality}

```python
modality: str = 'actual'
```

## `model_config` \{#ontogr_mem-service-contracts-FactOut-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `object` \{#ontogr_mem-service-contracts-FactOut-object}

```python
object: EntityRef
```

## `recorded_at` \{#ontogr_mem-service-contracts-FactOut-recorded_at}

```python
recorded_at: UtcDatetime | None = None
```

## `relation` \{#ontogr_mem-service-contracts-FactOut-relation}

```python
relation: str = ''
```

## `score` \{#ontogr_mem-service-contracts-FactOut-score}

```python
score: float = 0.0
```

## `statement` \{#ontogr_mem-service-contracts-FactOut-statement}

```python
statement: str
```

## `status` \{#ontogr_mem-service-contracts-FactOut-status}

```python
status: str = 'active'
```

## `subject` \{#ontogr_mem-service-contracts-FactOut-subject}

```python
subject: EntityRef
```

## `tenant` \{#ontogr_mem-service-contracts-FactOut-tenant}

```python
tenant: TenantId
```

## `time_basis` \{#ontogr_mem-service-contracts-FactOut-time_basis}

```python
time_basis: str = 'stated'
```

## `valid_from` \{#ontogr_mem-service-contracts-FactOut-valid_from}

```python
valid_from: UtcDatetime | None = None
```

## `valid_to` \{#ontogr_mem-service-contracts-FactOut-valid_to}

```python
valid_to: UtcDatetime | None = None
```
