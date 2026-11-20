---
title: ontogr_mem.service.contracts.HostFact
sidebar_label: HostFact
---

# `ontogr_mem.service.contracts.HostFact` \{#ontogr_mem-service-contracts-HostFact}

Bases: <code>[ContractModel](ContractModel.md)</code>

A host-extracted link supported by the accompanying source text.

Every text field holds at most `MAX_HOST_FACT_CHARS` characters.

**Attributes:**

- [**subject**](#ontogr_mem-service-contracts-HostFact-subject) (<code>str</code>) – Subject name as written in the source.
- [**subject_type**](#ontogr_mem-service-contracts-HostFact-subject_type) (<code>str</code>) – Ontology label for the subject type.
- [**relation**](#ontogr_mem-service-contracts-HostFact-relation) (<code>str</code>) – Ontology label for the directed relation.
- [**object**](#ontogr_mem-service-contracts-HostFact-object) (<code>str</code>) – Object name as written in the source.
- [**object_type**](#ontogr_mem-service-contracts-HostFact-object_type) (<code>str</code>) – Ontology label for the object type.
- [**evidence_quote**](#ontogr_mem-service-contracts-HostFact-evidence_quote) (<code>str</code>) – Verbatim supporting text from this source.
- [**statement**](#ontogr_mem-service-contracts-HostFact-statement) (<code>str | None</code>) – Optional sentence expressing the fact.
- [**subject_attributes**](#ontogr_mem-service-contracts-HostFact-subject_attributes) (<code>dict\[str, str\]</code>) – Identity values of the subject, by attribute name.
  Each value must occur in the source text.
- [**object_attributes**](#ontogr_mem-service-contracts-HostFact-object_attributes) (<code>dict\[str, str\]</code>) – Identity values of the object, by attribute name.
  Each value must occur in the source text.
- [**valid_from**](#ontogr_mem-service-contracts-HostFact-valid_from) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) – Optional UTC start of validity.
- [**valid_to**](#ontogr_mem-service-contracts-HostFact-valid_to) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) – Optional UTC end of validity.

## `evidence_quote` \{#ontogr_mem-service-contracts-HostFact-evidence_quote}

```python
evidence_quote: str = Field(max_length=MAX_HOST_FACT_CHARS)
```

## `model_config` \{#ontogr_mem-service-contracts-HostFact-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `object` \{#ontogr_mem-service-contracts-HostFact-object}

```python
object: str = Field(max_length=MAX_HOST_FACT_CHARS)
```

## `object_attributes` \{#ontogr_mem-service-contracts-HostFact-object_attributes}

```python
object_attributes: dict[str, str] = Field(default_factory=dict)
```

## `object_type` \{#ontogr_mem-service-contracts-HostFact-object_type}

```python
object_type: str = Field(max_length=MAX_HOST_FACT_CHARS)
```

## `relation` \{#ontogr_mem-service-contracts-HostFact-relation}

```python
relation: str = Field(max_length=MAX_HOST_FACT_CHARS)
```

## `statement` \{#ontogr_mem-service-contracts-HostFact-statement}

```python
statement: str | None = Field(default=None, max_length=MAX_HOST_FACT_CHARS)
```

## `subject` \{#ontogr_mem-service-contracts-HostFact-subject}

```python
subject: str = Field(max_length=MAX_HOST_FACT_CHARS)
```

## `subject_attributes` \{#ontogr_mem-service-contracts-HostFact-subject_attributes}

```python
subject_attributes: dict[str, str] = Field(default_factory=dict)
```

## `subject_type` \{#ontogr_mem-service-contracts-HostFact-subject_type}

```python
subject_type: str = Field(max_length=MAX_HOST_FACT_CHARS)
```

## `valid_from` \{#ontogr_mem-service-contracts-HostFact-valid_from}

```python
valid_from: UtcDatetime | None = None
```

## `valid_to` \{#ontogr_mem-service-contracts-HostFact-valid_to}

```python
valid_to: UtcDatetime | None = None
```
