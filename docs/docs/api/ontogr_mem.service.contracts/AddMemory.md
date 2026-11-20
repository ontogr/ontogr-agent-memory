---
title: ontogr_mem.service.contracts.AddMemory
sidebar_label: AddMemory
---

# `ontogr_mem.service.contracts.AddMemory` \{#ontogr_mem-service-contracts-AddMemory}

Bases: <code>[WriteRequest](WriteRequest.md)</code>

Store a fact or extract facts from text.

`kind` decides which fields the request needs: free text for message
and text, one triple for fact, and source text plus host facts for
sourced_facts. `speaker` is the actor id and display name; `role`
defaults to `user`. Content is limited to `MAX_CONTENT_BYTES` UTF-8
bytes. A sourced_facts request carries at most `MAX_HOST_FACTS` facts.

**Attributes:**

- [**content**](#ontogr_mem-service-contracts-AddMemory-content) (<code>str | None</code>) –
- [**facts**](#ontogr_mem-service-contracts-AddMemory-facts) (<code>list\[[HostFact](HostFact.md)\] | None</code>) –
- [**idempotency_key**](#ontogr_mem-service-contracts-AddMemory-idempotency_key) (<code>str | None</code>) –
- [**kind**](#ontogr_mem-service-contracts-AddMemory-kind) (<code>[MemoryKind](MemoryKind.md)</code>) –
- [**model_config**](#ontogr_mem-service-contracts-AddMemory-model_config) –
- [**object**](#ontogr_mem-service-contracts-AddMemory-object) (<code>str | None</code>) –
- [**reference_time**](#ontogr_mem-service-contracts-AddMemory-reference_time) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –
- [**relation**](#ontogr_mem-service-contracts-AddMemory-relation) (<code>str | None</code>) –
- [**role**](#ontogr_mem-service-contracts-AddMemory-role) (<code>Role</code>) –
- [**session_id**](#ontogr_mem-service-contracts-AddMemory-session_id) (<code>str | None</code>) –
- [**source_description**](#ontogr_mem-service-contracts-AddMemory-source_description) (<code>str</code>) –
- [**speaker**](#ontogr_mem-service-contracts-AddMemory-speaker) (<code>str | None</code>) –
- [**statement**](#ontogr_mem-service-contracts-AddMemory-statement) (<code>str | None</code>) –
- [**subject**](#ontogr_mem-service-contracts-AddMemory-subject) (<code>str | None</code>) –
- [**tenant**](#ontogr_mem-service-contracts-AddMemory-tenant) (<code>[TenantId](TenantId.md) | None</code>) –
- [**thread**](#ontogr_mem-service-contracts-AddMemory-thread) (<code>str | None</code>) –
- [**valid_from**](#ontogr_mem-service-contracts-AddMemory-valid_from) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –
- [**valid_to**](#ontogr_mem-service-contracts-AddMemory-valid_to) (<code>[UtcDatetime](UtcDatetime.md) | None</code>) –
- [**wait_seconds**](#ontogr_mem-service-contracts-AddMemory-wait_seconds) (<code>float</code>) –

## `content` \{#ontogr_mem-service-contracts-AddMemory-content}

```python
content: str | None = Field(default=None, description='Source text for text, message or sourced_facts.')
```

## `facts` \{#ontogr_mem-service-contracts-AddMemory-facts}

```python
facts: list[HostFact] | None = Field(default=None, max_length=MAX_HOST_FACTS, description=f'Host facts for kind=sourced_facts, in input order. At most {MAX_HOST_FACTS} facts.')
```

## `idempotency_key` \{#ontogr_mem-service-contracts-AddMemory-idempotency_key}

```python
idempotency_key: str | None = Field(default=None, max_length=128)
```

## `kind` \{#ontogr_mem-service-contracts-AddMemory-kind}

```python
kind: MemoryKind = MemoryKind.TEXT
```

## `model_config` \{#ontogr_mem-service-contracts-AddMemory-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `object` \{#ontogr_mem-service-contracts-AddMemory-object}

```python
object: str | None = Field(default=None, description='kind=fact: object name.')
```

## `reference_time` \{#ontogr_mem-service-contracts-AddMemory-reference_time}

```python
reference_time: UtcDatetime | None = Field(default=None, description='When the event happened. Defaults to now.')
```

## `relation` \{#ontogr_mem-service-contracts-AddMemory-relation}

```python
relation: str | None = Field(default=None, description='kind=fact: the relation label.')
```

## `role` \{#ontogr_mem-service-contracts-AddMemory-role}

```python
role: Role = Role.USER
```

## `session_id` \{#ontogr_mem-service-contracts-AddMemory-session_id}

```python
session_id: str | None = None
```

## `source_description` \{#ontogr_mem-service-contracts-AddMemory-source_description}

```python
source_description: str = ''
```

## `speaker` \{#ontogr_mem-service-contracts-AddMemory-speaker}

```python
speaker: str | None = Field(default=None, description='Actor id and display name. Defaults to the caller id.')
```

## `statement` \{#ontogr_mem-service-contracts-AddMemory-statement}

```python
statement: str | None = Field(default=None, description='kind=fact: the fact as a sentence.')
```

## `subject` \{#ontogr_mem-service-contracts-AddMemory-subject}

```python
subject: str | None = Field(default=None, description='kind=fact: subject name.')
```

## `tenant` \{#ontogr_mem-service-contracts-AddMemory-tenant}

```python
tenant: TenantId | None = None
```

## `thread` \{#ontogr_mem-service-contracts-AddMemory-thread}

```python
thread: str | None = None
```

## `valid_from` \{#ontogr_mem-service-contracts-AddMemory-valid_from}

```python
valid_from: UtcDatetime | None = None
```

## `valid_to` \{#ontogr_mem-service-contracts-AddMemory-valid_to}

```python
valid_to: UtcDatetime | None = None
```

## `wait_seconds` \{#ontogr_mem-service-contracts-AddMemory-wait_seconds}

```python
wait_seconds: float = Field(default=0.0, ge=0.0, le=60.0)
```
