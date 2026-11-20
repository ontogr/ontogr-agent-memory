---
title: ontogr_mem.service.contracts.ErrorInfo
sidebar_label: ErrorInfo
---

# `ontogr_mem.service.contracts.ErrorInfo` \{#ontogr_mem-service-contracts-ErrorInfo}

Bases: <code>[ContractModel](ContractModel.md)</code>

The wire form of a service error.

**Functions:**

- [**from_error**](#ontogr_mem-service-contracts-ErrorInfo-from_error) – Build the wire form from a raised service error.

**Attributes:**

- [**code**](#ontogr_mem-service-contracts-ErrorInfo-code) (<code>ErrorCode</code>) –
- [**hint**](#ontogr_mem-service-contracts-ErrorInfo-hint) (<code>str</code>) –
- [**message**](#ontogr_mem-service-contracts-ErrorInfo-message) (<code>str</code>) –
- [**model_config**](#ontogr_mem-service-contracts-ErrorInfo-model_config) –

## `code` \{#ontogr_mem-service-contracts-ErrorInfo-code}

```python
code: ErrorCode
```

## `from_error` \{#ontogr_mem-service-contracts-ErrorInfo-from_error}

```python
from_error(error:ServiceError) -> ErrorInfo
```

Build the wire form from a raised service error.

**Parameters:**

- **error** (<code>ServiceError</code>) – The error to translate.

**Returns:**

- <code>ErrorInfo</code> – The same code, message and hint as a model.

## `hint` \{#ontogr_mem-service-contracts-ErrorInfo-hint}

```python
hint: str = ''
```

## `message` \{#ontogr_mem-service-contracts-ErrorInfo-message}

```python
message: str
```

## `model_config` \{#ontogr_mem-service-contracts-ErrorInfo-model_config}

```python
model_config = ConfigDict(extra='forbid')
```
