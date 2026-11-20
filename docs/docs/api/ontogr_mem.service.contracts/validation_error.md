---
title: ontogr_mem.service.contracts.validation_error
sidebar_label: validation_error
---

# `ontogr_mem.service.contracts.validation_error` \{#ontogr_mem-service-contracts-validation_error}

```python
validation_error(errors:Sequence[Mapping[str, Any]]) -> ServiceError
```

Turn pydantic error dicts into one typed service error.

**Parameters:**

- **errors** (<code>Sequence\[Mapping\[str, Any\]\]</code>) – The error dicts of a pydantic `ValidationError`.

**Returns:**

- <code>ServiceError</code> – PayloadTooLargeError when any field was over its byte limit, so the
- <code>ServiceError</code> – caller shrinks the content first; otherwise InvalidArgumentError.
- <code>ServiceError</code> – Both carry every field path and message.
