---
title: ontogr_mem.service.contracts.validate_request
sidebar_label: validate_request
---

# `ontogr_mem.service.contracts.validate_request` \{#ontogr_mem-service-contracts-validate_request}

```python
validate_request(model:type[ModelT], data:Mapping[str, Any]) -> ModelT
```

Parse a raw request mapping into its contract model.

**Parameters:**

- **model** (<code>type\[[ModelT](ModelT.md)\]</code>) – The contract model of the verb.
- **data** (<code>Mapping\[str, Any\]</code>) – The raw request fields.

**Returns:**

- <code>[ModelT](ModelT.md)</code> – The validated model.

**Raises:**

- <code>InvalidArgumentError</code> – If the fields do not validate.
- <code>PayloadTooLargeError</code> – If a field is over its byte limit and the
  validation failed.
