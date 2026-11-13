---
name: memory-write
description: Saves source-backed facts through memory tools. Use when the user asks to remember information or a task requires an authorized memory write.
---

# Saving memory

Use the tools listed by the connected memory server. The host is the application running your model. In keyless mode the server has no extraction model, so you supply the source text and the facts you extracted from it.

## Choose the write

- If `add_memory` is listed, it can extract facts from the source text. Use its `content` argument for an ordinary text write.
- If you extract the facts yourself, use `write_facts` with `source_text` and `facts`. This tool also works when `add_memory` is absent.
- If neither write tool is listed, report that this connection does not allow saving memory.

## Prepare a sourced write

Keep the source text unchanged. Use names that occur in this source for `subject` and `object`. Supply their entity type labels and the directed relation label. Copy an `evidence_quote` from this same source that supports the link. Do not invent a fact, name, type or timestamp to fill a field.

Each fact has `subject`, `subject_type`, `relation`, `object`, `object_type` and `evidence_quote`. `statement`, `valid_from` and `valid_to` are optional. Use UTC timestamps only when the source supplies their meaning. Omit an unknown validity interval.

Example arguments for `write_facts`:

```json
{
  "source_text": "Ana works at Acme.",
  "facts": [{
    "subject": "Ana",
    "subject_type": "Person",
    "relation": "works_at",
    "object": "Acme",
    "object_type": "Organization",
    "evidence_quote": "Ana works at Acme."
  }]
}
```

The example labels are valid only if this server's ontology accepts them. A preferred unknown relation is recorded for review and its fact is dropped. That review record does not mean the fact was stored. Do not change labels repeatedly to bypass a refusal.

## Check the outcome

A receipt can describe a queued job. Use `get_job` with its `job_id` until it is terminal before saying the facts were saved. If `get_job` is unavailable, report the job id and that completion is unconfirmed.

Read `result.fact_ids` and `result.dropped`. A dropped `index` is zero-based in your submitted facts list. Explain dropped items and their reasons. The source Episode can be retained even when every fact is dropped. Do not describe that as saving all facts.

For a retry of the same write, reuse the same `idempotency_key` if you supplied one. Do not resubmit a job merely because it is still queued. If the connection fails after submission, inspect the job before retrying when its id is known.

To forget a known fact, use `forget_memory` only when listed and authorized by the user's request. It closes validity and keeps history. It does not erase the source or delete every copy of the data.
