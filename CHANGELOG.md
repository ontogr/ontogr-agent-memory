# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Temporal writes store events and planned, conditional, desired, and unplaced facts. Events and actual states appear in default reads; other modalities stay hidden by default.
- The closure judge checks candidates once per new actual state fact during text and message writes. A contradiction closes the earlier fact by date, and a failed call leaves both facts live.
- Write results report stored facts without a relation type in `unplaced`.
- Entity resolution uses ontology type ids and inherited identity attributes to decide which entities can match or merge.
- Entities expose stable ontology type ids and `proposed_type` when the model's type name cannot be matched. Such entities use the `Unknown` type.
- `MEMORY_EXTRACTION_MODE` selects `two_pass` (the default) or `combined` extraction. Combined mode requests entities and facts in one pass. Invalid model replies can trigger retries.
- `Memory.open(ontology=...)` accepts a YAML, JSON or Turtle file path, a list of
  Pydantic classes or an `OntologyDeclaration`. The `chat` and `mcp` commands
  accept `--ontology`, and MCP also reads `MEMORY_MCP_ONTOLOGY`.
- A default ontology with ten entity types and 31 relation types.
- Relation kinds (`single`, `many`, `judged`) and entity identity attributes.
- Search accepts entity type, relation type, and modality filters. Entity type filters include descendants, and ontology labels and aliases resolve per tenant.
- `Memory.list_entities()` pages through entities by type, and `Memory.get_ontology()` returns the tenant's type catalog. MCP exposes both read tools.
- `Memory.stale_episodes()` reports message and document episodes extracted under an older ontology version. Refresh an episode by writing its text again and forgetting the old episode.
- `MEMORY_RETRIEVAL_FILTER_RECHECKS` bounds extra candidate windows fetched to fill search results after type filtering. Set it to `0` to disable refills.

### Changed

- In text and sourced writes, an entity type that the ontology does not hold is kept with the Unknown type.
- Facts with unknown relation labels or relation pattern violations are stored as unplaced facts. Facts without a usable stated time start at the episode time with basis `reference`.
- Custom resolver integrations must accept a `TypeLinks` argument in `resolve`, `deduplicate`, and `merge`, and use `resolver.linked(links)` to bind one snapshot for consolidation; `Entity` construction now requires `ontology_type_id`.
- People verification is off by default.
- The entity extraction prompt uses ontology type ids and records the model's type name in `proposed_type` when no id matches.
- Relations use a kind in place of the `exclusive` flag. PyYAML and RDFLib are
  core dependencies.

- Deduplication selects entities by their own revision checkpoints. Incomplete
  candidate discovery or verification leaves them eligible, and merge or
  unmerge writes advance revisions so changed active entities are reconsidered.
- Alias batches advance each owner revision in the same graph query as the alias
  write, so a checkpoint cannot hide a concurrent alias change. Batches reject
  ownership changes; merge and unmerge move aliases with revision checks.
- Stale merge plans are rejected, and merge recovery keeps an intent applying
  when newer entity revisions or checkpoints prevent a safe replay.
- Graph entity writes expose `ConcurrentEntityWriteError` when a stale full-row
  update would replace a newer revision or checkpoint.
- `ResolverStore.list_for_deduplication` no longer accepts `after_revision`.
  Custom adapters must use the keyword-only `provisional` and `sweep_limit`
  arguments and implement snapshot-conditional `mark_deduplication_checked`.
- The default consolidation gate no longer reopens for a checked provisional
  entity without a newer revision.

### Removed

- The pair judge and its pending-slot workflow. The `ontology_refused` result field and `unknown_relation` drop reason. `MEMORY_JUDGE_` closure settings are replaced by `MEMORY_CLOSURE_` settings.
- The `DEFAULT_CATALOG` change set and `OntologyManager.load_catalog`.

- Ontology modes (`strict`, `preferred` and `open`) and the `SetModes` change.
- Ontology proposals, the drafting model role (`ONTOLOGY`) with its
  `MEMORY_LLM_ONTOLOGY_*` and `MEMORY_ONTOLOGY_*` settings, the stored unmatched
  rows, and `OntologyManager.stale_episodes`.
- Declared type attributes, table mappings, type URIs and the memory entity
  extractor.
- The `single_valued` field of `AddMemory` and the creation of relations by
  structured writes.

## [0.0.1](https://github.com/ontogr/ontogr-agent-memory/tree/v0.0.1) - 2026-10-27

First release.

### Added

- Episode ingestion for messages, documents, tool outputs, events, and records. Each episode keeps its source, actor, and content hash.
- Entity extraction with LLM, GLiNER, and spaCy extractors, and fact extraction that keeps a verbatim quote for each fact.
- A versioned ontology for each tenant, with strict, preferred, and open modes for unknown types.
- Entity resolution that checks the cheapest evidence first, and merges that can be undone.
- Facts on two time axes, valid time and system time, with a hash-chained ledger of temporal writes.
- Contradiction handling by rules, an optional model judge, and a person's decision.
- Four-lane retrieval with rank fusion, an optional reranker, and a budgeted context block.
- Consolidation in nine ordered stages under a per-tenant lease.
- Access control with grants for each tenant, a hash-chained audit log, and forget, erase, and export workflows.
- Storage on FalkorDB or Neo4j for the graph, graph-native vectors or pgvector, and SQLite or Postgres for jobs and audit rows.
- `MemoryService`, an async facade for the write, read, forget, and export verbs.
- OpenTelemetry tracing through the optional `otel` extra.
