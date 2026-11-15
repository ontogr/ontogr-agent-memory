# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Deduplication selects entities by their own revision checkpoints. Incomplete
  candidate discovery or verification leaves them eligible, and merge or
  unmerge writes advance revisions so changed active entities are reconsidered.
- Alias batches advance each owner revision in the same graph query as the alias
  write, so a checkpoint cannot hide a concurrent alias change.
- `ResolverStore.list_for_deduplication` no longer accepts `after_revision`.
  Custom adapters must use the keyword-only `provisional` and `sweep_limit`
  arguments and implement snapshot-conditional `mark_deduplication_checked`.
- The default consolidation gate no longer reopens for a checked provisional
  entity without a newer revision.

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
