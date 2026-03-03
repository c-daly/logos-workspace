# Cognitive Loop Implementation Plan — Spec Review

**Reviewer:** Staff Engineer  
**Date:** 2026-02-18  
**Spec under review:** `2026-02-18-cognitive-loop-implementation.md`  
**Review criteria:** Is this spec complete enough for a competent engineer to implement without questions?

---

## Overall Verdict

This is well above average for an implementation plan. The dependency graph is clear, the TDD cadence is enforced consistently, the code samples are concrete, the architectural constraints are repeated where they matter, and the "Standards for the Implementer" section upfront is unusually good — most plans skip this entirely and then wonder why someone rewrites `HCGMilvusSync` from scratch.

That said, there are real gaps that would generate questions, incorrect assumptions that would cause wasted time, and a few places where the plan hand-waves at exactly the moment precision matters most. Below is everything I'd flag in a design review, organized by severity.

---

## Critical Issues (would block or derail implementation)

### 1. `add_edge()` uses `CREATE` instead of `MERGE` — no idempotency

`add_node()` correctly uses `MERGE` (line 424), but `add_edge()` (line 482) uses `CREATE`. If the seeder runs twice, or if Hermes proposes the same entity relationship in two turns, you get duplicate edge nodes. This is a data integrity bug baked into Task 1 that will silently corrupt the graph.

The fix is to either MERGE on a composite key (source + target + relation), or to add a deduplication check before CREATE. You need to decide the semantics: is "Paris LOCATED_IN France" proposed twice the same edge or two observations? The spec doesn't say, and this decision affects the entire ingestion pipeline.

### 2. `ProposalProcessor.process()` is synchronous but the Sophia API is async FastAPI

The processor at line 1770 is `def process(self, proposal)` — a sync method. But it calls `self._hcg.add_node()` and `self._milvus.search_similar()`, and in Task 6 Step 5 (line 1878) it's called inside what is presumably an `async def` endpoint handler. If the HCG client or Milvus client use blocking I/O (which Neo4j's Python driver does by default), this blocks the FastAPI event loop.

The spec needs to either make `process()` async, or explicitly call it via `run_in_executor`, or confirm the underlying clients are async-safe. As written, an implementer will wire this up, it'll "work" in smoke testing, and then deadlock under any concurrent load.

### 3. The `has_ancestor()` query is a combinatorial explosion that won't scale

The unrolled 5-level `OR EXISTS` pattern (lines 843–871) generates 5 separate subqueries, each doing increasingly deep pattern matches. For the current ontology at depth ~3, it works. But the `_is_a_check` utility at line 691 generates this same pattern dynamically and is used in query rewriting throughout. Neo4j's planner will struggle with this at larger graph sizes because each EXISTS clause is planned independently.

The spec should acknowledge this is a known performance tradeoff, specify that APOC path expanders (`apoc.path.subgraphNodes`) are the proper solution for variable-depth traversal, and either use them now or create a follow-up ticket. Without this, the first time someone adds a deeper type hierarchy, query performance will cliff.

### 4. No entity resolution / deduplication strategy

Hermes extracts "Paris" as a GPE entity from spaCy. Sophia creates a node named "Paris" with type "GPE". Next turn, Hermes extracts "Paris" again. What happens? `add_node()` uses `MERGE (n:Node {uuid: $uuid})` — but the UUID is freshly generated each time (line 410–411), so every mention of Paris creates a **new node**. The graph fills with duplicate Paris nodes that never connect.

This is the single biggest functional gap in the spec. The ProposalProcessor needs either name+type deduplication (MERGE on name+type), embedding-similarity deduplication (find existing node within threshold before creating), or explicit entity linking. Without it, the cognitive loop doesn't actually work — Turn 2 won't find Turn 1's knowledge because the embeddings are scattered across duplicate nodes.

### 5. `search_similar()` uses L2 metric but the spec never establishes index creation

The Milvus `search_similar()` (line 544) hardcodes `"metric_type": "L2"`. But Milvus requires the collection's index to match the search metric type. The spec never shows index creation for the new Edge collection, and doesn't verify existing collections use L2 indexes. If a collection was created with IP (inner product) or COSINE metric, the search silently returns wrong results.

The spec should include explicit index creation params for each collection, or at minimum a verification step.

---

## Significant Issues (would generate questions or cause rework)

### 6. Task 4 (Sophia Alignment) is underspecified compared to every other task

Tasks 1–3 and 5 have complete code samples for every function. Task 4 has partial implementations (`query_edges_from` at line 1370 trails off with `# ... decode properties as before`, `get_subgraph` at line 1391 says `# ... format as {"nodes": [...], "edges": [...]}`) and multiple hand-waves: "Update `query_neighbors()` — use edge node traversal" (line 1394), "Rewrite `list_all_edges()` — query edge nodes" (line 1396), and the SHACL validator section (lines 1398–1403) is three bullet points with no code.

Task 4 touches the most files (8) across the most complex codebase (sophia). An implementer will spend more time reverse-engineering sophia's existing code than on any other task, and this is exactly where the spec gets vague. Provide complete implementations or at least full method signatures with Cypher queries for every method being rewritten.

### 7. No specification of the `upsert_embedding()` method signature

Task 6's `ProposalProcessor` calls `self._milvus.upsert_embedding(node_type, uuid, embedding, model)` at line 1837. Task 1 adds `search_similar()` to `HCGMilvusSync` but never mentions `upsert_embedding()`. Does it already exist? With what signature? The spec says to use existing code, but never confirms this method exists or what params it takes. If it doesn't exist, the implementer has to write it, and the spec gives no guidance on collection schema, embedding dimension validation, or update-vs-insert semantics.

### 8. The `_get_sophia_context()` function swallows all errors silently

Lines 2001–2017: the entire Sophia interaction is wrapped in a bare `except Exception` that returns `[]`. This is fine for resilience, but there's no distinction between transient failures (connection timeout), configuration errors (missing token → returns `[]` at line 1999 with no log), and data errors (malformed response). At minimum, log at different levels: WARNING for transient failures, ERROR for config issues, DEBUG for empty responses. As written, debugging why context never arrives will be painful.

### 9. `_build_context_message()` property filtering is hardcoded and fragile

Lines 2042–2044 filter out `source`, `derivation`, `confidence`, `raw_text`, `created_at`, `updated_at` from context properties. This hardcoded exclusion list will break silently when new properties are added to nodes. Better to use an inclusion list (show `description`, `value`, `unit`, `label`) or have Sophia return a pre-formatted `display_properties` dict so Hermes doesn't need to know Sophia's internal schema.

### 10. No rollback or cleanup strategy for partial failures in ProposalProcessor

If `add_node()` succeeds for 2 of 3 proposed nodes but the third fails (line 1831), and then Milvus embedding storage fails for the first node (line 1843), the graph is in an inconsistent state: nodes exist without embeddings, some nodes ingested and others not. The spec should specify whether partial ingestion is acceptable (and document that it is), or whether the processor should use transactions to ensure atomicity.

### 11. The smoke test is insufficient as an acceptance test

The integration test (lines 2104–2118) sends two curl requests and visually inspects the response. There's no assertion script, no automated verification, no check that the graph actually contains the right nodes and edges. For a spec that emphasizes "evidence before assertions," the final validation is entirely manual. Provide a test script that asserts specific conditions: node count, edge existence, context array non-empty, LLM response containing expected terms.

---

## Minor Issues (quality and robustness)

### 12. `Edge.name` is set to `relation` in `add_edge()`

Line 468 sets `"name": relation`. This means every IS_A edge node is named "IS_A", making name-based lookups on edge nodes useless. Consider making the name more descriptive (e.g., `f"{source_name}_IS_A_{target_name}"`) or document why this is intentional.

### 13. The `_n()` helper creates IS_A edges to type definitions unconditionally

Line 1098 creates an IS_A edge from every instance to its type definition. But the type definition node UUID is constructed as `f"type_{node_type}"`, and if the node_type doesn't match a seeded type (e.g., a Hermes-proposed node with type "GPE"), this will fail silently or create a dangling edge. The spec doesn't address how Hermes's spaCy entity labels (GPE, FAC, ORG, PERSON) map to the ontology's type definitions (entity, location, agent, etc.).

### 14. No mention of Neo4j indexes

The queries rely on `Node.uuid`, `Node.name`, `Node.type`, and `Node.relation` for lookups. Without explicit index creation (at minimum a uniqueness constraint on `uuid` and composite indexes on `type`+`name`), query performance on any non-trivial graph will be poor. Add index creation to the seeder or to a migration step.

### 15. Test fixtures use hardcoded Neo4j credentials

Every test file hardcodes `password="logosdev"`. This should come from environment variables or a shared conftest fixture to avoid credential sprawl.

### 16. `BOOTSTRAP_TYPES` and `EDGE_TYPES` are referenced but never defined

The seeder rewrite (lines 1053, 1063, 1078) references `BOOTSTRAP_TYPES` and `EDGE_TYPES` constants that are presumably in the existing seeder but are never shown. An implementer needs to know what these contain to understand the control flow.

### 17. No versioning or migration path for existing graph data

If a dev environment has existing nodes with `ancestors` properties and native relationships, there's no migration script to convert them. The spec says "clear and reseed" implicitly (`--clear` flag in the smoke test), but doesn't say this explicitly as a requirement, and doesn't address non-dev environments.

---

## What's Done Well

To be clear about what doesn't need changes: the dependency graph and parallelization guidance is excellent. The "Standards for the Implementer" section would prevent the top 3 mistakes I see coding agents make. The reified edge model explanation with the query transformation table is clear enough that I'd trust a mid-level engineer to apply it mechanically. The TDD cadence (write failing test → verify failure → implement → verify pass → commit) is enforced consistently across all 7 tasks. And the architecture section correctly separates concerns between Hermes (language) and Sophia (structure) in a way that's easy to reason about.

The spec is about 85% of the way to "implement without questions." Fixing issues #1, #4, and #6 would get it to 95%.
