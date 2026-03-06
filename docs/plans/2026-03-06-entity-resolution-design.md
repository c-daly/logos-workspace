# Entity Resolution Design — logos #503

**Date**: 2026-03-06
**Status**: Design approved
**Parent**: logos #499 — KG Maintenance
**Components**: sophia, hermes

## Goal

Give Sophia the ability to detect that separate nodes refer to the same entity and resolve them through hypothesis edges with confidence accumulation. Entity resolution is the first application of a broader capability: detecting meaningful geometric patterns in embedding space.

## Core Principle

The structure is the information. Sophia reasons over geometry, not words. Detection signals come from the embedding space and graph topology. Language (Hermes) is consulted only for translation — naming what Sophia has already understood structurally.

## Detection

Two independent entry points, both feeding into hypothesis creation:

### 1. Embedding Triage

Cosine similarity (via `milvus.search_similar()`) flags pairs of nodes that are close in embedding space. This is a cheap first indicator that something is worth investigating, not a decision signal.

- Triggered **post-ingestion**: new nodes checked against existing nodes
- Triggered **periodically**: broader scan for pairs that drifted into similarity as the graph evolved

### 2. Structural Triage

Graph analysis flags pairs that share relationship signatures, regardless of embedding distance. Two nodes that connect to the same targets via the same relation types are suspicious even if their embeddings are far apart.

- Shared relationship targets (N >= 2 non-trivial shared targets)
- Complementary edges (A has edges B doesn't, with no contradictions)
- Recurring geometric motifs — the same topological shape appearing across different regions of the embedding space, independent of node type

### Future: Geometric Pattern Detection

The detection signals above are starting points. The real capability is type-agnostic geometric pattern detection: finding recurring structures in embedding space without reference to node types or labels. Entity resolution is one consumer; relationship inference, ontology evolution, and type correction are others. This capability emerges as the maintenance jobs get built and the embeddings improve.

## Hypothesis Edges

Detection does not trigger merges. It creates hypothesis edges.

When Sophia detects a candidate pair, she creates a `POSSIBLE_ALIAS_OF` reified edge with:
- **confidence**: initial rating based on detection signal strength
- **evidence**: which signals triggered (embedding proximity, structural overlap, etc.)
- **created_at**: when the hypothesis was formed

### Confidence Accumulation

Confidence on hypothesis edges changes over time:
- **Corroborated**: new evidence supports the alias (new shared relationships arrive, embedding distance decreases) — confidence goes up
- **Contradicted**: evidence against (conflicting relationships, diverging embeddings) — confidence goes down
- **Stale**: no new evidence either way — no change (or optional slow decay)

### Using Hypotheses Before Merge

Hypothesis edges are usable knowledge. If a `POSSIBLE_ALIAS_OF` edge has high confidence, Sophia can reason as if the nodes are the same entity, weighted by confidence. The merge is cleanup, not the moment the information becomes available.

### Merge

When confidence on a hypothesis edge is high enough — and supported by topological similarity in the embedding space (not just a scalar threshold) — the merge happens:

- Canonical (surviving) node absorbs all edges from consumed node
- Surviving node gets `merge_history` recording what was merged, when, and evidence
- Hermes determines canonical name via `/alias-check` (language judgment)
- Consumed node is removed
- `ALIAS_OF` edge retained for audit trail
- Embeddings updated: consumed node's embedding removed from Milvus

The merge decision logic is designed to be swappable. Initial implementation uses confidence threshold + structural overlap. When the universal encoder lands and embeddings support richer geometric analysis, the decision logic upgrades without changing the pipeline.

## Hermes Endpoint

**POST /alias-check**

Sophia sends structured data (node names, types, relationships) — not natural language. Hermes uses LLM judgment to determine if two entities are aliases.

Request:
- `entity_a`: name, type, relationship summary
- `entity_b`: name, type, relationship summary
- `context`: optional additional context

Response:
- `is_alias`: bool
- `confidence`: float (LLM's confidence in the judgment)
- `canonical_name`: suggested surviving name
- `reason`: brief explanation

Follows the same pattern as existing `/name-type` and `/name-relationship` endpoints: structured request, zero-temperature LLM call, JSON response.

## Integration

### Sophia

- New maintenance handler: `entity_resolution` registered with the maintenance scheduler
- Post-ingestion trigger: check newly stored nodes against existing nodes via embedding + structural triage
- Periodic trigger: broader pairwise scan at configurable interval
- Hypothesis edge CRUD: create, update confidence, check for merge readiness
- Merge executor: absorb edges, update embeddings, clean up consumed node

### Hermes

- New `/alias-check` endpoint following existing naming endpoint patterns
- Service function using `generate_completion()` with zero temperature
- Structured request/response via Pydantic models

### Maintenance Scheduler

- Register `entity_resolution` handler in scheduler's handler dict
- Subscribe to `logos:sophia:proposal_processed` events for post-ingestion trigger
- Periodic scan on configurable interval (default: part of the scheduler's periodic cycle)

## Detection Signals Explicitly Excluded

- **String similarity** (Jaro-Winkler, Soundex, etc.) — brittle, fails multilingual, and Sophia is non-linguistic. Embedding proximity already captures semantic equivalence regardless of surface form.

## What This Enables

Entity resolution is the first application of geometric reasoning over the embedding space. The same detection machinery — finding recurring structures, comparing neighborhoods, analyzing topological similarity — will power relationship inference (#506), ontology evolution (#505), and type correction (#504). Build it once, apply it everywhere.

As the universal encoder improves, the geometric signals become richer. The detection and hypothesis machinery stays the same; the quality of what it finds improves automatically. Discoveries from pattern detection feed back into training better embeddings, closing the loop.
