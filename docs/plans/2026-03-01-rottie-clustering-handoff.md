# Experiment: Rottie/Rottweiler Entity Clustering

## Goal

Test whether Sophia's ingestion pipeline can detect that "rottie" and "rottweiler" refer to the same entity, using only the existing NER → embedding → dedup pipeline (no alias types, no special handling).

## Background / Design Discussion

We explored several approaches to knowledge graph duplicate/alias detection:

1. **Embedding similarity alone** — tested bare word embeddings: "rottie" vs "rottweiler" L2=1.02, cosine=0.48. Almost identical to "rottweiler" vs "labrador" (L2=1.03). **Unusable for alias detection.**
2. **Contextual sentence embeddings** — embedded sentences *about* rotties and rottweilers. Centroid L2=0.56, avg cross-group cosine=0.45 (vs 0.24 for rottweiler/labrador). **They do cluster together** in context, but the current hard threshold (L2 < 0.5) would still miss it.
3. **Alias node type** — discussed but deferred. User preference: Sophia should *intuit* aliases through reasoning, not depend on preloaded alias data.
4. **ALIAS_OF relationships** — discussed as the representation for detected aliases. Directional (alias → canonical). Alias nodes would be leaf-only (no non-alias edges).
5. **String-level signals** — prefix overlap, Jaro-Winkler, phonetic encoding (Soundex/Metaphone) could supplement embedding similarity.
6. **KG maintenance/pruning** — the user wants alias detection to be part of a broader graph maintenance capability where Sophia reasons about her own graph (structural similarity, complementary distribution, LLM confirmation). This is linked to cognitive loop priorities.

## Current State

### Infrastructure
- **Neo4j**: Cleared and re-seeded with 26 type definitions only (`--ontology-only`)
- **Milvus**: All collections dropped and recreated (Entity, Concept, State, Process, Edge, TypeCentroid). **Type centroids are NOT seeded** — they develop incrementally as data flows through.
- **Hermes**: Running on port 17000, healthy
- **Sophia**: Running on port 47000, healthy

### What Was Done
- Cleared Neo4j (`MATCH (n) DETACH DELETE n`)
- Dropped all Milvus collections
- Re-seeded type definitions: `cd logos && poetry run python -m logos_hcg.seeder --uri bolt://localhost:7687 --user neo4j --password logosdev --ontology-only`
- Recreated Milvus collections programmatically via `HCGMilvusSync.ensure_collection()`
- Sent one rottweiler text to `/llm` endpoint — **haven't verified if it ingested yet**

### What Needs to Happen

1. **Verify the first rottweiler text was ingested** — check Neo4j for any new nodes beyond the 26 type definitions
2. **Send rottweiler-focused text** through Hermes `/llm`:
   ```
   "The Rottweiler is a large, powerful breed of domestic dog originally bred in Germany. Rottweilers were used to herd livestock and pull carts for butchers. They are known for their loyalty, confidence, and protective nature. A well-trained Rottweiler is calm and courageous. The breed has a short, dense black coat with distinctive tan markings on the face, chest, and legs. Rottweilers are prone to hip dysplasia and osteosarcoma. They typically weigh between 80 and 135 pounds."
   ```
3. **Send rottie-focused text** through Hermes `/llm`:
   ```
   "My rottie is the most loyal dog I have ever owned. Rotties are often misunderstood but they are gentle giants. I took my rottie to the vet for his annual checkup. Rotties need lots of exercise and mental stimulation. Our rottie loves playing fetch in the backyard."
   ```
4. **Query Neo4j** to see what nodes were created — did "rottie" and "rottweiler" end up as the same node or separate nodes?
5. **Check Milvus** embeddings to see how close the entities landed in vector space

### API Format

Hermes `/llm` endpoint triggers the full cognitive loop (NER → proposal → Sophia ingestion):

```bash
curl -s -X POST http://localhost:17000/llm \
  -H "Content-Type: application/json" \
  -d '{"prompt": "your text here", "provider": "openai"}'
```

### Neo4j Query to Check Results

```cypher
MATCH (n:Node) WHERE NOT n.is_type_definition
RETURN n.name, n.type, n.uuid
ORDER BY n.name
```

## Key Files

| File | Purpose |
|------|---------|
| `hermes/src/hermes/main.py:769` | `/llm` endpoint — triggers full pipeline |
| `hermes/src/hermes/main.py:170` | `_get_sophia_context()` — builds proposal, sends to Sophia |
| `hermes/src/hermes/proposal_builder.py` | NER + relation extraction + embeddings → proposal |
| `hermes/src/hermes/combined_extractor.py` | Single-call LLM NER + relation extraction |
| `sophia/src/sophia/ingestion/proposal_processor.py` | Processes proposals — dedup, type classification, graph writes |
| `sophia/src/sophia/hcg_client/client.py:75` | `add_node()` — name+type dedup check |

## Expected Outcome

"Rottie" and "rottweiler" will create **separate nodes** — there's no mechanism to merge them during ingestion. The real question is: **after ingesting lots of information about both, does the graph structure make it obvious they need to be merged?** Shared relationships (e.g., both linked to "dog", "Germany", "hip dysplasia"), overlapping neighbors, similar property patterns — these are the signals a future KG maintenance capability could use to detect aliases through relational reasoning rather than string/embedding matching alone.

## Results (2026-03-01)

### Texts Ingested

1. **Rottweiler facts** — breed origin, physical traits, health issues, weight range
2. **Rottie personal anecdote** — loyalty, vet visits, exercise, fetch
3. **Rottweiler care** — vet checkups, exercise, mental stimulation, training, socialization

### Entities Created

Three separate nodes: `Rottweiler`, `rottie`, `Rotties` (NER didn't even consolidate rottie/Rotties).

### Relationship Overlap After 3 Texts

| Relationship | Rottweiler | rottie | Rotties |
|---|---|---|---|
| NEEDS → exercise | yes | yes | |
| NEEDS → mental stimulation | yes | yes | |
| REQUIRES → veterinary checkups | yes | | |
| REQUIRES → annual checkup | | yes | |
| NEEDS → vet | | | yes |
| IS_A → object | yes | yes | yes |
| IS_A → dog | | yes | |

Two non-trivial shared edges (`NEEDS → exercise`, `NEEDS → mental stimulation`) with identical relation types and targets. Plus near-matches on vet/checkup edges.

### Conclusion

With just 3 short texts, the graph already accumulates enough shared relationship structure that a maintenance pass could plausibly detect these as merge candidates. The signals available:

1. **Same type** (`object`) — trivial but necessary filter
2. **String similarity** — "rottie"/"Rotties"/"Rottweiler" share a prefix; Jaro-Winkler or prefix overlap would flag these
3. **Overlapping relationship signatures** — identical `NEEDS` edges to the same targets, similar `REQUIRES` edges to semantically related targets
4. **Complementary distribution** — rottie has edges that Rottweiler doesn't (and vice versa), but no contradictions

With more data, these overlaps would only increase. A KG maintenance capability that combines string-level signals with structural similarity (shared neighbors via same relation types) has enough to work with. LLM confirmation ("are rottie and rottweiler the same entity?") could serve as the final decision step.

### Finding 2: Type and Relationship Correctness

The NER/type classification pipeline produces significant misclassifications. Everything that isn't obviously a location gets dumped into `object`:

| Entity | Assigned Type | Should Be |
|---|---|---|
| `osteosarcoma` | location | concept (disease) |
| `butchers` | location | object or concept (profession) |
| `fetch` | object | activity/process |
| `daily walks` | object | activity/process |
| `obedience training` | object | activity/process |
| `exercise` | object | activity/process |
| `exercise routine` | object | activity/process |
| `socialization` | object | activity/process |
| `training` | object | activity/process |
| `mental stimulation` | object | activity/process |
| `annual checkup` | object | activity/process |
| `veterinary checkups` | object | activity/process |
| `health issues` | object | concept |
| `hip dysplasia` | object | concept (disease) |
| `backyard` | object | location |

The relationship context itself carries signals that could inform type reclassification:
- Something targeted by `PRONE_TO` is probably a disease/condition, not a location
- Something targeted by `NEEDS` that involves physical movement is probably an activity, not an object
- Something targeted by `REQUIRES` that describes a scheduled event is probably a process

This means KG maintenance has (at least) two jobs:
1. **Entity resolution** — merge aliases (rottie/Rotties/Rottweiler; annual checkup/veterinary checkups)
2. **Type correction** — re-evaluate node types using relationship context and LLM reasoning

Both can leverage the same graph-structural signals and LLM confirmation step.

### Finding 3: Missing Relationships the Graph Implies

Several node pairs are clearly related but have no explicit edge between them:

**Missing links from structural context:**
- `Rottweiler` → `dog` — rottie IS_A dog, but Rottweiler has no relationship to `dog`
- `Rottweiler` → `breed` — `breed` exists as a node, nothing connects it
- `exercise routine` → `Rottweiler` — exercise routine INCLUDES fetch/walks/training, Rottweiler NEEDS exercise, but no path connects them
- `hip dysplasia` / `osteosarcoma` → `health issues` — Rottweiler MONITORED_FOR health issues and PRONE_TO both diseases, but no edge says those diseases *are* health issues
- `vet` → `annual checkup` / `veterinary checkups` — you go to a vet for checkups, but no edge says so

These are cases where Sophia could detect disconnected subgraphs that should be linked, purely from structural analysis, and ask Hermes to fill in the missing relationship.

## Design: Sophia's Graph Maintenance Agency

### The Problem

Currently Sophia is a graph *writer*, not a graph *thinker*. All semantic decisions (entity extraction, typing, relationship creation) happen in Hermes' single LLM call. Sophia just dedup-checks and stores. She has no mechanism to create relationships, correct types, or merge entities based on what she observes in her own graph.

### The Insight

Hermes sees one text at a time. Sophia sees the accumulated graph across all texts. New structural knowledge emerges from the accumulation that no single text contains. Sophia's agency comes from detecting patterns in that accumulated structure.

### Architecture: Sophia Asks, Hermes Answers

Sophia does NOT compose natural language questions. She sends structured requests to Hermes endpoints with nodes/edges as JSON. Hermes uses LLM reasoning on that structured data and returns proposed changes. Sophia applies or rejects the changes.

Example flow:
1. Sophia detects two nodes with overlapping relationship signatures
2. Sophia sends both nodes (with their full edge sets) to Hermes via a structured endpoint
3. Hermes returns a structured response: merge recommendation, new relationship, type correction, etc.
4. Sophia applies the change to the graph

### Asking Threshold

The cost of asking Hermes is low (one LLM call). The cost of not asking is a fragmented graph. So Sophia can be aggressive about asking. A threshold as simple as "two nodes of the same type sharing N≥2 non-trivial relationship targets" is enough to trigger a query. Weaker signals (one shared relationship + string similarity, same document embeddings but no direct edge) could also trigger.

### Carrying Uncertainty

Sophia doesn't have to merge immediately or resolve every question. She can record suspected relationships as candidate edges (e.g., `rottie → POSSIBLE_ALIAS_OF → Rottweiler`) with a confidence score. As more data arrives:

- Overlapping relationships grow → confidence increases
- Diverging relationships appear → confidence decreases
- At a threshold → ask Hermes to confirm → merge or dismiss

This makes the system self-correcting. Premature merges don't happen because confidence must accumulate. If early data suggests two things are aliases but later data shows they're distinct (e.g., two people with similar names but different relationships), the diverging signatures lower confidence automatically.

### Alias Direction

When merging, Sophia needs to know which name is canonical. This is a language judgment — "Rottweiler" is the formal name, "rottie" is informal. Sophia can provide structural hints (the Rottweiler node has 14 edges, rottie has 6 — richer node is likely canonical) but defers to Hermes for the final call:

```json
// Sophia sends:
{ "nodes": [
    {"name": "Rottweiler", "edge_count": 14, "relationships": [...]},
    {"name": "rottie", "edge_count": 6, "relationships": [...]}
  ],
  "question_type": "alias_check"
}

// Hermes returns:
{ "canonical": "Rottweiler",
  "aliases": ["rottie", "Rotties"],
  "confidence": 0.95 }
```

### Four Jobs for KG Maintenance

1. **Entity resolution** — detect and merge aliases (rottie/Rotties/Rottweiler; annual checkup/veterinary checkups)
2. **Type correction** — re-evaluate node types using relationship context (fetch is an activity, not an object)
3. **Relationship inference** — detect structural gaps and ask Hermes to fill missing edges (Rottweiler → dog; hip dysplasia → health issues)
4. **Ontology evolution** — detect when new types are needed by measuring type cohesion. If nodes within a type have low relationship-signature similarity (e.g., `object` contains `hip dysplasia`, `backyard`, `fetch`, and `Rottweiler` which have almost nothing in common), the type is a junk drawer. Sophia clusters nodes within low-cohesion types by their relationship signatures, sends the clusters to Hermes, and Hermes proposes new type definitions. The clusters *are* the missing types — high intra-cluster similarity, low inter-cluster similarity. Conversely, a type with high cohesion (e.g., all `location` nodes are targets of `ORIGINATED_IN`, `LOCATED_AT`) is working fine and needs no splitting. Two independent signals reinforce each other here: **structural clusters** (shared relationship patterns in Neo4j) and **embedding clusters** (semantic proximity in Milvus). When both agree — structurally similar AND semantically close — that's a very strong candidate. Sophia already has both data sources; she just needs the logic to combine them. The heuristic: large clusters with high internal variance indicate a type that isn't capturing anything meaningful. As data accumulates, internal sub-clusters become more distinct and actionable. Once Sophia identifies a candidate cluster, she sends the nodes to Hermes — not to validate the cluster, but to **name** it. Sophia knows *that* they belong together; Hermes knows *what* they are.

All four use the same pattern: Sophia detects structural signal → sends structured request to Hermes → applies the result.

### Hermes Improvements (Pre-Ingestion Quality)

Two changes to Hermes reduce the maintenance burden on Sophia:

1. **Name normalization** — As part of the NER step (no extra LLM call), Hermes normalizes entity names: lowercase canonical forms, fix typos, singularize plurals. `Rotties` → `rottie`, `Rottweiller` → `Rottweiler`. Sophia never sees the noise.

2. **Type-aware NER** — Hermes uses Sophia's current type and edge type lists in the NER prompt: "Classify using these types: {types}. If none fit, use 'object'." This dramatically improves first-pass type accuracy and keeps `object` as an intentional junk drawer for unknowns. Sophia publishes ontology changes (new types, edge types, merges) via pub/sub; Hermes subscribes and updates its cached copy immediately. No polling, no stale caches.

3. **NER on prompt + reply** — Currently Hermes runs NER on just the user's text. Instead, it should run NER on both the prompt and the LLM reply together. The reply restates things more formally, uses canonical names, makes implicit connections explicit — yielding a richer proposal. Sophia never sees raw text; she just gets a better structured proposal.

### Self-Improving Ontology Loop

These pieces form a closed loop:

1. Hermes classifies with known types → unknowns fall to `object`
2. Sophia ingests → `object` grows
3. Sophia detects high-variance clusters in `object`
4. Sophia sends clusters to Hermes → Hermes names the new type
5. Sophia creates the type, retypes the nodes
6. Hermes fetches the updated type list on next refresh
7. Next time similar entities arrive, they get classified correctly

The ontology emerges from data and refines itself. No one designs it up front. If a type gets too broad later, the same variance detection splits it further. The ontology keeps evolving as long as data keeps flowing.

### Role of Type Centroids

Milvus already maintains a TypeCentroid collection. Centroids are a third signal alongside graph structure and string similarity, and they're cheap to compute incrementally:

- **Type cohesion** — measure average distance from members to their type centroid. High spread = incoherent type (`object`). Tight spread = healthy type (`location`). No pairwise clustering needed.
- **Type assignment validation** — a node far from its assigned type centroid but close to another type's centroid is probably mistyped. Detectable without any graph analysis.
- **New type detection** — outliers far from the `object` centroid cluster into sub-groups. Each sub-cluster's natural centroid becomes the centroid for the new type.
- **Merge candidates** — two nodes very close in embedding space but far from their type centroid may be aliases that are both mistyped.
- **Scale** — don't cluster all 50,000 `object` nodes pairwise. Measure distance from centroid, pull out the outliers, cluster only those. Well-behaved members near the centroid can be ignored.

### Merge Mechanics

When a merge happens, the canonical (surviving) node absorbs all edges from the consumed node. No information lost. The surviving node gets a history entry recording what was merged:

```json
{
  "merge_history": [
    {"merged": "rottie", "date": "2026-03-01T...", "confidence": 0.92},
    {"merged": "Rotties", "date": "2026-03-01T...", "confidence": 0.88}
  ]
}
```

This serves as provenance (why does this node have these edges?), aids debugging, and gives Sophia useful metadata — nodes with long merge histories are high-confidence canonical entities.

### Competing Edges and Contradictions

Rather than treating contradictions as errors to resolve, competing claims share the same representation: multiple edges from the same source with the same relation type, confidences summing to 1, context determines which Sophia uses.

- `Rottweilers → ORIGINATED_IN → Germany` 0.95 / `France` 0.05 — effectively resolved
- `Rottweilers → described_as → dangerous` 0.4 / `gentle` 0.6 — both valid, context picks
- `integrals → SOLVED_BY → trig substitution` 0.33 / `factoring` 0.33 / `parts` 0.33 — all valid alternatives

New evidence adjusts confidences proportionally. Low-confidence edges never get promoted to long-term memory, so most "wrong" claims resolve themselves naturally.

**Open question**: Should competing edges always be zero-sum (competitive) or can they be additive (independent alternatives)? `ORIGINATED_IN` implies one answer; `SOLVED_BY` implies many. A new `SOLVED_BY` edge shouldn't lower confidence in existing methods. But initially, before enough alternatives accumulate, a new edge looks like a contradiction. The distinction between competitive and additive relations — and when/how to transition between them — is unresolved.

### Taxonomic Scaffolding (IS_A, HAS_A, BELONGS_TO)

The basic ontological relationships (IS_A, HAS_A, BELONGS_TO) are almost always implicit in text. Nobody writes "hip dysplasia is a health condition" — they write "Rottweilers are prone to hip dysplasia." NER extracts what's explicitly stated, so the graph will always be sparse on these structural relationships.

This is likely the biggest single job for maintenance — not fixing errors, but filling in the scaffolding that text never states.

**Rule: type changes must create IS_A edges.** When maintenance retypes `hip dysplasia` from `object` to `condition`, it must also create `hip dysplasia IS_A condition`. The type assignment and the IS_A edge are the same fact — they must stay in sync. This is bookkeeping, not inference.

Similarly, merges propagate IS_A edges. If `rottie IS_A dog` and rottie merges into Rottweiler, the surviving node gets `Rottweiler IS_A dog`.

Beyond mechanical bookkeeping, Sophia can ask Hermes to fill in deeper taxonomic chains: `dog IS_A animal`, `Germany IS_A country IS_A location`. These build the ontological backbone that makes the graph navigable.

### Edge Normalization

Edge embeddings aren't precise enough to detect synonymous relation types — the gap between similar pairs (NEEDS/REQUIRES cosine 0.72) and unrelated pairs (~0.55) is too narrow to threshold reliably. For normalizing relation types, just send Hermes the list periodically and let the LLM sort them. Same as node types: Hermes fetching the existing edge type list and using it during NER would prevent most synonyms at the source.

### Detecting Missing Edges

Sophia detects candidate missing edges through:
- **Shared neighbors** — two nodes connected to many of the same nodes but not to each other
- **Embedding proximity** — close in Milvus but no path in Neo4j
- **Co-occurrence** — repeatedly appear in the same proposals/documents but never as a proposed edge
- **Pattern completion** — "every node in this cluster has edge X except this one"

Sophia sends the candidate node pair with context to Hermes, Hermes proposes the relationship type and direction.

### Prerequisite: Centralized Redis with Pub/Sub (logos #469)

Redis pub/sub is a prerequisite for KG maintenance work. A Redis container definition already exists in `sophia/containers/docker-compose.test.yml` (`redis:7-alpine`) but needs to move to `logos/infra/docker-compose.hcg.dev.yml` as shared infrastructure alongside Neo4j and Milvus. Both Hermes and Sophia already have `redis-py` client code (context cache, proposal queue); pub/sub is a built-in capability of the same client.

**Areas enhanced by pub/sub:**

- **Ontology distribution** — Sophia publishes type/edge type changes; Hermes subscribes and updates its NER prompt immediately. No polling, no stale caches, no extra per-request latency. This is the primary use case — the self-improving ontology loop depends on Hermes always having the current type list.
- **Maintenance event triggers** — After ingestion, Sophia can publish "nodes changed in neighborhood X" events. A maintenance worker (same process or separate) subscribes and runs targeted checks on the affected subgraph rather than scanning the whole graph periodically.
- **Merge notifications** — When Sophia merges entities, she publishes the merge event. Hermes can update its name normalization cache. Other consumers (Apollo, future services) can react to graph changes without polling.
- **Confidence updates** — As competing edges accumulate evidence, Sophia can publish confidence threshold crossings ("edge X just crossed 0.8 confidence") to trigger promotion to long-term memory or to notify downstream consumers.
- **Latency reduction** — The current synchronous Sophia call in the `/llm` path adds latency. With pub/sub, proposals can be fire-and-forget (published to a channel), processed asynchronously by Sophia, and context returned via cached results on the next turn. The existing `ContextCache` already does part of this with Redis lists; pub/sub generalizes it.
- **Edge type normalization** — Sophia publishes the canonical edge type list alongside node types. Hermes uses it during NER to produce consistent relation labels, avoiding the NEEDS/REQUIRES/ESSENTIAL_FOR synonymy problem at the source.
- **Sophia → Hermes structured queries** — Maintenance requests (alias checks, cluster naming, relationship proposals) could be published as messages on a channel rather than synchronous HTTP calls, allowing Hermes to batch and prioritize them.

### Open Questions

- **When does maintenance run?** Post-ingestion, periodic, event-driven, or some combination? Pub/sub enables event-driven triggers.
- **Type hierarchy** — Emergent types go under `root` as peers of `object` and `location` (e.g., `process`, `condition`, `activity`). They are NOT `reserved_*` — those are for Sophia's internal machinery (plans, goals, executions, agents). If Sophia later detects that related types should be grouped, that's another round of ontology evolution.
- **Competitive vs additive edges** — How does Sophia determine whether competing edges are mutually exclusive or independent alternatives?
- **Confidence mechanics** — How exactly do confidence scores update? What raises/lowers them? What's the promotion threshold to long-term memory?
- **Hermes endpoint contracts** — The structured request/response formats for maintenance queries (alias_check, name_cluster, type_proposal, relationship inference) need to be defined.

## Broader Context

This experiment feeds into a larger discussion about KG maintenance/pruning capabilities for Sophia. Entity resolution, type correction, and relationship inference should all be cognitive capabilities — Sophia reasoning about her own graph with LLM assistance from Hermes — not mechanical lookups or preloaded data. The accumulated graph structure provides sufficient signal even with small amounts of data, and the system is self-correcting through confidence-based deferred merging.

Redis pub/sub (logos #469) is a prerequisite — it enables the communication patterns that make Sophia's maintenance agency and the self-improving ontology loop practical.
