# Cognitive Loop Design

## Goal

Close the cognitive loop — Apollo → Hermes → Sophia → Hermes → Apollo — so that each conversational turn produces structured knowledge in the graph and retrieves relevant context for the LLM.

## Architecture

**Apollo** is the interface layer — the API between external clients and the system's internal services. He receives user input from whatever client modality is available, routes it to Hermes for processing, and returns the response. Apollo is where the loop starts and ends from the user's perspective.

**Hermes** is stateless and non-persistent. He is the only component that understands language. On each turn, he receives the user's message from Apollo, extracts entities and generates embeddings, builds a structured proposal, and sends it to Sophia.

**Sophia** operates in embedding space — she does not read text. She ingests what's worth persisting into the HCG (nodes and reified edges, each with embeddings), searches the graph by embedding similarity for relevant context, and returns it.

Hermes then stitches the returned graph context (which includes text properties on nodes from both the current and previous turns) into a coherent LLM prompt, generates a response, and returns it through Apollo to the client. This is how Hermes gets his own data back despite being stateless — Sophia returns it as part of the graph context.

## Key Principles

- **Hermes does all language work.** Text → structure (proposals going to Sophia). Structure → text (graph context coming back for the LLM prompt).
- **Sophia operates on embeddings.** She reasons about similarity, structure, and relationships — never raw text. Text lives on nodes as properties for Hermes's benefit.
- **Proposals are proposals, not commands.** Hermes says "here's how to add this." Sophia has cognitive authority over the graph.
- **Sophia is the system's memory.** Hermes is stateless. Sophia returns semantically relevant context from the graph, scoped by embedding similarity to the current turn — not a full conversation dump.
- **Hermes assembles the prompt.** The quality of the LLM response depends on how well Hermes stitches graph context into the prompt.
- **Structure encodes semantics.** The graph's shape is informative. Directionality, type hierarchy, and relationships are all expressed through structure — not flags or denormalized properties (with the sole exception of `bidirectional` on edges, a pragmatic concession).

## Ontology Model

### Graph Structure

Neo4j native relationships are used **only** as structural plumbing — `:FROM` and `:TO` connectors that wire edge nodes to their endpoints. All knowledge lives on nodes. Sophia never reasons about Neo4j relationships; she deals only with nodes, each with properties, embeddings, and UUIDs.

### Existing Code

The following already exists and must be accounted for:

| Code | Location | What It Does |
|------|----------|--------------|
| `Node`, `Edge` models | `sophia/src/sophia/knowledge_graph/` | Pydantic models for graph objects |
| `KnowledgeGraph` | `sophia/src/sophia/knowledge_graph/graph.py` | In-memory NetworkX graph (not Neo4j) |
| `Entity`, `Concept`, `State`, `Process`, etc. | `logos/logos_hcg/models.py` (917 lines) | Rich typed Pydantic models with embedding metadata |
| `HCGClient` (read-only on main) | `logos/logos_hcg/client.py` | Neo4j queries, traversal, connection pooling |
| `HCGClient` (write ops on feature branch) | `logos/logos_hcg/client.py` | `add_node()`, `add_typed_edge()`, `add_relation()`, `clear_all()` |
| `HCGClient` (sophia extension) | `sophia/src/sophia/hcg_client/client.py` | Extends foundry: SHACL validation, auto-ancestors, provenance, property encoding |
| `HCGQueries` | `logos/logos_hcg/queries.py` (1020 lines) | Cypher queries — all assume native Neo4j relationships |
| `HCGMilvusSync` | `logos/logos_hcg/sync.py` | Milvus upsert/search/verify per node type collection |
| `HCGPlanner` | `logos/logos_hcg/planner.py` | Backward chaining over REQUIRES/CAUSES edges |
| `HCGSeeder` (logos) | `logos/logos_hcg/seeder.py` | Type definitions, demo scenario, persona diary |
| `HCGSeeder` (sophia) | `sophia/src/sophia/hcg_client/seeder.py` | Separate seeder for sophia's pick-and-place demo |
| `SHACLValidator` | `sophia/src/sophia/hcg_client/shacl_validator.py` | Shapes for Node, Edge, HermesProposal, ProposedPlanStep |
| `HermesProposalRequest/Response` | `sophia/src/sophia/api/models.py` | Proposal contract (endpoint stubbed at app.py:1321) |
| `FeedbackDispatcher/Queue/Worker` | `sophia/src/sophia/feedback/` | Redis-backed feedback system (Redis not deployed) |
| `_forward_llm_to_sophia()` | `hermes/src/hermes/main.py:589` | Sends raw LLM text to Sophia (needs replacement) |
| `process_nlp()`, `generate_embedding()` | `hermes/src/hermes/services.py` | NER + embedding generation (ready to use) |
| `HermesMilvusClient` | `hermes/src/hermes/milvus_client.py` | Insert-only embedding persistence (no search) |

### Nodes

- Every node carries properties and embeddings.
- Schemas are defined in code as Pydantic models. The logos foundry (`logos_hcg/models.py`) has typed models for Entity, Concept, State, Process, Fact, Association, etc. — each with `embedding_id`, `embedding_model`, `last_sync` fields. Sophia's `knowledge_graph/node.py` has a general `Node` model with uuid, name, type, properties.
- Type hierarchy is expressed through reified IS_A edge nodes, not stored as a denormalized `ancestors` property. Ancestry is always a graph traversal.
- Type definition nodes exist in the graph with their own embeddings. Instance nodes connect to their type via IS_A edge nodes.
- Sophia can create new types as she discovers them. The ontology is emergent, not prescribed.
- The seeder bootstraps a starting ontology — temporary scaffolding until Sophia maintains it herself.

### Edges

Edges are reified — they are **nodes** in the graph connected to their source and target via structural `:FROM`/`:TO` native Neo4j relationships. These are the only native relationships in the graph.

```
(source:Node)<-[:FROM]-(edge:Node {type: "LOCATED_AT", ...})-[:TO]->(target:Node)
```

The `Edge` model:

```python
class Edge(BaseModel):
    id: str                         # UUID
    source: str                     # source node UUID
    target: str                     # target node UUID
    relation: str                   # edge type (references edge type definition)
    bidirectional: bool = False     # True if relationship goes both ways
    properties: dict[str, Any]      # provenance, confidence, temporal info, etc.
```

Edge nodes:
- Have their own UUIDs, properties, and embeddings in Milvus.
- Can be the source or target of other edge nodes (e.g., an INFERRED_FROM edge connecting back to an observation node).
- Connect to their type definition node via an IS_A edge node.
- Carry provenance (who created, when, from what evidence), confidence scores, and temporal information.
- Retrieval methods abstract the reified structure — consumers get clean `Edge` objects, not raw graph internals.

**Impact on existing code:** `queries.py` (1020 lines of Cypher), `planner.py` (backward chaining), and both HCGClient implementations all assume native Neo4j relationships. These must be rewritten to work with edge nodes + `:FROM`/`:TO` structural connectors. This is the largest prerequisite for the cognitive loop.

### Embeddings

Sophia's primary reasoning mechanism. She finds relevant nodes and edges via Milvus embedding similarity search. `HCGMilvusSync` in `logos_hcg/sync.py` already provides `upsert_embedding()`, `batch_upsert_embeddings()`, `get_embedding()`, and `verify_sync()` with separate collections per node type.

**Type definition embeddings** are generated from the full JSON definition of the type — name, description, property schema, constraints. This produces semantically rich embeddings where related types cluster together (e.g., spatial relationship types like LOCATED_IN, CONTAINED_BY, OCCUPIES group naturally). Both entity type definitions and edge type definitions are nodes with embeddings. Edge type definitions are already seeded.

**Instance embeddings** are generated from the full JSON definition of the specific instance, which includes what it connects to. An edge instance's definition includes its source and target node information, so the embedding captures not just "this is a spatial relationship" but "this is a specific spatial relationship between a city and a country." This means Sophia can search at two levels:
- **Type-level**: find edges whose type definition embedding is similar (all spatial relationships)
- **Instance-level**: find edges whose specific embedding is similar (other city-country relationships)

Both type definitions and instances are nodes in the graph with embeddings in Milvus — the same search mechanism serves both levels.

## Per-Turn Flow

```
User → Apollo → Hermes                          Sophia
                  │                                │
                  ├─ 1. Extract entities (NER)     │
                  ├─ 2. Generate embeddings        │
                  ├─ 3. Build proposal             │
                  │     (entities + embeddings      │
                  │      + text metadata)           │
                  │                                │
                  ├────── proposal ────────────────→│
                  │                                ├─ 4. Search graph by embedding similarity
                  │                                ├─ 5. Decide what to ingest
                  │                                ├─ 6. Create/update nodes + reified edges
                  │                                ├─ 7. Store embeddings in Milvus
                  │                                │
                  │←───── graph context ───────────┤
                  │       (nodes + edges + scores   │
                  │        with properties/text)    │
                  │                                │
                  ├─ 8. Translate context to text  │
                  ├─ 9. Stitch into LLM prompt     │
                  ├─ 10. Generate LLM response     │
                  │                                │
User ← Apollo ← Hermes                          Sophia
```

**Existing infrastructure used in this flow:**
- Steps 1-2: `process_nlp()` and `generate_embedding()` in `hermes/services.py` (ready to use)
- Step 3: `_forward_llm_to_sophia()` in `hermes/main.py:589` (needs replacement with ProposalBuilder)
- Step 4: `HCGMilvusSync.search_similar()` or equivalent (exists in `logos_hcg/sync.py`)
- Steps 5-7: `ingest_hermes_proposal()` in `sophia/api/app.py:1285` (stubbed — TODO at line 1321)
- Step 7: `HCGMilvusSync.upsert_embedding()` (exists in `logos_hcg/sync.py`)

## What's Out of Scope

| Item | Why |
|------|-----|
| Ephemeral memory / CWM | Separate system. Conversation trivia lives outside the graph. Has its own lifecycle — items are either promoted to the graph or discarded. Not needed for the loop. |
| Session boundaries (sophia#101) | Not needed — Hermes sends current turn, Sophia returns semantically relevant context. No session tracking required. |
| Executor (sophia#20) | Separate from the chat loop. |
| Planner stub deprecation (logos#403) | Mechanical cleanup, separate concern. |

## Infrastructure

- **Redis**: Goes in `logos/infra/docker-compose.hcg.dev.yml` (shared dev compose) and per-repo test compose files in `logos/infra/sophia/` and `logos/infra/hermes/`. Needed for Sophia's feedback system.
- **Milvus**: Already deployed. Sophia's primary search mechanism for the loop.
- **Neo4j**: Already deployed. The graph.

## Prerequisite: Edge Reification and Graph Model

This is the largest prerequisite. The entire existing graph layer assumes native Neo4j relationships. Moving to edge nodes with `:FROM`/`:TO` structural connectors requires changes across both repos.

### logos foundry (`logos_hcg/`)

On the `feat/hcg-write-operations-and-seeder` branch:

- **Update `Edge` model** — `source`/`target` as single strings, add `bidirectional` flag, add provenance/confidence/timestamp fields.
- **Drop `ancestors` from `add_node()`** — type hierarchy is graph traversal via IS_A edge nodes, not a stored property.
- **Drop `is_type_definition` from `add_node()`** — type definitions are nodes like any other; their role is expressed through graph structure (they are targets of IS_A edge nodes).
- **Replace `add_typed_edge()` and `add_relation()`** — new method creates an edge node and wires it with `:FROM`/`:TO` structural relationships. Both existing methods create native Neo4j relationships which won't work.
- **Rewrite `queries.py`** — all 1020 lines assume native relationships. Must be updated to traverse edge nodes via `:FROM`/`:TO`. This includes relationship queries, causal traversal, planning queries, link creation queries.
- **Update `planner.py`** — backward chaining over REQUIRES/CAUSES must traverse edge nodes instead of native relationships.
- **Update seeder** — create IS_A edge nodes for type hierarchy instead of storing `ancestors` lists. Ensure all edge type definitions have embeddings.
- **Fix `datetime.utcnow()`** — use `datetime.now(UTC)` in new code.
- Merge to main.

### sophia (`sophia/src/sophia/`)

- **Update `Edge` model in `knowledge_graph/edge.py`** — align with foundry's updated Edge model.
- **Update `KnowledgeGraph` in `knowledge_graph/graph.py`** — in-memory graph must handle edge nodes (edges as nodes connected to source/target).
- **Update sophia's `HCGClient`** — `add_edge()`, `get_edge()`, `query_edges_from()`, `list_all_edges()` must work with edge nodes. Remove `_get_type_ancestors()` (ancestors are graph traversal now).
- **Update `SHACLValidator`** — edge shapes must validate edge nodes, not native relationships.
- **Update sophia's seeder** — align with new edge model.

## Acceptance Criteria

The loop is closed when:

1. User sends a message (via Apollo or direct API)
2. Hermes builds a structured proposal (entities + embeddings) from the turn
3. Hermes sends the proposal to Sophia
4. Sophia searches the graph by embedding for relevant existing context
5. Sophia ingests new knowledge (nodes + reified edge nodes + embeddings)
6. Sophia returns relevant graph context (nodes, edges, scores, with text properties)
7. Hermes translates graph context into natural language for the LLM prompt
8. LLM generates a response informed by graph knowledge
9. **On the NEXT turn**, step 4 finds context from the knowledge created in step 5

Step 9 is the critical test — the loop is only closed when information flows back.
