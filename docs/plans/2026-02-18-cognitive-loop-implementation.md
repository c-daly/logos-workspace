# Cognitive Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the cognitive loop — Apollo → Hermes → Sophia → Hermes → Apollo — so each conversational turn produces structured knowledge in the graph and retrieves relevant context for the LLM.

**Architecture:** Hermes extracts entities and embeddings from the user's message, builds a structured proposal, and sends it to Sophia. Sophia ingests nodes and reified edge nodes into Neo4j, stores embeddings in Milvus, searches by embedding similarity for relevant context, and returns it. Hermes stitches the graph context into a coherent LLM prompt and generates a response. See `docs/plans/2026-02-18-cognitive-loop-design.md` for the full design.

**Tech Stack:** Python 3.12, FastAPI, Neo4j 5.11 (with APOC), Milvus 2.3, spaCy 3.8, sentence-transformers (all-MiniLM-L6-v2, 384-dim), httpx

---

## Standards for the Implementer

**Read before writing.** Before modifying any file, read it. Before creating anything new, search for whether it already exists. The codebase has 917 lines of typed Pydantic models (`logos/logos_hcg/models.py`), a full Milvus sync layer (`logos/logos_hcg/sync.py`), SHACL validation (`sophia/src/sophia/hcg_client/shacl_validator.py`), and two HCGClient implementations (`logos/logos_hcg/client.py` and `sophia/src/sophia/hcg_client/client.py`). If you find yourself writing something that feels like it should already exist, stop and look.

**Don't reinvent.** If the "Existing Code" table below lists it, use it. `HCGMilvusSync` is the embedding store — extend it, don't replace it. `process_nlp()` and `generate_embedding()` in `hermes/src/hermes/services.py` are the NER and embedding functions. The `Edge` and `Node` models in `sophia/src/sophia/knowledge_graph/` are the schema.

**Understand before assuming.** A model called `Edge` with `source` and `target` fields is an abstraction — not a description of how it's stored in Neo4j. Read the Cypher queries to understand storage. Read the design doc to understand intent.

**Don't claim done until it's done.** If tests fail, the task is not complete. If the implementation is partial, say so. If you hit a blocker, report it — don't mark the task as completed and move on. Evidence before assertions.

**Always use a feature branch.** Never commit to main. Create a branch from main for your work and commit there. The branch will be reviewed before merging.

**Test against real infrastructure.** Neo4j and Milvus run in Docker locally (`logos/infra/docker-compose.hcg.dev.yml`). Tests that touch the graph layer should verify actual Cypher, not mock it away.

**Small commits, working code.** Each step should leave the codebase in a passing state. Don't batch unrelated changes.

---

## Background for the Implementer

### What is the cognitive loop?

When a user sends a message, Apollo (the API gateway) routes it to Hermes (the language service). Hermes extracts entities, generates embeddings, and builds a structured proposal. He sends the proposal to Sophia (the cognitive core). Sophia ingests new knowledge into the graph (nodes + reified edge nodes + embeddings in Milvus), searches for relevant existing context by embedding similarity, and returns it. Hermes stitches the returned context into a system message for the LLM, generates a response, and returns it through Apollo.

The loop is closed when, on the NEXT turn, Sophia finds context from the PREVIOUS turn. Information flows back.

### Key architectural principles

1. **Hermes is stateless.** He has no memory between requests. Sophia is his memory.
2. **Sophia is non-linguistic.** She operates on embeddings and graph structure, never raw text. Text exists on nodes as properties for Hermes's benefit when context comes back.
3. **Proposals are proposals, not commands.** Hermes says "here's what I found." Sophia decides what enters the graph.
4. **Edges are reified.** All knowledge relationships are nodes in Neo4j, connected via `:FROM`/`:TO` structural relationships. They carry properties, embeddings, and provenance.
5. **`:FROM` and `:TO` are the ONLY native Neo4j relationships.** All knowledge lives on nodes. There are no native `[:IS_A]`, `[:CAUSES]`, etc.
6. **Type hierarchy is graph traversal.** No `ancestors` property on nodes. No `is_type_definition` flag. Types connect via IS_A edge nodes. A node's role as a type definition is expressed by graph structure (IS_A edges point to it).

### The reified edge model

Edges are **nodes** in Neo4j. The only native Neo4j relationships are structural connectors:

```
(source:Node)<-[:FROM]-(edge:Node {relation: "LOCATED_AT", ...})-[:TO]->(target:Node)
```

Both `:FROM` and `:TO` emanate FROM the edge node. The edge node carries:
- `uuid`: unique identifier
- `name`: human-readable label (e.g., `"Paris_LOCATED_IN_France"`)
- `type`: `"edge"` or a specific edge type name
- `relation`: the edge type (e.g., `"IS_A"`, `"HAS_STATE"`, `"CAUSES"`)
- `source`: source node UUID (denormalized for convenience)
- `target`: target node UUID (denormalized for convenience)
- `bidirectional`: boolean flag (default `false`)
- Additional properties: provenance, confidence, timestamps, etc.

Edge nodes have their own embeddings in Milvus. They can be the source or target of other edge nodes (e.g., an INFERRED_FROM edge pointing at an observation edge).

**Query transformation pattern.** Every native relationship query changes:

| Before (native relationship) | After (edge node traversal) |
|-----|-----|
| `(a)-[:CAUSES]->(b)` | `(a)<-[:FROM]-(e:Node {relation: "CAUSES"})-[:TO]->(b)` |
| `(a)-[:IS_A]->(b)` | `(a)<-[:FROM]-(e:Node {relation: "IS_A"})-[:TO]->(b)` |
| `(a)-[:IS_A*]->(b)` | Utility function — see Task 2 |
| `CREATE (a)-[r:CAUSES]->(b)` | Call `add_edge(source=a, target=b, relation="CAUSES")` |

### Repository layout

| Repo | Path | Purpose |
|------|------|---------|
| logos (foundry) | `logos/logos_hcg/` | Shared graph client, models, queries, Milvus sync, seeder, planner |
| sophia | `sophia/src/sophia/` | Cognitive core: extended HCG client, knowledge graph models, SHACL, API |
| hermes | `hermes/src/hermes/` | Language service: NLP, embeddings, LLM proxy, proposal builder |
| apollo | `apollo/` | API gateway: routes user messages to Hermes, returns responses |

### Existing code you MUST use

| What | Where | Status |
|------|-------|--------|
| `Node` model | `sophia/src/sophia/knowledge_graph/node.py` | Needs update: drop `is_type_definition`, `ancestors` |
| `Edge` model | `sophia/src/sophia/knowledge_graph/edge.py` | Needs update: add `bidirectional` |
| `KnowledgeGraph` | `sophia/src/sophia/knowledge_graph/graph.py` | Needs update: edge-as-node pattern |
| `HCGClient` (foundry) | `logos/logos_hcg/client.py` | On `feat/hcg-write-operations-and-seeder` branch. Needs `add_edge()`, drop old edge methods |
| `HCGClient` (sophia) | `sophia/src/sophia/hcg_client/client.py` | Extends foundry. Needs edge reification, drop `_get_type_ancestors()` |
| `HCGQueries` | `logos/logos_hcg/queries.py` (1020 lines) | All assume native relationships. Full rewrite needed |
| `HCGMilvusSync` | `logos/logos_hcg/sync.py` | Works. Extend with edge collections + `search_similar()` |
| `HCGPlanner` | `logos/logos_hcg/planner.py` | Backward chaining over REQUIRES/CAUSES. Needs edge node traversal |
| `HCGSeeder` (logos) | `logos/logos_hcg/seeder.py` | Drop ANCESTORS dict, create IS_A edge nodes |
| `HCGSeeder` (sophia) | `sophia/src/sophia/hcg_client/seeder.py` | Align with new edge model |
| `SHACLValidator` | `sophia/src/sophia/hcg_client/shacl_validator.py` | Update edge shapes for edge nodes |
| `HermesProposalRequest` | `sophia/src/sophia/api/models.py:450` | Exists. Add `proposed_nodes`, `document_embedding` fields |
| `HermesProposalResponse` | `sophia/src/sophia/api/models.py:535` | Exists. Add `relevant_context` field |
| `ingest_hermes_proposal()` | `sophia/src/sophia/api/app.py:1285` | Stubbed — TODO at line 1321. Implement |
| `_forward_llm_to_sophia()` | `hermes/src/hermes/main.py:589` | Sends raw text. Replace with proposal builder |
| `process_nlp()` | `hermes/src/hermes/services.py:213` | Ready to use (spaCy NER) |
| `generate_embedding()` | `hermes/src/hermes/services.py:258` | Ready to use (all-MiniLM-L6-v2, 384-dim) |
| Typed models | `logos/logos_hcg/models.py` (917 lines) | Entity, Concept, State, Process, etc. with embedding metadata |
| `COLLECTION_NAMES` | `logos/logos_hcg/sync.py:27` | Entity, Concept, State, Process collections. Add Edge |

### Infrastructure

| Service | Port | How to start |
|---------|------|-------------|
| Neo4j 5.11 | 7474 (HTTP), 7687 (Bolt) | `docker compose -f logos/infra/docker-compose.hcg.dev.yml up -d` |
| Milvus 2.3 | 19530 (gRPC) | Same compose file |
| Sophia | 47000 | `cd sophia && poetry run uvicorn sophia.api.app:app --port 47000` |
| Hermes | 17000 | `cd hermes && poetry run uvicorn hermes.main:app --port 17000` |
| Apollo API | 27000 | `cd apollo && ./run_apollo.sh` |

**Python version:** MUST use Python 3.12 (`/usr/local/bin/python3.12`). System `python3` is 3.14 and breaks protobuf/pymilvus. Ensure Poetry venvs use 3.12: `poetry env use /usr/local/bin/python3.12`.

---

## Dependency Graph

```
Task 1 (Edge reification — logos) ──→ Task 2 (Query rewrite — logos) ──→ Task 3 (Seeder + planner — logos)
                                  ├──→ Task 4 (Sophia alignment) ──→ Task 6 (Sophia proposal processor) ──→ Task 7 (Context injection)
                                  └──→ Task 5 (Hermes proposal builder) ──────────────────────────────────→ Task 7
```

- Task 1 must complete first — everything depends on the Edge model and `add_edge()`.
- Tasks 2 and 4 can proceed in parallel after Task 1.
- Task 3 depends on Task 2 (seeder uses updated queries).
- Task 5 (Hermes) has no dependency on sophia alignment — can start after Task 1.
- Task 6 depends on Task 4 (sophia alignment).
- Task 7 depends on Tasks 5 and 6.

---

## Task 1: Edge Reification — logos foundry

**Why:** The entire graph layer assumes native Neo4j relationships. We need reified edge nodes with `:FROM`/`:TO` structural connectors. This is the largest prerequisite for the cognitive loop.

**Branch:** Create `feat/edge-reification` from `main`. The existing `feat/hcg-write-operations-and-seeder` branch has unmerged write operations — cherry-pick what's needed, don't build on a stale branch.

**Repo:** logos

**Files:**
- Create: `logos_hcg/edge.py`
- Modify: `logos_hcg/client.py` (lines 928–1089)
- Modify: `logos_hcg/sync.py` (lines 27–34, add Edge collection)
- Test: `tests/test_edge_model.py` (new)
- Test: `tests/test_edge_reification.py` (new)

### Step 1: Write failing test for Edge model

```python
# tests/test_edge_model.py
"""Tests for the reified Edge model."""
from logos_hcg.edge import Edge


def test_edge_default_fields():
    edge = Edge(source="node-1", target="node-2", relation="IS_A")
    assert edge.id  # auto-generated UUID
    assert edge.source == "node-1"
    assert edge.target == "node-2"
    assert edge.relation == "IS_A"
    assert edge.bidirectional is False
    assert edge.properties == {}


def test_edge_with_all_fields():
    edge = Edge(
        id="edge-1",
        source="node-1",
        target="node-2",
        relation="CAUSES",
        bidirectional=False,
        properties={"confidence": 0.9, "source": "hermes"},
    )
    assert edge.id == "edge-1"
    assert edge.properties["confidence"] == 0.9


def test_edge_hashable():
    e = Edge(source="a", target="b", relation="X")
    s = {e}
    assert e in s
    assert hash(e) == hash(e.id)


def test_edge_equality_by_id():
    e1 = Edge(id="e1", source="a", target="b", relation="X")
    e2 = Edge(id="e1", source="c", target="d", relation="Y")
    assert e1 == e2
```

### Step 2: Run test to verify it fails

Run: `cd logos && poetry run pytest tests/test_edge_model.py -v`
Expected: ImportError — `logos_hcg.edge` doesn't exist yet.

### Step 3: Implement Edge model

```python
# logos_hcg/edge.py
"""Reified edge model for the Hybrid Cognitive Graph.

Edges are nodes in Neo4j connected to source and target via :FROM/:TO
structural relationships. This is the ONLY edge representation in the graph.
"""
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Edge(BaseModel):
    """Reified edge in the HCG.

    Stored as a node in Neo4j:
        (source)<-[:FROM]-(edge_node)-[:TO]->(target)
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    target: str
    relation: str
    bidirectional: bool = False
    properties: dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False
        return self.id == other.id
```

### Step 4: Run test to verify it passes

Run: `cd logos && poetry run pytest tests/test_edge_model.py -v`
Expected: PASS

### Step 5: Commit

```bash
cd logos && git add logos_hcg/edge.py tests/test_edge_model.py && git commit -m "$(cat <<'EOF'
feat: add reified Edge model to logos foundry
EOF
)"
```

### Step 6: Write failing tests for add_edge() on HCGClient

These are integration tests — they need Neo4j running.

```python
# tests/test_edge_reification.py
"""Integration tests for reified edge creation. Requires Neo4j."""
import pytest
from logos_hcg.client import HCGClient


@pytest.fixture
def client():
    c = HCGClient(uri="bolt://localhost:7687", user="neo4j", password="logosdev")
    yield c
    c.clear_all()
    c.close()


class TestAddEdge:

    def test_creates_edge_node(self, client):
        src = client.add_node(name="Paris", node_type="entity")
        tgt = client.add_node(name="France", node_type="entity")
        edge_id = client.add_edge(
            source_uuid=src, target_uuid=tgt, relation="LOCATED_IN",
        )
        assert edge_id
        # Verify edge node exists
        result = client._execute_read(
            "MATCH (e:Node {uuid: $uuid}) RETURN e", {"uuid": edge_id},
        )
        assert len(result) == 1
        edge_props = dict(result[0]["e"])
        assert edge_props["relation"] == "LOCATED_IN"
        assert edge_props["source"] == src
        assert edge_props["target"] == tgt
        assert edge_props["bidirectional"] is False

    def test_creates_from_to_structural_rels(self, client):
        src = client.add_node(name="A", node_type="entity")
        tgt = client.add_node(name="B", node_type="entity")
        edge_id = client.add_edge(
            source_uuid=src, target_uuid=tgt, relation="CAUSES",
        )
        result = client._execute_read(
            """
            MATCH (source:Node)<-[:FROM]-(edge:Node {uuid: $edge_uuid})-[:TO]->(target:Node)
            RETURN source.uuid AS src, target.uuid AS tgt
            """,
            {"edge_uuid": edge_id},
        )
        assert len(result) == 1
        assert result[0]["src"] == src
        assert result[0]["tgt"] == tgt

    def test_edge_with_properties(self, client):
        src = client.add_node(name="A", node_type="entity")
        tgt = client.add_node(name="B", node_type="entity")
        edge_id = client.add_edge(
            source_uuid=src, target_uuid=tgt, relation="REQUIRES",
            properties={"confidence": 0.85},
        )
        result = client._execute_read(
            "MATCH (e:Node {uuid: $uuid}) RETURN e.confidence AS conf",
            {"uuid": edge_id},
        )
        assert result[0]["conf"] == 0.85

    def test_bidirectional_flag(self, client):
        src = client.add_node(name="A", node_type="entity")
        tgt = client.add_node(name="B", node_type="entity")
        edge_id = client.add_edge(
            source_uuid=src, target_uuid=tgt, relation="RELATED_TO",
            bidirectional=True,
        )
        result = client._execute_read(
            "MATCH (e:Node {uuid: $uuid}) RETURN e.bidirectional AS bidir",
            {"uuid": edge_id},
        )
        assert result[0]["bidir"] is True

    def test_add_edge_idempotent(self, client):
        """Same source+target+relation should not create duplicate edge nodes."""
        src = client.add_node(name="Paris", node_type="entity")
        tgt = client.add_node(name="France", node_type="entity")
        edge1 = client.add_edge(
            source_uuid=src, target_uuid=tgt, relation="LOCATED_IN",
        )
        edge2 = client.add_edge(
            source_uuid=src, target_uuid=tgt, relation="LOCATED_IN",
        )
        assert edge1 == edge2  # same edge node returned
        # Verify only one edge node exists
        result = client._execute_read(
            """MATCH (e:Node {source: $src, target: $tgt, relation: "LOCATED_IN"})
            RETURN count(e) AS count""",
            {"src": src, "tgt": tgt},
        )
        assert result[0]["count"] == 1

    def test_edge_name_is_descriptive(self, client):
        """Edge name should include source and target node names."""
        src = client.add_node(name="Paris", node_type="entity")
        tgt = client.add_node(name="France", node_type="entity")
        edge_id = client.add_edge(
            source_uuid=src, target_uuid=tgt, relation="LOCATED_IN",
        )
        result = client._execute_read(
            "MATCH (e:Node {uuid: $uuid}) RETURN e.name AS name",
            {"uuid": edge_id},
        )
        assert result[0]["name"] == "Paris_LOCATED_IN_France"

    def test_no_native_relationships_created(self, client):
        """add_edge() must NOT create native Neo4j relationships other than :FROM/:TO."""
        src = client.add_node(name="A", node_type="entity")
        tgt = client.add_node(name="B", node_type="entity")
        client.add_edge(
            source_uuid=src, target_uuid=tgt, relation="CAUSES",
        )
        # Check no native CAUSES relationship exists
        result = client._execute_read(
            "MATCH (a:Node {uuid: $src})-[r:CAUSES]->(b:Node {uuid: $tgt}) RETURN r",
            {"src": src, "tgt": tgt},
        )
        assert len(result) == 0


class TestAddNodeClean:

    def test_add_node_no_ancestors_param(self, client):
        """add_node() should not accept 'ancestors' — hierarchy is via IS_A edges."""
        import inspect
        sig = inspect.signature(client.add_node)
        assert "ancestors" not in sig.parameters

    def test_add_node_no_is_type_definition_param(self, client):
        """add_node() should not accept 'is_type_definition' — structure, not flags."""
        import inspect
        sig = inspect.signature(client.add_node)
        assert "is_type_definition" not in sig.parameters

    def test_add_node_basic(self, client):
        uuid = client.add_node(name="Paris", node_type="location")
        assert uuid
        result = client._execute_read(
            "MATCH (n:Node {uuid: $uuid}) RETURN n", {"uuid": uuid},
        )
        assert len(result) == 1
        node = dict(result[0]["n"])
        assert node["name"] == "Paris"
        assert node["type"] == "location"
        assert "ancestors" not in node
        assert "is_type_definition" not in node
```

### Step 7: Run tests to verify they fail

Run: `cd logos && poetry run pytest tests/test_edge_reification.py -v`
Expected: FAIL — `add_edge()` doesn't exist, `add_node()` still has old params.

### Step 8: Implement add_edge() and clean up add_node()

In `logos/logos_hcg/client.py`:

1. **Fix import** — replace `from datetime import datetime` with `from datetime import UTC, datetime`.

2. **Update `add_node()`** (line 932) — remove `ancestors` and `is_type_definition` params:

```python
def add_node(
    self,
    name: str,
    node_type: str,
    uuid: str | None = None,
    properties: dict[str, Any] | None = None,
) -> str:
    """Create or merge a node in the graph.

    Type hierarchy is expressed through IS_A edge nodes, not stored
    as node properties. Use add_edge() with relation="IS_A" for hierarchy.
    """
    node_uuid = uuid or str(uuid4())
    now = datetime.now(UTC).isoformat()

    props = {
        "uuid": node_uuid,
        "name": name,
        "type": node_type,
        "created_at": now,
        "updated_at": now,
    }
    if properties:
        props.update(properties)

    query = """
    MERGE (n:Node {uuid: $uuid})
    SET n += $props
    RETURN n
    """
    self._execute_query(query, {"uuid": node_uuid, "props": props})
    return node_uuid
```

3. **Add `add_edge()` method** after `add_node()`:

```python
def add_edge(
    self,
    source_uuid: str,
    target_uuid: str,
    relation: str,
    edge_uuid: str | None = None,
    bidirectional: bool = False,
    properties: dict[str, Any] | None = None,
) -> str:
    """Create a reified edge node with :FROM/:TO structural relationships.

    The edge is stored as a node connected to source and target:
        (source)<-[:FROM]-(edge_node)-[:TO]->(target)

    These are the ONLY native Neo4j relationships in the graph.

    Args:
        source_uuid: Source node UUID
        target_uuid: Target node UUID
        relation: Edge type (e.g., "IS_A", "CAUSES", "HAS_STATE")
        edge_uuid: Edge node UUID (auto-generated if not provided)
        bidirectional: Whether the relationship is bidirectional
        properties: Additional properties on the edge node

    Returns:
        The UUID of the created edge node
    """
    eid = edge_uuid or str(uuid4())
    now = datetime.now(UTC).isoformat()

    props = {
        "uuid": eid,
        "type": "edge",
        "relation": relation,
        "source": source_uuid,
        "target": target_uuid,
        "bidirectional": bidirectional,
        "created_at": now,
        "updated_at": now,
    }
    if properties:
        props.update(properties)

    # MERGE on composite key (source + target + relation) for idempotency.
    # If the same structural relationship is proposed twice, update the
    # existing edge's metadata rather than creating a duplicate.
    # The edge name is set descriptively using source/target names for
    # debugging and display — this requires fetching the node names.
    query = """
    MATCH (src:Node {uuid: $source_uuid})
    MATCH (tgt:Node {uuid: $target_uuid})
    MERGE (edge:Node {source: $source_uuid, target: $target_uuid, relation: $relation})
    ON CREATE SET edge += $props,
                  edge.name = src.name + '_' + $relation + '_' + tgt.name
    ON MATCH SET edge.updated_at = $now
    MERGE (edge)-[:FROM]->(src)
    MERGE (edge)-[:TO]->(tgt)
    RETURN edge.uuid AS uuid
    """
    result = self._execute_query(query, {
        "source_uuid": source_uuid,
        "target_uuid": target_uuid,
        "relation": relation,
        "props": props,
        "now": now,
    })
    # If MERGE matched an existing edge, return its UUID instead
    if result and result[0].get("uuid"):
        return result[0]["uuid"]
    return eid
```

4. **Remove `add_typed_edge()`** (lines 981–1033) and **`add_relation()`** (lines 1035–1080). They create native Neo4j relationships which are no longer used. All edge creation goes through `add_edge()`.

### Step 9: Run tests to verify they pass

Run: `cd logos && poetry run pytest tests/test_edge_reification.py tests/test_edge_model.py -v`
Expected: PASS

### Step 10: Add Edge collection to HCGMilvusSync

In `logos/logos_hcg/sync.py`, update `COLLECTION_NAMES` (line 27) and `NodeType` (line 34):

```python
COLLECTION_NAMES = {
    "Entity": "hcg_entity_embeddings",
    "Concept": "hcg_concept_embeddings",
    "State": "hcg_state_embeddings",
    "Process": "hcg_process_embeddings",
    "Edge": "hcg_edge_embeddings",
}

NodeType = Literal["Entity", "Concept", "State", "Process", "Edge"]
```

Add an `ensure_collection()` method that creates a collection with the correct schema and index if it doesn't exist. This is needed for the new Edge collection and ensures existing collections have matching index params:

```python
def ensure_collection(self, node_type: NodeType) -> None:
    """Create collection with correct schema and L2 index if it doesn't exist."""
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

    name = COLLECTION_NAMES[node_type]
    if utility.has_collection(name):
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="uuid", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    ]
    schema = CollectionSchema(fields, description=f"Embeddings for {node_type}")
    collection = Collection(name=name, schema=schema)

    # Create IVF_FLAT index with L2 metric — must match search metric_type
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index("embedding", index_params)
    collection.load()
    logger.info(f"Created collection {name} with L2 IVF_FLAT index")
```

Call `ensure_collection("Edge")` in the `connect()` method after connecting to Milvus. This ensures the Edge collection exists with the correct index before any search or upsert operations.

Check if `HCGMilvusSync` has a `search_similar()` method. If not, add one:

```python
def search_similar(
    self,
    node_type: NodeType,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[dict]:
    """Search for similar embeddings in a node type collection.

    Args:
        node_type: Which collection to search
        query_embedding: Query vector (384-dim)
        top_k: Number of results

    Returns:
        List of dicts with keys: uuid, score (L2 distance — lower is more similar)
    """
    collection = self._collections.get(node_type)
    if not collection:
        logger.warning(f"Collection for {node_type} not available")
        return []

    try:
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["uuid"],
        )
        return [
            {"uuid": hit.entity.get("uuid"), "score": hit.distance}
            for hit in results[0]
        ]
    except Exception as e:
        logger.error(f"Embedding search failed for {node_type}: {e}")
        return []
```

### Step 10b: Add Neo4j index creation

Add a method to `HCGClient` that creates indexes for common query patterns. This should be called during seeding or as a setup step:

```python
def ensure_indexes(self) -> None:
    """Create Neo4j indexes for common query patterns.

    Indexes:
    - Uniqueness constraint on Node.uuid
    - Index on Node.type (used in most queries)
    - Index on Node.name (used in lookups)
    - Index on Node.relation (used in edge node queries)
    """
    indexes = [
        "CREATE CONSTRAINT node_uuid_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.uuid IS UNIQUE",
        "CREATE INDEX node_type_idx IF NOT EXISTS FOR (n:Node) ON (n.type)",
        "CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:Node) ON (n.name)",
        "CREATE INDEX node_relation_idx IF NOT EXISTS FOR (n:Node) ON (n.relation)",
    ]
    for query in indexes:
        try:
            self._execute_query(query, {})
        except Exception as e:
            logger.debug(f"Index creation (may already exist): {e}")
```

Call `ensure_indexes()` at the start of the seeder's `seed_type_definitions()` method.

### Step 11: Commit

```bash
cd logos && git add logos_hcg/client.py logos_hcg/sync.py tests/test_edge_reification.py && git commit -m "$(cat <<'EOF'
feat: reified edge creation — add_edge() with :FROM/:TO, clean add_node()

- add_edge() creates edge nodes with structural :FROM/:TO rels
- add_node() no longer accepts ancestors or is_type_definition
- Removed add_typed_edge() and add_relation() (native rels)
- Added Edge collection to HCGMilvusSync
- Added search_similar() to HCGMilvusSync
EOF
)"
```

---

## Task 2: Query Rewrite — logos foundry

**Why:** `logos/logos_hcg/queries.py` (1020 lines) assumes native Neo4j relationships throughout. Every relationship query, link creation, causality traversal, and planning query must be rewritten for edge node traversal via `:FROM`/`:TO`.

**Repo:** logos

**Files:**
- Modify: `logos_hcg/queries.py` (full rewrite)
- Test: `tests/test_queries_reified.py` (new)

### Step 1: Write failing tests for reified query patterns

```python
# tests/test_queries_reified.py
"""Integration tests for reified edge queries. Requires Neo4j."""
import pytest
from logos_hcg.client import HCGClient
from logos_hcg.queries import HCGQueries


@pytest.fixture
def client():
    c = HCGClient(uri="bolt://localhost:7687", user="neo4j", password="logosdev")
    yield c
    c.clear_all()
    c.close()


@pytest.fixture
def seeded_graph(client):
    """Create a small type hierarchy + instances using reified edges."""
    # Type definitions
    thing = client.add_node(name="thing", node_type="type_definition", uuid="type_thing")
    entity = client.add_node(name="entity", node_type="type_definition", uuid="type_entity")
    location = client.add_node(name="location", node_type="type_definition", uuid="type_location")

    # IS_A hierarchy: location -> entity -> thing
    client.add_edge(source_uuid=entity, target_uuid=thing, relation="IS_A")
    client.add_edge(source_uuid=location, target_uuid=entity, relation="IS_A")

    # Instances
    paris = client.add_node(name="Paris", node_type="location", uuid="paris")
    france = client.add_node(name="France", node_type="entity", uuid="france")
    client.add_edge(source_uuid=paris, target_uuid=location, relation="IS_A")
    client.add_edge(source_uuid=france, target_uuid=entity, relation="IS_A")

    # Knowledge edge
    client.add_edge(source_uuid=paris, target_uuid=france, relation="LOCATED_IN")

    return {"paris": paris, "france": france, "entity": entity,
            "location": location, "thing": thing}


class TestRelationshipQueries:

    def test_get_outgoing_edges(self, client, seeded_graph):
        """Query outgoing edges from a node via :FROM/:TO traversal."""
        query = HCGQueries.get_outgoing_edges()
        result = client._execute_read(query, {"uuid": seeded_graph["paris"]})
        relations = [r["relation"] for r in result]
        assert "LOCATED_IN" in relations
        assert "IS_A" in relations

    def test_get_typed_edges(self, client, seeded_graph):
        """Query edges of a specific type from a node."""
        query = HCGQueries.get_edges_by_relation()
        result = client._execute_read(
            query, {"uuid": seeded_graph["paris"], "relation": "LOCATED_IN"}
        )
        assert len(result) == 1
        assert result[0]["target_uuid"] == seeded_graph["france"]


class TestTypeHierarchy:

    def test_find_type_definitions(self, client, seeded_graph):
        """Find nodes that are targets of IS_A edges (type definitions)."""
        query = HCGQueries.find_type_definitions()
        result = client._execute_read(query, {})
        names = {r["name"] for r in result}
        assert "thing" in names
        assert "entity" in names
        assert "location" in names

    def test_has_ancestor(self, client, seeded_graph):
        """Check if a node has a given ancestor in its IS_A chain."""
        query = HCGQueries.has_ancestor()
        # Paris IS_A location IS_A entity IS_A thing
        result = client._execute_read(
            query, {"uuid": seeded_graph["paris"], "ancestor_name": "thing"}
        )
        assert len(result) > 0  # Paris is a descendant of "thing"

    def test_find_instances_of_type(self, client, seeded_graph):
        """Find all instances that have IS_A chain to a type."""
        query = HCGQueries.find_instances_of_type()
        result = client._execute_read(query, {"type_name": "entity"})
        uuids = {r["uuid"] for r in result}
        assert seeded_graph["france"] in uuids
```

### Step 2: Run tests to verify they fail

Run: `cd logos && poetry run pytest tests/test_queries_reified.py -v`
Expected: FAIL — the query methods don't exist or return wrong results.

### Step 3: Rewrite queries.py

Rewrite `logos/logos_hcg/queries.py`. The file is 1020 lines; here are the key patterns. Apply the same transformation to every query.

**IS_A traversal utility** — add this as a class method at the top of `HCGQueries`:

```python
@staticmethod
def _is_a_check(node_var: str, ancestor_param: str, max_depth: int = 5) -> str:
    """Generate a WHERE clause checking IS_A ancestry via reified edge nodes.

    Checks up to max_depth levels of IS_A hierarchy. The current ontology
    has max depth ~3 (e.g., agent -> physical_entity -> entity -> thing),
    so depth 5 covers all cases with margin.

    Args:
        node_var: Cypher variable name of the node to check
        ancestor_param: Name of the Cypher parameter holding the ancestor name
        max_depth: Maximum IS_A chain depth to check

    Returns:
        Cypher WHERE clause fragment (without WHERE keyword)
    """
    clauses = []
    for depth in range(1, max_depth + 1):
        # Build chain: (n)<-[:FROM]-(e1)-[:TO]->(t1)<-[:FROM]-(e2)-[:TO]->(t2)...
        chain = f"({node_var})"
        for i in range(depth):
            chain += f"<-[:FROM]-(:Node {{relation: 'IS_A'}})-[:TO]->()"
        # Replace last () with named target
        chain = chain[:-2] + f"(anc_{depth}:Node {{name: ${ancestor_param}}})"
        clauses.append(f"EXISTS {{ MATCH {chain} }}")
    return " OR ".join(clauses)
```

**Rewritten query categories:**

**a) Generic node queries** — `find_node_by_uuid()`, `find_nodes_by_type()`, `find_node_by_name()` stay mostly the same (they don't use relationships). Remove `find_nodes_by_ancestor()` — replace with `find_instances_of_type()`.

**b) Type definition queries** — rewrite to use IS_A structure:

```python
@staticmethod
def find_type_definitions() -> str:
    """Find all type definitions (nodes that are targets of IS_A edges)."""
    return """
    MATCH (instance:Node)<-[:FROM]-(e:Node {relation: "IS_A"})-[:TO]->(type_def:Node)
    RETURN DISTINCT type_def.uuid AS uuid, type_def.name AS name,
           type_def.type AS type
    ORDER BY type_def.name
    """

@staticmethod
def find_instances_of_type() -> str:
    """Find all instances whose IS_A chain reaches a given type.
    Parameters: $type_name
    """
    # Direct instances + instances of subtypes (up to 5 levels)
    return """
    MATCH (instance:Node)<-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->(t1:Node)
    WHERE t1.name = $type_name
    RETURN DISTINCT instance.uuid AS uuid, instance.name AS name,
           instance.type AS type
    UNION
    MATCH (instance:Node)<-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->(t1:Node)
          <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->(t2:Node)
    WHERE t2.name = $type_name
    RETURN DISTINCT instance.uuid AS uuid, instance.name AS name,
           instance.type AS type
    UNION
    MATCH (instance:Node)<-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->(t1:Node)
          <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->(t2:Node)
          <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->(t3:Node)
    WHERE t3.name = $type_name
    RETURN DISTINCT instance.uuid AS uuid, instance.name AS name,
           instance.type AS type
    """
```

**c) Relationship queries** — rewrite `get_entity_parts()`, `get_entity_states()`, `get_process_preconditions()`, `get_process_effects()`, etc.:

```python
@staticmethod
def get_outgoing_edges() -> str:
    """Get all outgoing edge nodes from a node.
    Parameters: $uuid (source node)
    Returns: edge node properties + target node uuid
    """
    return """
    MATCH (source:Node {uuid: $uuid})<-[:FROM]-(edge:Node)-[:TO]->(target:Node)
    RETURN edge.uuid AS edge_uuid, edge.relation AS relation,
           target.uuid AS target_uuid, target.name AS target_name,
           edge.bidirectional AS bidirectional, edge.confidence AS confidence
    """

@staticmethod
def get_edges_by_relation() -> str:
    """Get edges of a specific relation type from a node.
    Parameters: $uuid (source), $relation (e.g., "HAS_STATE")
    """
    return """
    MATCH (source:Node {uuid: $uuid})<-[:FROM]-(edge:Node {relation: $relation})-[:TO]->(target:Node)
    RETURN edge.uuid AS edge_uuid, target.uuid AS target_uuid,
           target.name AS target_name, target.type AS target_type,
           edge.bidirectional AS bidirectional
    """

@staticmethod
def get_entity_states() -> str:
    """Get all states of an entity. Parameters: $entity_uuid"""
    return """
    MATCH (e:Node {uuid: $entity_uuid})<-[:FROM]-(:Node {relation: "HAS_STATE"})-[:TO]->(s:Node)
    RETURN s
    ORDER BY s.timestamp DESC
    """

@staticmethod
def get_process_preconditions() -> str:
    """Get states required by a process. Parameters: $process_uuid"""
    return """
    MATCH (p:Node {uuid: $process_uuid})<-[:FROM]-(:Node {relation: "REQUIRES"})-[:TO]->(s:Node)
    RETURN s
    """

@staticmethod
def get_process_effects() -> str:
    """Get states caused by a process. Parameters: $process_uuid"""
    return """
    MATCH (p:Node {uuid: $process_uuid})<-[:FROM]-(:Node {relation: "CAUSES"})-[:TO]->(s:Node)
    RETURN s
    """
```

**d) Causality traversal** — rewrite `traverse_causality_forward()`:

```python
@staticmethod
def traverse_causality_forward(max_depth: int = 5) -> str:
    """Traverse causality: state -> [process REQUIRES state, process CAUSES result].
    Parameters: $state_uuid
    """
    return """
    MATCH (start:Node {uuid: $state_uuid})
    MATCH (start)<-[:TO]-(:Node {relation: "REQUIRES"})<-[:FROM]-(p:Node)
    MATCH (p)<-[:FROM]-(:Node {relation: "CAUSES"})-[:TO]->(result:Node)
    RETURN p, result
    """
```

**e) Link creation queries** — remove entirely. All link creation goes through `client.add_edge()`. Delete `link_is_a()`, `link_has_state()`, `link_requires()`, `link_causes()`.

**f) `has_ancestor()` query** — for checking IS_A chain membership:

```python
@staticmethod
def has_ancestor() -> str:
    """Check if a node has a given ancestor in IS_A chain (up to 5 levels).
    Parameters: $uuid, $ancestor_name
    """
    return """
    MATCH (n:Node {uuid: $uuid})
    WHERE
      EXISTS {
        MATCH (n)<-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->(a1:Node {name: $ancestor_name})
      }
      OR EXISTS {
        MATCH (n)<-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->()
              <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->(a2:Node {name: $ancestor_name})
      }
      OR EXISTS {
        MATCH (n)<-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->()
              <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->()
              <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->(a3:Node {name: $ancestor_name})
      }
      OR EXISTS {
        MATCH (n)<-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->()
              <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->()
              <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->()
              <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->(a4:Node {name: $ancestor_name})
      }
      OR EXISTS {
        MATCH (n)<-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->()
              <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->()
              <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->()
              <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->()
              <-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->(a5:Node {name: $ancestor_name})
      }
    RETURN n
    """
```

**Apply this pattern to every query in the file.** The complete list of queries that need rewriting (anything that references native relationships):

- `get_type_hierarchy()` — was `[:IS_A*]`, now edge node traversal
- `get_entity_type()` — was `[:IS_A]`, now single-hop edge node
- `get_entity_parts()` — was `[:PART_OF]`
- `get_entity_parent()` — was `[:PART_OF]`
- `get_entity_states()` — was `[:HAS_STATE]`
- `get_entity_current_state()` — was `[:HAS_STATE]`
- `get_process_preconditions()` — was `[:REQUIRES]`
- `get_process_effects()` — was `[:CAUSES]`
- `traverse_causality_forward()` — was `[:REQUIRES]`/`[:CAUSES]`
- `traverse_causality_backward()` — same
- `find_processes_causing_state()` — was `[:CAUSES]`
- `find_processes_for_entity_state()` — was `[:HAS_STATE]`/`[:CAUSES]`
- `find_capability_for_process()` — was `[:USES_CAPABILITY]`
- `check_state_satisfied()` — was `[:HAS_STATE]`
- `link_is_a()`, `link_has_state()`, `link_requires()`, `link_causes()` — DELETE (use `add_edge()`)
- All queries using `WHERE ... IN n.ancestors` — use `has_ancestor()` or `find_instances_of_type()`

Also remove all queries that filter by `is_type_definition` property — replace with IS_A structure queries.

### Step 4: Run tests

Run: `cd logos && poetry run pytest tests/test_queries_reified.py -v`
Expected: PASS

### Step 5: Run full foundry test suite

Run: `cd logos && poetry run pytest tests/ -v`
Fix any failures from the query changes.

### Step 6: Commit

```bash
cd logos && git add logos_hcg/queries.py tests/test_queries_reified.py && git commit -m "$(cat <<'EOF'
feat: rewrite queries.py for edge node traversal via :FROM/:TO

All relationship queries now traverse reified edge nodes instead of
native Neo4j relationships. IS_A hierarchy uses fixed-depth traversal
(up to 5 levels). Link creation queries removed — use add_edge().
EOF
)"
```

---

## Task 3: Update Seeder and Planner — logos foundry

**Why:** The seeder creates type hierarchy using `ancestors` lists and native relationships. The planner backward-chains over native `[:REQUIRES]`/`[:CAUSES]`. Both need to use reified edge nodes.

**Repo:** logos

**Files:**
- Modify: `logos_hcg/seeder.py`
- Modify: `logos_hcg/planner.py`
- Test: `tests/test_seeder_reified.py` (new)

### Step 1: Write failing test for seeder

```python
# tests/test_seeder_reified.py
"""Tests for seeder with reified IS_A edges. Requires Neo4j."""
import pytest
from logos_hcg.client import HCGClient
from logos_hcg.seeder import HCGSeeder


@pytest.fixture
def client():
    c = HCGClient(uri="bolt://localhost:7687", user="neo4j", password="logosdev")
    c.clear_all()
    yield c
    c.clear_all()
    c.close()


def test_seed_creates_is_a_edge_nodes(client):
    """Type hierarchy should use IS_A edge nodes, not native relationships."""
    seeder = HCGSeeder(client)
    seeder.seed_type_definitions()

    # Verify IS_A edge node exists between entity and thing
    result = client._execute_read("""
        MATCH (child:Node {name: "entity"})<-[:FROM]-(e:Node {relation: "IS_A"})-[:TO]->(parent:Node {name: "thing"})
        RETURN e.uuid AS edge_uuid
    """, {})
    assert len(result) > 0

def test_seed_no_native_is_a_relationships(client):
    """No native [:IS_A] relationships should exist after seeding."""
    seeder = HCGSeeder(client)
    seeder.seed_type_definitions()

    result = client._execute_read(
        "MATCH ()-[r:IS_A]->() RETURN count(r) AS count", {}
    )
    assert result[0]["count"] == 0

def test_seed_no_ancestors_property(client):
    """Nodes should not have an 'ancestors' property after seeding."""
    seeder = HCGSeeder(client)
    seeder.seed_type_definitions()

    result = client._execute_read("""
        MATCH (n:Node)
        WHERE n.ancestors IS NOT NULL
        RETURN count(n) AS count
    """, {})
    assert result[0]["count"] == 0

def test_seed_demo_uses_add_edge(client):
    """Demo scenario edges should be reified edge nodes."""
    seeder = HCGSeeder(client)
    seeder.seed_type_definitions()
    seeder.seed_demo_scenario()

    # Check that a PART_OF relationship is an edge node
    result = client._execute_read("""
        MATCH (child:Node)<-[:FROM]-(e:Node {relation: "PART_OF"})-[:TO]->(parent:Node)
        RETURN count(e) AS count
    """, {})
    assert result[0]["count"] > 0
```

### Step 2: Run test to verify it fails

Run: `cd logos && poetry run pytest tests/test_seeder_reified.py -v`
Expected: FAIL — seeder still uses old methods.

### Step 3: Rewrite seeder

In `logos/logos_hcg/seeder.py`:

**a) Replace `ANCESTORS` dict with `TYPE_PARENTS`** — only immediate parent needed:

```python
TYPE_PARENTS: dict[str, str] = {
    "entity": "thing",
    "physical_entity": "entity",
    "agent": "physical_entity",
    "object": "physical_entity",
    "manipulator": "physical_entity",
    "sensor": "physical_entity",
    "spatial_entity": "entity",
    "location": "spatial_entity",
    "workspace": "spatial_entity",
    "zone": "spatial_entity",
    "process": "entity",
    "action": "process",
    "step": "process",
    "imagined_process": "process",
    "proposed_plan_step": "process",
    "proposed_tool_call": "process",
    "intention": "entity",
    "goal": "intention",
    "plan": "intention",
    "hermes_proposal": "intention",
    "abstraction": "entity",
    "simulation": "abstraction",
    "execution": "abstraction",
    "data": "entity",
    "media_sample": "data",
    "capability": "data",
    "constraint": "concept",
    "state": "cognition",
    "imagined_state": "cognition",
    "proposed_imagined_state": "cognition",
}
```

**b) Rewrite `seed_type_definitions()`:**

```python
def seed_type_definitions(self) -> int:
    """Create type definition nodes and IS_A hierarchy via edge nodes."""
    count = 0

    # Create type definition nodes
    for type_name, parent in TYPE_PARENTS.items():
        if type_name in BOOTSTRAP_TYPES:
            continue
        self.client.add_node(
            uuid=f"type_{type_name}",
            name=type_name,
            node_type=parent,
        )
        count += 1

    # Create edge type definition nodes
    for edge_name in EDGE_TYPES:
        if edge_name in BOOTSTRAP_TYPES:
            continue
        self.client.add_node(
            uuid=f"type_edge_{edge_name.lower()}",
            name=edge_name,
            node_type="edge_type",
        )
        count += 1

    # Create IS_A edge nodes for type hierarchy
    for type_name, parent in TYPE_PARENTS.items():
        if type_name in BOOTSTRAP_TYPES:
            continue
        child_uuid = f"type_{type_name}"
        parent_uuid = f"type_{parent}" if parent not in BOOTSTRAP_TYPES else parent
        self.client.add_edge(
            source_uuid=child_uuid,
            target_uuid=parent_uuid,
            relation="IS_A",
        )

    logger.info("Seeded %d type definitions", count)
    return count
```

**c) Update `seed_demo_scenario()`** — change the `_n()` helper to not pass ancestors, and change the `_e()` helper to use `add_edge()`:

```python
def _n(key, name, node_type, *, props=None):
    uuid = str(uuid4())
    self.client.add_node(uuid=uuid, name=name, node_type=node_type, properties=props)
    ids[key] = uuid
    # Create IS_A edge to type definition
    type_uuid = f"type_{node_type}" if f"type_{node_type}" not in BOOTSTRAP_TYPES else node_type
    self.client.add_edge(source_uuid=uuid, target_uuid=type_uuid, relation="IS_A")
    return uuid

def _e(src, tgt, rel, **kw):
    self.client.add_edge(
        source_uuid=ids[src], target_uuid=ids[tgt],
        relation=rel, properties=kw or None,
    )
```

### Step 4: Update planner

In `logos/logos_hcg/planner.py`, the `_find_achieving_processes()` and `_find_requirements()` methods query the graph using native `[:REQUIRES]`/`[:CAUSES]` relationships. Update them to use the reified pattern.

The planner uses `HCGClient` methods to query — check which methods it calls and ensure they've been updated (or update them here). The key pattern change is:

```python
# Before (in planner's graph queries)
MATCH (p:Node)-[:CAUSES]->(s:Node {uuid: $state_uuid})

# After
MATCH (p:Node)<-[:FROM]-(:Node {relation: "CAUSES"})-[:TO]->(s:Node {uuid: $state_uuid})
```

### Step 5: Run tests

Run: `cd logos && poetry run pytest tests/ -v`
Expected: PASS

### Step 6: Commit

```bash
cd logos && git add logos_hcg/seeder.py logos_hcg/planner.py tests/test_seeder_reified.py && git commit -m "$(cat <<'EOF'
feat: update seeder and planner for reified IS_A edge nodes

- Seeder creates IS_A edge nodes for type hierarchy
- ANCESTORS dict replaced with TYPE_PARENTS (immediate parent only)
- Demo scenario uses add_edge() for all relationships
- Planner queries use edge node traversal
EOF
)"
```

---

## Task 4: Sophia Alignment

**Why:** Sophia's knowledge graph models, HCGClient extension, SHACL validator, and seeder all need to align with the reified edge model.

**Repo:** sophia

**Files:**
- Modify: `src/sophia/knowledge_graph/edge.py` — add `bidirectional`
- Modify: `src/sophia/knowledge_graph/node.py` — drop `is_type_definition`, `ancestors`
- Modify: `src/sophia/knowledge_graph/graph.py` — edge-as-node pattern
- Modify: `src/sophia/hcg_client/client.py` — rewrite edge methods, drop `_get_type_ancestors()`
- Modify: `src/sophia/hcg_client/shacl_validator.py` — edge node shapes
- Modify: `src/sophia/hcg_client/seeder.py` — align with reified model
- Test: `tests/unit/knowledge_graph/test_models_reified.py` (new)
- Test: `tests/unit/hcg_client/test_client_reified.py` (new)

### Step 1: Write failing tests for updated models

```python
# tests/unit/knowledge_graph/test_models_reified.py
"""Tests for updated Node/Edge models."""


def test_node_has_no_ancestors():
    from sophia.knowledge_graph.node import Node
    n = Node(name="Paris", type="location")
    assert not hasattr(n, "ancestors") or "ancestors" not in n.model_fields

def test_node_has_no_is_type_definition():
    from sophia.knowledge_graph.node import Node
    n = Node(name="Paris", type="location")
    assert not hasattr(n, "is_type_definition") or "is_type_definition" not in n.model_fields

def test_edge_has_bidirectional():
    from sophia.knowledge_graph.edge import Edge
    e = Edge(source="a", target="b", relation="RELATED_TO", bidirectional=True)
    assert e.bidirectional is True

def test_edge_bidirectional_defaults_false():
    from sophia.knowledge_graph.edge import Edge
    e = Edge(source="a", target="b", relation="IS_A")
    assert e.bidirectional is False
```

### Step 2: Run test to verify it fails

Run: `cd sophia && poetry run pytest tests/unit/knowledge_graph/test_models_reified.py -v`
Expected: FAIL — Node still has ancestors, Edge lacks bidirectional.

### Step 3: Update Node model

In `sophia/src/sophia/knowledge_graph/node.py`, remove `is_type_definition` and `ancestors` fields:

```python
class Node(BaseModel):
    """Represents a node in the knowledge graph.

    Type hierarchy is expressed via IS_A edge nodes, not stored properties.
    """
    model_config = ConfigDict(frozen=False)

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.uuid

    def __hash__(self) -> int:
        return hash(self.uuid)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.uuid == other.uuid
```

### Step 4: Update Edge model

In `sophia/src/sophia/knowledge_graph/edge.py`, add `bidirectional`:

```python
class Edge(BaseModel):
    """Reified edge — stored as a node in Neo4j with :FROM/:TO connectors."""
    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    target: str
    relation: str
    bidirectional: bool = False
    properties: Dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False
        return self.id == other.id
```

### Step 5: Update KnowledgeGraph

In `sophia/src/sophia/knowledge_graph/graph.py`, edges are now nodes in the graph. The `add_edge()` method needs to store the edge as a node connected to source and target:

```python
def add_edge(self, edge: Edge) -> None:
    """Add a reified edge — the edge is a node connected via structural edges."""
    if edge.source not in self._nodes:
        raise ValueError(f"Source node {edge.source} not found")
    if edge.target not in self._nodes:
        raise ValueError(f"Target node {edge.target} not found")

    self._edges[edge.id] = edge
    # Store as a node in the NetworkX graph
    self._graph.add_node(edge.id, data=edge, is_edge=True)
    # Structural connections: edge -> source (FROM), edge -> target (TO)
    self._graph.add_edge(edge.id, edge.source, rel="FROM")
    self._graph.add_edge(edge.id, edge.target, rel="TO")

def get_edges_from(self, node_id: str) -> List[Edge]:
    """Get all outgoing edges from a node (where node is the source)."""
    if node_id not in self._nodes:
        return []
    return [e for e in self._edges.values() if e.source == node_id]

def get_edges_to(self, node_id: str) -> List[Edge]:
    """Get all incoming edges to a node (where node is the target)."""
    if node_id not in self._nodes:
        return []
    return [e for e in self._edges.values() if e.target == node_id]
```

Update `remove_node()` to also remove edge nodes that reference the removed node.

### Step 6: Run tests

Run: `cd sophia && poetry run pytest tests/unit/knowledge_graph/ -v`
Expected: PASS. Fix any failures from model changes.

### Step 7: Rewrite sophia's HCGClient

In `sophia/src/sophia/hcg_client/client.py`:

**a) Remove `_get_type_ancestors()`** (line 71) — ancestry is graph traversal now.

**b) Update `add_node()`** (line 92) — remove `ancestors` and `is_type_definition` params. Keep provenance params (`source`, `derivation`, `confidence`, `tags`, `links`). Remove auto-ancestor computation. Remove the `SET n.ancestors = $ancestors, n.is_type_definition = $is_type_definition` from the Cypher query.

**c) Rewrite `add_edge()`** (line 292) — currently creates `:RELATION` native edges. Replace with reified edge node creation via the foundry's `add_edge()`:

```python
def add_edge(
    self,
    edge_id: str,
    source_uuid: str,
    target_uuid: str,
    relation: str,
    properties: Optional[Dict[str, Any]] = None,
    bidirectional: bool = False,
) -> str:
    """Create a reified edge node after SHACL validation."""
    edge_data = {
        "id": edge_id,
        "source": source_uuid,
        "target": target_uuid,
        "relation": relation,
        "bidirectional": bidirectional,
        "properties": properties or {},
    }
    is_valid, errors = self._validator.validate_edge(edge_data)
    if not is_valid:
        raise ValueError(f"Edge validation failed: {'; '.join(errors)}")

    # Use foundry's add_edge() which creates :FROM/:TO structure
    return super().add_edge(
        source_uuid=source_uuid,
        target_uuid=target_uuid,
        relation=relation,
        edge_uuid=edge_id,
        bidirectional=bidirectional,
        properties=properties,
    )
```

**d) Rewrite `get_edge()`** (line 362) — query edge nodes instead of `:RELATION`:

```python
def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a reified edge node by ID."""
    query = """
    MATCH (edge:Node {uuid: $id, type: "edge"})
    RETURN edge.uuid AS id, edge.source AS source, edge.target AS target,
           edge.relation AS relation, edge.bidirectional AS bidirectional,
           properties(edge) AS props
    """
    records = self._execute_read(query, {"id": edge_id})
    if not records:
        return None
    props = dict(records[0]["props"])
    for key in ["uuid", "name", "type", "relation", "source", "target", "bidirectional"]:
        props.pop(key, None)
    props = self._decode_properties(props)
    return {
        "id": records[0]["id"],
        "source": records[0]["source"],
        "target": records[0]["target"],
        "relation": records[0]["relation"],
        "bidirectional": records[0]["bidirectional"],
        "properties": props,
    }
```

**e) Rewrite `query_edges_from()`** (line 412) — traverse `:FROM`/`:TO`:

```python
def query_edges_from(self, uuid: str) -> List[Dict[str, Any]]:
    """Return outgoing reified edges from a node."""
    query = """
    MATCH (source:Node {uuid: $uuid})<-[:FROM]-(edge:Node)-[:TO]->(target:Node)
    RETURN edge.uuid AS id, source.uuid AS source, target.uuid AS target,
           edge.relation AS relation, edge.bidirectional AS bidirectional,
           properties(edge) AS props
    """
    records = self._execute_read(query, {"uuid": uuid})
    results = []
    for record in records:
        props = dict(record["props"])
        for key in ["uuid", "name", "type", "relation", "source", "target", "bidirectional"]:
            props.pop(key, None)
        props = self._decode_properties(props)
        results.append({
            "id": record["id"],
            "source": record["source"],
            "target": record["target"],
            "relation": record["relation"],
            "bidirectional": record["bidirectional"],
            "properties": props,
        })
    return results
```

**f) Rewrite `query_neighbors()`** (line 385) — find all nodes connected via edge nodes:

```python
def query_neighbors(self, uuid: str) -> List[Dict[str, Any]]:
    """Return all nodes connected to this node via any edge (outgoing or incoming)."""
    query = """
    MATCH (n:Node {uuid: $uuid})<-[:FROM]-(edge:Node)-[:TO]->(target:Node)
    RETURN target.uuid AS uuid, target.name AS name, target.type AS type,
           edge.relation AS via_relation, 'outgoing' AS direction
    UNION
    MATCH (n:Node {uuid: $uuid})<-[:TO]-(edge:Node)<-[:FROM]-(source:Node)
    RETURN source.uuid AS uuid, source.name AS name, source.type AS type,
           edge.relation AS via_relation, 'incoming' AS direction
    """
    records = self._execute_read(query, {"uuid": uuid})
    return [dict(r) for r in records]
```

**g) Rewrite `list_all_edges()`** — query all edge nodes in the graph:

```python
def list_all_edges(self) -> List[Dict[str, Any]]:
    """Return all reified edge nodes."""
    query = """
    MATCH (source:Node)<-[:FROM]-(edge:Node)-[:TO]->(target:Node)
    WHERE edge.type = "edge" OR edge.relation IS NOT NULL
    RETURN edge.uuid AS id, edge.relation AS relation,
           source.uuid AS source, target.uuid AS target,
           edge.bidirectional AS bidirectional,
           properties(edge) AS props
    """
    records = self._execute_read(query, {})
    results = []
    for record in records:
        props = dict(record["props"])
        for key in ["uuid", "name", "type", "relation", "source", "target", "bidirectional"]:
            props.pop(key, None)
        props = self._decode_properties(props)
        results.append({
            "id": record["id"],
            "source": record["source"],
            "target": record["target"],
            "relation": record["relation"],
            "bidirectional": record["bidirectional"],
            "properties": props,
        })
    return results
```

**h) Add graph retrieval method** — returns nodes and edges as clean objects for Sophia to reason about:

```python
def get_subgraph(self, node_uuids: List[str]) -> Dict[str, Any]:
    """Return a subgraph as clean Node/Edge objects.

    Given a set of node UUIDs, returns those nodes and all reified
    edge nodes connecting them. This is how Sophia reasons about
    the graph without needing to understand :FROM/:TO internals.
    """
    query = """
    MATCH (n:Node) WHERE n.uuid IN $uuids
    OPTIONAL MATCH (n)<-[:FROM]-(edge:Node)-[:TO]->(target:Node)
    WHERE target.uuid IN $uuids
    RETURN collect(DISTINCT properties(n)) AS nodes,
           collect(DISTINCT properties(edge)) AS edges
    """
    records = self._execute_read(query, {"uuids": node_uuids})
    if not records:
        return {"nodes": [], "edges": []}

    raw_nodes = records[0].get("nodes", [])
    raw_edges = records[0].get("edges", [])

    nodes = []
    for n in raw_nodes:
        if n is None:
            continue
        n = dict(n)
        nodes.append({
            "uuid": n.get("uuid"),
            "name": n.get("name"),
            "type": n.get("type"),
            "properties": {k: v for k, v in n.items()
                          if k not in ("uuid", "name", "type", "created_at", "updated_at")},
        })

    edges = []
    for e in raw_edges:
        if e is None:
            continue
        e = dict(e)
        edges.append({
            "id": e.get("uuid"),
            "source": e.get("source"),
            "target": e.get("target"),
            "relation": e.get("relation"),
            "bidirectional": e.get("bidirectional", False),
        })

    return {"nodes": nodes, "edges": edges}
```

**i) Rewrite `get_node()`** (line 336) — remove `is_type_definition` and `ancestors` from return:

```python
def get_node(self, uuid: str) -> Optional[Dict[str, Any]]:
    """Fetch a node by UUID."""
    query = """
    MATCH (n:Node {uuid: $uuid})
    RETURN n.uuid AS uuid, n.name AS name, n.type AS type,
           properties(n) AS props
    """
    records = self._execute_read(query, {"uuid": uuid})
    if not records:
        return None
    props = dict(records[0]["props"])
    for key in ["uuid", "name", "type"]:
        props.pop(key, None)
    props = self._decode_properties(props)
    return {
        "uuid": records[0]["uuid"],
        "name": records[0]["name"],
        "type": records[0]["type"],
        "properties": props,
    }
```

**j) Rewrite `list_all_nodes()`** (line 437) — remove `is_type_definition` and `ancestors`:

```python
def list_all_nodes(self) -> List[Dict[str, Any]]:
    """Return all non-edge nodes."""
    query = """
    MATCH (n:Node)
    WHERE n.relation IS NULL
    RETURN n.uuid AS uuid, n.name AS name, n.type AS type,
           properties(n) AS props
    ORDER BY n.name
    """
    records = self._execute_read(query, {})
    results = []
    for record in records:
        props = dict(record["props"])
        for key in ["uuid", "name", "type"]:
            props.pop(key, None)
        props = self._decode_properties(props)
        results.append({
            "uuid": record["uuid"],
            "name": record["name"],
            "type": record["type"],
            "properties": props,
        })
    return results
```

### Step 8: Update SHACLValidator

In `sophia/src/sophia/hcg_client/shacl_validator.py`, rewrite the edge shape to validate edge nodes instead of native relationships:

```python
EDGE_NODE_SHAPE = {
    "required_properties": {
        "relation": {"type": "string", "description": "Edge type (e.g., IS_A, CAUSES)"},
        "source": {"type": "string", "description": "Source node UUID"},
        "target": {"type": "string", "description": "Target node UUID"},
    },
    "optional_properties": {
        "bidirectional": {"type": "boolean", "default": False},
        "confidence": {"type": "float", "min": 0.0, "max": 1.0},
        "source_service": {"type": "string"},
        "derivation": {"type": "string"},
    },
}

def validate_edge(self, edge_data: dict) -> tuple[bool, list[str]]:
    """Validate an edge node against the SHACL shape."""
    errors = []
    for prop, spec in EDGE_NODE_SHAPE["required_properties"].items():
        if prop not in edge_data or not edge_data[prop]:
            errors.append(f"Missing required property: {prop}")
        elif spec["type"] == "string" and not isinstance(edge_data[prop], str):
            errors.append(f"Property {prop} must be a string")

    if "bidirectional" in edge_data:
        if not isinstance(edge_data["bidirectional"], bool):
            errors.append("bidirectional must be a boolean")

    if "confidence" in edge_data.get("properties", {}):
        conf = edge_data["properties"]["confidence"]
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            errors.append("confidence must be a float between 0.0 and 1.0")

    return (len(errors) == 0, errors)
```

### Step 9: Update sophia's seeder

In `sophia/src/sophia/hcg_client/seeder.py`, follow the same pattern as the logos seeder:

```python
# Replace any ancestors-based hierarchy with IS_A edge nodes.
# The seeder should call add_edge() for all relationships.

def seed_type_definitions(self) -> int:
    """Create type definition nodes and IS_A edges."""
    count = 0
    for type_name, parent in TYPE_PARENTS.items():
        self._client.add_node(
            uuid=f"type_{type_name}",
            name=type_name,
            node_type=parent,
        )
        parent_uuid = f"type_{parent}"
        self._client.add_edge(
            edge_id=f"isa_{type_name}_{parent}",
            source_uuid=f"type_{type_name}",
            target_uuid=parent_uuid,
            relation="IS_A",
        )
        count += 1
    return count
```

Remove any references to `ancestors`, `is_type_definition`, `_get_type_ancestors()`, or native relationship creation.

### Step 10: Run full sophia test suite

Run: `cd sophia && poetry run pytest tests/ -v`
Fix all failures. Key areas:
- Tests that assert `ancestors` on nodes
- Tests that pass `is_type_definition` to `add_node()`
- Tests that query native `[:RELATION]` edges
- Integration tests in `tests/integration/`

### Step 11: Commit

```bash
cd sophia && git add src/ tests/ && git commit -m "$(cat <<'EOF'
feat: align sophia with reified edge model

- Node model: dropped is_type_definition and ancestors
- Edge model: added bidirectional flag
- KnowledgeGraph: edges are nodes connected via structural rels
- HCGClient: reified add_edge(), get_edge(), query_edges_from()
- Removed _get_type_ancestors() — hierarchy is graph traversal
- Added get_subgraph() for clean graph retrieval
- Updated SHACLValidator for edge node shapes
- Updated seeder for IS_A edge nodes
EOF
)"
```

---

## Task 5: Hermes Proposal Builder

**Why:** Hermes currently forwards raw LLM text to Sophia (`_forward_llm_to_sophia()` at `main.py:589`). He should extract entities, generate embeddings, and build a structured proposal. Hermes is the only component that understands text.

**Repo:** hermes

**Files:**
- Create: `src/hermes/proposal_builder.py`
- Test: `tests/unit/test_proposal_builder.py` (new)

### Step 1: Write failing test for ProposalBuilder

```python
# tests/unit/test_proposal_builder.py
"""Tests for ProposalBuilder — builds structured proposals from text."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
class TestProposalBuilder:

    async def test_build_returns_required_fields(self):
        from hermes.proposal_builder import ProposalBuilder
        builder = ProposalBuilder()

        with patch("hermes.proposal_builder.process_nlp", new_callable=AsyncMock) as mock_nlp, \
             patch("hermes.proposal_builder.generate_embedding", new_callable=AsyncMock) as mock_emb:
            mock_nlp.return_value = {"entities": []}
            mock_emb.return_value = {
                "embedding": [0.1] * 384, "dimension": 384,
                "model": "all-MiniLM-L6-v2", "embedding_id": "doc-id",
            }
            proposal = await builder.build(text="Hello world", metadata={})

        assert "proposal_id" in proposal
        assert "proposed_nodes" in proposal
        assert "document_embedding" in proposal
        assert proposal["source_service"] == "hermes"

    async def test_extracts_entities_as_proposed_nodes(self):
        from hermes.proposal_builder import ProposalBuilder
        builder = ProposalBuilder()

        with patch("hermes.proposal_builder.process_nlp", new_callable=AsyncMock) as mock_nlp, \
             patch("hermes.proposal_builder.generate_embedding", new_callable=AsyncMock) as mock_emb:
            mock_nlp.return_value = {
                "entities": [
                    {"text": "Eiffel Tower", "label": "FAC", "start": 4, "end": 16},
                    {"text": "Paris", "label": "GPE", "start": 23, "end": 28},
                ]
            }
            mock_emb.return_value = {
                "embedding": [0.1] * 384, "dimension": 384,
                "model": "all-MiniLM-L6-v2", "embedding_id": "test-id",
            }
            proposal = await builder.build(
                text="The Eiffel Tower is in Paris", metadata={},
            )

        assert len(proposal["proposed_nodes"]) == 2
        assert proposal["proposed_nodes"][0]["name"] == "Eiffel Tower"
        assert proposal["proposed_nodes"][0]["type"] == "object"  # FAC → object
        assert proposal["proposed_nodes"][1]["type"] == "location"  # GPE → location
        assert "embedding" in proposal["proposed_nodes"][0]

    async def test_degrades_gracefully_without_nlp(self):
        from hermes.proposal_builder import ProposalBuilder
        builder = ProposalBuilder()

        with patch("hermes.proposal_builder.process_nlp", new_callable=AsyncMock) as mock_nlp, \
             patch("hermes.proposal_builder.generate_embedding", new_callable=AsyncMock) as mock_emb:
            mock_nlp.side_effect = Exception("spaCy not available")
            mock_emb.return_value = {
                "embedding": [0.1] * 384, "dimension": 384,
                "model": "all-MiniLM-L6-v2", "embedding_id": "fallback-id",
            }
            proposal = await builder.build(text="Hello", metadata={})

        assert proposal["proposed_nodes"] == []
        assert proposal["document_embedding"] is not None
```

### Step 2: Run test to verify it fails

Run: `cd hermes && poetry run pytest tests/unit/test_proposal_builder.py -v`
Expected: FAIL — module does not exist.

### Step 3: Implement ProposalBuilder

```python
# src/hermes/proposal_builder.py
"""Builds structured graph-ready proposals from conversational text.

Hermes is the only component that understands language. The ProposalBuilder
translates text into structured data (entities + embeddings) that Sophia
can process without reading text.
"""
import logging
import uuid as uuid_mod
from datetime import UTC, datetime

from hermes.services import generate_embedding, process_nlp

logger = logging.getLogger(__name__)

# Map spaCy NER labels to ontology type definitions.
# spaCy produces labels like GPE, FAC, ORG; the ontology has types like
# location, object, agent. Without this mapping, Sophia receives proposals
# with types that don't match any type definition in the graph.
SPACY_TO_ONTOLOGY: dict[str, str] = {
    "GPE": "location",       # geopolitical entity (cities, countries)
    "LOC": "location",       # non-GPE location (mountain ranges, water bodies)
    "FAC": "object",         # facility (buildings, airports, bridges)
    "ORG": "agent",          # organization
    "PERSON": "agent",       # person
    "NORP": "entity",        # nationality, religious, political group
    "PRODUCT": "object",     # product (vehicles, food, software)
    "EVENT": "process",      # event (wars, sports events)
    "WORK_OF_ART": "entity", # title of creative work
    "LAW": "concept",        # law or legal document
    "LANGUAGE": "concept",   # language
    "DATE": "state",         # date or period
    "TIME": "state",         # time
    "QUANTITY": "data",      # quantity (e.g., "5 pounds")
    "CARDINAL": "data",      # cardinal number
    "ORDINAL": "data",       # ordinal number ("first", "second")
    "MONEY": "data",         # monetary value
    "PERCENT": "data",       # percentage
}


class ProposalBuilder:
    """Builds structured proposals from conversation turns."""

    async def build(
        self,
        text: str,
        metadata: dict,
        *,
        correlation_id: str | None = None,
        llm_provider: str = "unknown",
        model: str = "unknown",
        confidence: float = 0.7,
    ) -> dict:
        """Build a structured proposal from text.

        Returns dict matching HermesProposalRequest schema.
        """
        proposal_id = str(uuid_mod.uuid4())
        now = datetime.now(UTC).isoformat()

        proposed_nodes = await self._extract_entities(text)
        document_embedding = await self._generate_document_embedding(text)

        return {
            "proposal_id": proposal_id,
            "correlation_id": correlation_id,
            "source_service": "hermes",
            "llm_provider": llm_provider,
            "model": model,
            "generated_at": now,
            "confidence": confidence,
            "raw_text": text,
            "proposed_nodes": proposed_nodes,
            "document_embedding": document_embedding,
            "metadata": metadata,
        }

    async def _extract_entities(self, text: str) -> list[dict]:
        """Extract named entities and generate per-entity embeddings."""
        try:
            nlp_result = await process_nlp(text, ["ner"])
            entities = nlp_result.get("entities", [])
        except Exception:
            logger.warning("NER extraction failed, returning empty entities")
            return []

        proposed_nodes = []
        for entity in entities:
            try:
                emb = await generate_embedding(entity["text"])
                ontology_type = SPACY_TO_ONTOLOGY.get(entity["label"], "entity")
                proposed_nodes.append({
                    "name": entity["text"],
                    "type": ontology_type,
                    "embedding": emb["embedding"],
                    "embedding_id": emb["embedding_id"],
                    "dimension": emb["dimension"],
                    "model": emb["model"],
                    "properties": {"start": entity["start"], "end": entity["end"]},
                })
            except Exception:
                logger.warning(f"Embedding failed for entity '{entity['text']}'")
        return proposed_nodes

    async def _generate_document_embedding(self, text: str) -> dict | None:
        """Generate embedding for the full text."""
        try:
            return await generate_embedding(text)
        except Exception:
            logger.warning("Document embedding generation failed")
            return None
```

### Step 4: Run test, verify pass, commit

Run: `cd hermes && poetry run pytest tests/unit/test_proposal_builder.py -v`

```bash
cd hermes && git add src/hermes/proposal_builder.py tests/unit/test_proposal_builder.py && git commit -m "$(cat <<'EOF'
feat: add ProposalBuilder — extracts entities and embeddings from text
EOF
)"
```

---

## Task 6: Sophia Proposal Processor

**Why:** `POST /ingest/hermes_proposal` currently logs and returns `stored_node_ids=[]` (TODO at `app.py:1321`). This task makes it actually process proposals — ingesting nodes into Neo4j, storing embeddings in Milvus, and returning relevant context.

**Repo:** sophia

**Files:**
- Create: `src/sophia/ingestion/proposal_processor.py`
- Modify: `src/sophia/api/models.py` — add `proposed_nodes`, `document_embedding`, `relevant_context` fields
- Modify: `src/sophia/api/app.py` — wire ProposalProcessor into endpoint
- Test: `tests/unit/test_proposal_processor.py` (new)

### Step 1: Write failing test

```python
# tests/unit/test_proposal_processor.py
"""Tests for ProposalProcessor — cognitive intake of Hermes proposals."""
import pytest
from unittest.mock import MagicMock


class TestProposalProcessor:

    def test_ingests_proposed_nodes(self):
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "new-uuid"
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process({
            "proposal_id": "p1",
            "proposed_nodes": [{
                "name": "Paris", "type": "GPE",
                "embedding": [0.1] * 384, "embedding_id": "emb-1",
                "dimension": 384, "model": "all-MiniLM-L6-v2",
                "properties": {},
            }],
            "document_embedding": {"embedding": [0.5] * 384, "embedding_id": "doc-1",
                                   "dimension": 384, "model": "all-MiniLM-L6-v2"},
            "raw_text": "Tell me about Paris",
            "source_service": "hermes",
            "confidence": 0.7,
            "metadata": {},
        })

        assert len(result["stored_node_ids"]) == 1
        mock_hcg.add_node.assert_called_once()
        mock_milvus.upsert_embedding.assert_called()

    def test_returns_relevant_context(self):
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.get_node.return_value = {
            "uuid": "existing-uuid", "name": "France",
            "type": "entity", "properties": {"capital": "Paris"},
        }
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = [
            {"uuid": "existing-uuid", "score": 0.15},
        ]

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process({
            "proposal_id": "p1",
            "proposed_nodes": [],
            "document_embedding": {"embedding": [0.5] * 384, "embedding_id": "doc-1",
                                   "dimension": 384, "model": "all-MiniLM-L6-v2"},
            "raw_text": "Tell me about France",
            "source_service": "hermes",
            "confidence": 0.7,
            "metadata": {},
        })

        assert len(result["relevant_context"]) >= 1
        assert result["relevant_context"][0]["name"] == "France"

    def test_skips_creation_when_similar_entity_exists(self):
        """Sophia should not create a duplicate when a similar node exists."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.get_node.return_value = {
            "uuid": "existing-paris", "name": "Paris",
            "type": "location", "properties": {},
        }
        mock_milvus = MagicMock()
        # Document-level search returns nothing, but entity-level returns a match
        mock_milvus.search_similar.side_effect = [
            [],  # Entity doc search
            [],  # Concept doc search
            [],  # State doc search
            [],  # Process doc search
            [],  # Edge doc search
            [{"uuid": "existing-paris", "score": 0.2}],  # entity match (below threshold)
        ]

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process({
            "proposal_id": "p1",
            "proposed_nodes": [{
                "name": "Paris", "type": "location",
                "embedding": [0.1] * 384, "embedding_id": "emb-1",
                "dimension": 384, "model": "all-MiniLM-L6-v2",
                "properties": {},
            }],
            "document_embedding": {"embedding": [0.5] * 384, "embedding_id": "doc-1",
                                   "dimension": 384, "model": "all-MiniLM-L6-v2"},
            "raw_text": "Tell me about Paris",
            "source_service": "hermes",
            "confidence": 0.7,
            "metadata": {},
        })

        # Node should NOT be created — existing match found
        assert result["stored_node_ids"] == []
        mock_hcg.add_node.assert_not_called()
        # Existing node should appear in context
        assert any(c["node_uuid"] == "existing-paris" for c in result["relevant_context"])

    def test_skips_empty_name_nodes(self):
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process({
            "proposal_id": "p1",
            "proposed_nodes": [{"name": "", "type": "X", "embedding": [0.1] * 384,
                                "embedding_id": "e1", "dimension": 384, "model": "m",
                                "properties": {}}],
            "document_embedding": None,
            "raw_text": "", "source_service": "hermes",
            "confidence": 0.7, "metadata": {},
        })

        assert result["stored_node_ids"] == []
        mock_hcg.add_node.assert_not_called()
```

### Step 2: Run test to verify it fails

Run: `cd sophia && poetry run pytest tests/unit/test_proposal_processor.py -v`
Expected: FAIL — module does not exist.

### Step 3: Implement ProposalProcessor

Create `src/sophia/ingestion/__init__.py` (empty) and `src/sophia/ingestion/proposal_processor.py`:

```python
"""Processes Hermes proposals — Sophia's cognitive intake pathway.

Sophia receives structured proposals from Hermes (entities, embeddings,
text metadata) and decides what to ingest into the graph. She also searches
for relevant existing context to return.

Sophia operates on embeddings, not text. Text properties exist on nodes
for Hermes's benefit when context is returned.

Key design decisions:
- Sophia has cognitive authority over the graph. She decides whether a
  proposed node is genuinely new or matches an existing one.
- Entity resolution uses embedding similarity: before creating a node,
  search for existing nodes with similar embeddings. If a match is found
  above the threshold, return it as context instead of creating a duplicate.
- Partial ingestion is acceptable. If 2 of 3 nodes succeed, those 2 should
  be in the graph. Graph consistency is an eventual property maintained by
  Sophia over time through pruning jobs, not a transactional guarantee on
  each request.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

# L2 distance threshold for entity matching. If an existing node's embedding
# is within this distance of a proposed node's embedding, Sophia considers
# them the same entity and returns the existing node as context instead of
# creating a duplicate.
#
# L2 distance of 0.0 = identical embeddings. Lower = more similar.
# 0.5 is conservative — tune based on observed deduplication quality.
# Too low: duplicates proliferate. Too high: distinct entities merge.
ENTITY_MATCH_THRESHOLD = 0.5


class ProposalProcessor:
    """Processes proposals from Hermes into graph knowledge.

    Sophia decides what enters the graph. For each proposed node, she:
    1. Searches for existing nodes with similar embeddings
    2. If a match is found within ENTITY_MATCH_THRESHOLD, skips creation
       and returns the existing node as context
    3. If no match, creates the node and stores its embedding

    Uses HCGMilvusSync from logos_hcg for embedding operations (not a
    custom embedding store — HCGMilvusSync already handles per-type
    collections, upsert, and now search_similar).
    """

    def __init__(self, hcg_client: Any, milvus_sync: Any) -> None:
        self._hcg = hcg_client
        self._milvus = milvus_sync

    def process(self, proposal: dict) -> dict:
        """Process a proposal: search for context, decide what to ingest.

        Returns dict with stored_node_ids and relevant_context.
        """
        stored_ids: list[str] = []
        relevant_context: list[dict] = []

        # 1. Search for relevant existing context using document embedding
        doc_emb = proposal.get("document_embedding")
        if doc_emb and doc_emb.get("embedding"):
            # Search across node type collections
            for node_type in ["Entity", "Concept", "State", "Process", "Edge"]:
                try:
                    matches = self._milvus.search_similar(
                        node_type=node_type,
                        query_embedding=doc_emb["embedding"],
                        top_k=5,
                    )
                    for match in matches:
                        node = self._hcg.get_node(match["uuid"])
                        if node:
                            relevant_context.append({
                                "node_uuid": match["uuid"],
                                "name": node.get("name", ""),
                                "type": node.get("type", ""),
                                "properties": node.get("properties", {}),
                                "score": match["score"],
                            })
                except Exception as e:
                    logger.debug(f"Search in {node_type} failed: {e}")

            # Sort by similarity score (lower = more similar for L2)
            relevant_context.sort(key=lambda x: x.get("score", float("inf")))
            relevant_context = relevant_context[:10]  # top 10

        # 2. Ingest proposed nodes — Sophia decides for each one
        for proposed in proposal.get("proposed_nodes", []):
            name = proposed.get("name", "").strip()
            if not name:
                continue

            node_type = proposed.get("type", "unknown")
            embedding = proposed.get("embedding")
            embedding_id = proposed.get("embedding_id")
            model = proposed.get("model", "unknown")

            # 2a. Search for existing node with similar embedding.
            # If a match is within threshold, this entity already exists
            # in the graph — return it as context, don't create a duplicate.
            if embedding:
                try:
                    existing = self._milvus.search_similar(
                        node_type="Entity",
                        query_embedding=embedding,
                        top_k=1,
                    )
                    if existing and existing[0]["score"] < ENTITY_MATCH_THRESHOLD:
                        # Existing node is similar enough — treat as same entity
                        existing_node = self._hcg.get_node(existing[0]["uuid"])
                        if existing_node:
                            logger.info(
                                f"Entity '{name}' matches existing node "
                                f"'{existing_node.get('name')}' "
                                f"(L2={existing[0]['score']:.3f}), skipping creation"
                            )
                            # Add to context if not already there
                            if not any(c["node_uuid"] == existing[0]["uuid"]
                                      for c in relevant_context):
                                relevant_context.append({
                                    "node_uuid": existing[0]["uuid"],
                                    "name": existing_node.get("name", ""),
                                    "type": existing_node.get("type", ""),
                                    "properties": existing_node.get("properties", {}),
                                    "score": existing[0]["score"],
                                })
                            continue  # skip creation
                except Exception as e:
                    logger.debug(f"Entity match search failed for '{name}': {e}")
                    # Fall through to creation — better to have a potential
                    # duplicate than to lose knowledge. Pruning handles it.

            # 2b. No match found — create the node
            try:
                node_uuid = self._hcg.add_node(
                    name=name,
                    node_type=node_type,
                    properties={
                        "source": proposal.get("source_service", "hermes"),
                        "derivation": "observed",
                        "confidence": proposal.get("confidence", 0.7),
                        "raw_text": proposal.get("raw_text", ""),
                        **proposed.get("properties", {}),
                    },
                )
                stored_ids.append(node_uuid)
            except Exception as e:
                # Partial ingestion is acceptable — continue with remaining
                # nodes. Pruning jobs handle inconsistencies after the fact.
                logger.error(f"Failed to create node '{name}': {e}")
                continue

            # 2c. Store embedding in Milvus
            if embedding and embedding_id:
                try:
                    self._milvus.upsert_embedding(
                        node_type="Entity",  # default collection for proposal nodes
                        uuid=node_uuid,
                        embedding=embedding,
                        model=model,
                    )
                except Exception as e:
                    # Node exists in Neo4j without embedding in Milvus.
                    # This is an inconsistent state but acceptable — pruning
                    # jobs will detect and repair embedding gaps.
                    logger.warning(f"Embedding storage failed for '{name}': {e}")

        return {
            "stored_node_ids": stored_ids,
            "relevant_context": relevant_context,
        }
```

### Step 4: Update API models

In `sophia/src/sophia/api/models.py`, add fields to `HermesProposalRequest` (after line 528):

```python
    proposed_nodes: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Structured entity proposals with embeddings",
    )
    document_embedding: Optional[Dict[str, Any]] = Field(
        default=None, description="Embedding of the full document text",
    )
```

Add to `HermesProposalResponse` (after line 543):

```python
    relevant_context: List[Dict[str, Any]] = Field(
        default_factory=list, description="Relevant graph context for LLM prompt",
    )
```

### Step 5: Wire ProposalProcessor into the endpoint

In `sophia/src/sophia/api/app.py`, replace the TODO block at line 1321 with actual processing. See the existing `ingest_hermes_proposal()` function at line 1285 — after the logging, add:

```python
    if _proposal_processor:
        result = _proposal_processor.process({
            "proposal_id": request.proposal_id,
            "proposed_nodes": request.proposed_nodes or [],
            "document_embedding": (request.document_embedding
                                   if hasattr(request, 'document_embedding') else None),
            "raw_text": request.raw_text,
            "source_service": request.source_service,
            "confidence": request.confidence,
            "metadata": request.metadata or {},
        })
        stored_ids = result["stored_node_ids"]
        context = result["relevant_context"]
    else:
        stored_ids = []
        context = []
```

Initialize `_proposal_processor` in the lifespan function, after `_hcg_client` and Milvus are set up.

### Step 6: Run tests, commit

Run: `cd sophia && poetry run pytest tests/ -v`

```bash
cd sophia && git add src/sophia/ingestion/ src/sophia/api/models.py src/sophia/api/app.py \
  tests/unit/test_proposal_processor.py && git commit -m "$(cat <<'EOF'
feat: implement proposal processing — ingest nodes, search embeddings, return context
EOF
)"
```

---

## Task 7: Context Injection and Loop Closing

**Why:** The `/llm` endpoint currently generates THEN forwards to Sophia. The flow must be reversed: extract → propose → get context → inject → generate. This closes the loop.

**Repo:** hermes

**Files:**
- Modify: `src/hermes/main.py` — restructure `llm_generate()`, add `_get_sophia_context()` and `_build_context_message()`
- Test: `tests/unit/test_context_injection.py` (new)

### Step 1: Write failing test

```python
# tests/unit/test_context_injection.py
"""Tests for context injection — the loop-closing flow."""
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
class TestContextInjection:

    async def test_build_context_message(self):
        from hermes.main import _build_context_message

        context = [
            {"name": "Paris", "type": "GPE", "properties": {}, "score": 0.1},
            {"name": "France", "type": "GPE", "properties": {}, "score": 0.2},
        ]
        msg = _build_context_message(context)
        assert msg["role"] == "system"
        assert "Paris" in msg["content"]
        assert "France" in msg["content"]

    async def test_build_context_message_empty(self):
        from hermes.main import _build_context_message
        assert _build_context_message([]) is None

    async def test_sophia_unavailable_returns_empty_context(self):
        from hermes.main import _get_sophia_context

        with patch("hermes.main._proposal_builder") as mock_builder, \
             patch("hermes.main.httpx") as mock_httpx:
            mock_builder.build = AsyncMock(return_value={
                "proposal_id": "p1", "proposed_nodes": [],
                "document_embedding": None, "source_service": "hermes",
                "metadata": {},
            })
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("Connection refused")
            mock_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(return_value=False)

            context = await _get_sophia_context("Hello", "req-1", {})

        assert context == []
```

### Step 2: Run test to verify it fails

Run: `cd hermes && poetry run pytest tests/unit/test_context_injection.py -v`
Expected: FAIL — functions don't exist.

### Step 3: Implement context functions and restructure LLM flow

In `src/hermes/main.py`:

**a) Add imports and builder instance** (near top):
```python
from hermes.proposal_builder import ProposalBuilder
_proposal_builder = ProposalBuilder()
```

**b) Add `_get_sophia_context()`** — replaces `_forward_llm_to_sophia()`:

```python
async def _get_sophia_context(text: str, request_id: str, metadata: dict) -> list[dict]:
    """Send proposal to Sophia, get relevant context back.

    Never raises — if Sophia is unavailable, returns empty list.
    LLM generation proceeds regardless.
    """
    sophia_host = get_env_value("SOPHIA_HOST", default="localhost") or "localhost"
    sophia_port = get_env_value("SOPHIA_PORT", default="47000") or "47000"
    sophia_token = get_env_value("SOPHIA_API_KEY") or get_env_value("SOPHIA_API_TOKEN")

    if not sophia_token:
        logger.error("SOPHIA_API_KEY/SOPHIA_API_TOKEN not configured — context disabled")
        return []

    try:
        proposal = await _proposal_builder.build(
            text=text, metadata=metadata, correlation_id=request_id,
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.post(
                f"http://{sophia_host}:{sophia_port}/ingest/hermes_proposal",
                json=proposal,
                headers={"Authorization": f"Bearer {sophia_token}"},
            )
            if response.status_code == 201:
                context = response.json().get("relevant_context", [])
                if not context:
                    logger.debug("Sophia returned empty context (no relevant knowledge)")
                return context
            logger.warning(f"Sophia returned {response.status_code}: {response.text[:200]}")
            return []
    except httpx.ConnectError as e:
        logger.warning(f"Cannot reach Sophia at {sophia_host}:{sophia_port}: {e}")
        return []
    except httpx.TimeoutException:
        logger.warning(f"Sophia request timed out ({sophia_host}:{sophia_port})")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during Sophia context retrieval: {e}", exc_info=True)
        return []
```

**c) Add `_build_context_message()`**:

```python
def _build_context_message(context: list[dict]) -> dict | None:
    """Translate Sophia's graph context into a system message for the LLM.

    Hermes does the language work — Sophia returns structured data,
    Hermes turns it into text the LLM can use.
    """
    if not context:
        return None

    lines = ["Relevant knowledge from memory:"]
    for item in context:
        name = item.get("name", "unknown")
        node_type = item.get("type", "")
        props = item.get("properties", {})
        desc = f"- {name}"
        if node_type:
            desc += f" ({node_type})"
        prop_str = ", ".join(
            f"{k}={v}" for k, v in props.items()
            if k not in ("source", "derivation", "confidence", "raw_text",
                         "created_at", "updated_at")
        )
        if prop_str:
            desc += f": {prop_str}"
        lines.append(desc)

    return {"role": "system", "content": "\n".join(lines)}
```

**d) Restructure `llm_generate()`** (line 663). The new flow:

1. Normalize messages
2. Get context from Sophia (proposal → search → context)
3. Inject context as system message
4. Generate LLM response

Remove the call to `_forward_llm_to_sophia()` at line 693. Delete `_forward_llm_to_sophia()` entirely — it's replaced by `_get_sophia_context()`.

### Step 4: Run tests

Run: `cd hermes && poetry run pytest tests/ -v`
Fix failures — existing tests for `/llm` may need updating since the flow changed.

### Step 5: Commit

```bash
cd hermes && git add src/hermes/main.py tests/unit/test_context_injection.py && git commit -m "$(cat <<'EOF'
feat: close the cognitive loop — proposal to Sophia, context into LLM prompt

Replaces _forward_llm_to_sophia() with _get_sophia_context().
New flow: extract entities → build proposal → send to Sophia →
get context back → inject into LLM prompt → generate response.
EOF
)"
```

---

## Integration Testing

After all tasks are complete, verify the end-to-end loop.

### Prerequisites

```bash
# Start infrastructure
cd logos/infra && docker compose -f docker-compose.hcg.dev.yml up -d

# Seed the graph (from logos repo)
cd logos && poetry run logos-seed-hcg --clear

# Start Sophia
cd sophia && poetry run uvicorn sophia.api.app:app --port 47000 &

# Start Hermes with Sophia config
cd hermes && SOPHIA_API_TOKEN=sophia_dev SOPHIA_HOST=localhost SOPHIA_PORT=47000 \
  poetry run uvicorn hermes.main:app --port 17000 &
```

### Smoke test

**Turn 1** — introduce knowledge:
```bash
curl -s -X POST http://localhost:17000/llm \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The Assembly Lab has a robot called LOGOS-01 with a Panda Arm."}' | jq .
```

**Turn 2** — verify context flows back:
```bash
curl -s -X POST http://localhost:17000/llm \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What robot is in the lab?"}' | jq .
```

The response to Turn 2 should reference LOGOS-01 and/or Panda Arm — knowledge from Turn 1 that flowed through the graph.

### Automated integration assertion script

The smoke test above is for manual exploration. The following script provides automated verification with specific assertions. Save as `tests/integration/test_cognitive_loop.sh` and run after both turns:

```bash
#!/usr/bin/env bash
# Automated integration test for the cognitive loop.
# Requires: Neo4j running, Sophia running, Hermes running.
set -euo pipefail

HERMES="http://localhost:17000"
NEO4J_URI="bolt://localhost:7687"
PASS=true

echo "=== Turn 1: Introduce knowledge ==="
RESP1=$(curl -sf -X POST "$HERMES/llm" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The Assembly Lab has a robot called LOGOS-01 with a Panda Arm."}')
echo "Turn 1 response received."

# Wait for async ingestion
sleep 2

echo "=== Verify graph state after Turn 1 ==="

# Check that Hermes-sourced nodes were created
NODE_COUNT=$(cypher-shell -u neo4j -p logosdev --format plain \
  "MATCH (n:Node) WHERE n.source = 'hermes' RETURN count(n) AS c" 2>/dev/null | tail -1)
if [ "$NODE_COUNT" -lt 1 ]; then
  echo "FAIL: Expected at least 1 Hermes-sourced node, got $NODE_COUNT"
  PASS=false
else
  echo "PASS: $NODE_COUNT Hermes-sourced nodes in graph"
fi

# Check that edge nodes use :FROM/:TO (no native knowledge relationships)
NATIVE_RELS=$(cypher-shell -u neo4j -p logosdev --format plain \
  "MATCH ()-[r]->() WHERE type(r) <> 'FROM' AND type(r) <> 'TO' RETURN count(r) AS c" 2>/dev/null | tail -1)
if [ "$NATIVE_RELS" -gt 0 ]; then
  echo "FAIL: Found $NATIVE_RELS native relationships (only :FROM/:TO allowed)"
  PASS=false
else
  echo "PASS: No native knowledge relationships — only :FROM/:TO"
fi

echo "=== Turn 2: Query for context ==="
RESP2=$(curl -sf -X POST "$HERMES/llm" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What robot is in the lab?"}')

# Check that the response references knowledge from Turn 1
if echo "$RESP2" | jq -e '.response' | grep -qi -e "LOGOS" -e "Panda" -e "robot"; then
  echo "PASS: Turn 2 response references Turn 1 knowledge"
else
  echo "FAIL: Turn 2 response does not reference Turn 1 knowledge"
  echo "Response: $(echo "$RESP2" | jq -r '.response' | head -3)"
  PASS=false
fi

echo ""
if [ "$PASS" = true ]; then
  echo "=== ALL ASSERTIONS PASSED — LOOP IS CLOSED ==="
else
  echo "=== SOME ASSERTIONS FAILED ==="
  exit 1
fi
```

### Acceptance criteria

The loop is closed when:

1. User sends a message (Turn 1)
2. Hermes builds a proposal with entities + embeddings
3. Sophia ingests nodes + edge nodes + embeddings
4. User sends another message (Turn 2)
5. Sophia finds Turn 1's knowledge via embedding similarity
6. Sophia returns it as context
7. Hermes injects context into the LLM prompt
8. LLM response references Turn 1's knowledge

Step 5 is the critical test — information flows back through the graph.
