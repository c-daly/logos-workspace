# Ontology Evolution — Graph Mutation Operations

**Date**: 2026-03-05
**Status**: Draft
**Parent Epic**: logos #499 (KG Maintenance & Self-Improving Ontology)
**Related Stories**: #503, #504, #505, #506, #507, #508

## Context

Epic #499 defines Sophia's KG maintenance capabilities — entity resolution,
type correction, ontology evolution, relationship inference. The stories
describe *what* should happen but not *how* the graph is actually mutated.

This document specifies:
1. The primitive operations that can be performed on any node
2. The detection system that identifies when operations are needed
3. The enactment logic that mutates the graph
4. The structural consequences that propagate from each mutation

### Key Principle: Uniform Node Model

The HCG encodes edges as first-class nodes (`type: "edge"`) with their own
UUIDs, embeddings, properties, and type assignments. **Any operation that can
happen to a regular node can happen to an edge node.** The same detection,
the same primitives, the same enactment logic. The only difference is which
analyzers are relevant and how structural consequences propagate.

---

## 1. Primitive Operations

Every graph mutation is a change to one or more **degrees of freedom**
on a node. These are the irreducible operations. All compound operations
(split, merge, reclassify, etc.) are composed from these.

### 1.1 Degrees of Freedom — All Nodes

Every node (Entity, Concept, State, Process, Edge) has:

| Degree of Freedom | Primitive Operations |
|-------------------|---------------------|
| **Existence** | Create node, Delete node |
| **Properties** | Update any property (name, confidence, timestamps, arbitrary key-value) |
| **Embedding** | Upsert or remove embedding in Milvus |
| **Type assignment** | Change IS_A edge (with centroid and member_count bookkeeping on both old and new type) |
| **Structural connections** | Add or remove edges connecting to/from this node |

### 1.2 Additional Degrees of Freedom — Edge Nodes

Edge nodes are first-class nodes with additional degrees of freedom
specific to their structural role:

| Degree of Freedom | Primitive Operations |
|-------------------|---------------------|
| **Source** | Change which node the edge originates from |
| **Target** | Change which node the edge points to |
| **Relation** | Change the semantic label (CAUSES, IS_A, HAS_STATE, etc.) |

### 1.3 Design Principle

These primitives are implemented **before** we know all the compound
operations we'll need. If we can create, delete, and modify any degree
of freedom on any node, then whatever patterns the detectors find — and
whatever topologies Hermes proposes — we can execute them. New compound
operations are new sequences of the same primitives.

---

## 2. Compound Operations

Split, Merge, Reclassify, and Confidence Update are **named patterns** —
recipes that describe which Creates, Updates, and Deletes happen and in
what order. The signal framework proposes compound operations; the
enactment layer decomposes them into sequenced primitives within a
transaction.

### 2.1 Reclassify

A node moves from one type assignment to another.

**Decomposition:**
1. **Update** — node's type assignment changes (IS_A edge adjusted)
2. **Update** — old type's centroid and member_count decremented
3. **Update** — new type's centroid and member_count incremented

### 2.2 Confidence Update

A node's confidence changes from reinforcement or contradiction.

**Decomposition:**
1. **Update** — node's confidence property adjusted
2. If confidence drops below threshold → may trigger Deprecate pattern
3. If competing nodes exist (e.g., two edges between same source/target
   with different relations): relative confidence determines canonical
   vs alternative

### 2.3 Split

A type's members are found to not form a coherent group.

The detector identifies clusters within a type's membership. It does
**not** determine the resulting graph topology — that is proposed by
Hermes and evaluated by Sophia. Possible outcomes include:

- **Refinement**: The original type survives as a parent, subtypes
  emerge beneath it. ("animal" gains "mammal" and "reptile" as children.)
- **Differentiation**: The original type was a bucket that caught two
  unrelated things. It gets deprecated and replaced by sibling types
  at the same level. No parent/child relationship is created.
- **Partial dissolution**: Some members belong to the original type,
  others form a new type. The original survives with reduced membership.
- **Other topologies**: The pipeline doesn't prescribe outcomes. Hermes
  proposes whatever structure the evidence supports.

**Decomposition:**
1. **Create** × N — new type Concept nodes with initial centroids
2. **Create** × N — IS_A or other structural edges as topology requires
3. **Reclassify** × M — member nodes reassigned to new types
   (each Reclassify is itself Update × 3, see above)
4. **Update** or **Delete** — original type either adjusts role
   (becomes supertype), reduces membership, or is deprecated

**Edge handling:**
- IS_A edges from members to the original type are replaced by edges
  to the appropriate new type
- All other edges touching the original type must be examined: which
  new type does the relationship actually apply to? The edge may be
  reassigned (Update), duplicated (Create) across new types, or
  removed (Delete)
- Edge nodes that are *members* of a relation type being split follow
  the same reclassification logic as any other node
- If the original type is deprecated: all its edges must be
  redistributed or deprecated

### 2.4 Merge

Two nodes collapse into one.

**Decomposition:**
1. **Update** — survivor's properties unioned with loser's properties
2. **Update** × N — all edges from/to the loser re-pointed to survivor
   - If a parallel edge already exists on the survivor: Confidence Update
     (reinforcement)
   - If no parallel edge exists: edge re-pointed (Update)
   - Duplicate edges (same source, target, relation) merged with combined
     confidence
3. **Update** — survivor's centroid/embedding recomputed from merged data
4. **Delete** — loser node removed

### 2.5 Deprecate

A node is retired from active reasoning.

**Decomposition:**
1. **Update** — node marked deprecated (property flag), or
   **Delete** — node removed entirely (policy decision, see open questions)
2. If type node: **Delete** centroid from Milvus TypeCentroid collection
3. Change published via pub/sub

---

## 3. Detection

The system needs to answer four questions:
1. Should a new type exist?
2. Should two types become one?
3. Should a new edge exist?
4. Should two edges become one?

Since edges are nodes, this reduces to two questions applied across node
kinds: **should something new exist?** and **should two things become
one?**

### 3.1 Detectors

Four detectors, each with a clear job:

- `SplitDetector.detect(type_uuid) → list[SplitCandidate]`
- `MergeDetector.detect() → list[MergeCandidate]`
- `TypeCreateDetector.detect() → list[CreateCandidate]`
- `EdgeCreateDetector.detect(node_uuids) → list[EdgeCandidate]`

Each detector is factory-initialized with a list of **analyzers** and
configuration parameters (thresholds, weights, scope). The detector
consults its analyzers, combines their evidence, and returns typed
candidates with confidence scores.

```python
class SplitDetector:
    """Detects when a type's members have diverged enough to warrant
    splitting into subtypes."""

    def __init__(self, analyzers: list[SplitAnalyzer], config: SplitConfig):
        self._analyzers = analyzers
        self._config = config

    def detect(self, type_uuid: str) -> list[SplitCandidate]:
        evidence = [a.analyze(type_uuid) for a in self._analyzers]
        return self._combine(evidence)
```

Same pattern for all four detectors. Adding a new signal means writing
one analyzer class and including it in the factory config.

### 3.2 Analyzer Interface

Each analyzer examines one aspect of the graph and returns evidence
toward the detector's question. Analyzers are typed per detector:

```python
class SplitAnalyzer(Protocol):
    """Produces evidence for or against splitting a type."""
    name: str
    def analyze(self, type_uuid: str) -> SplitEvidence: ...

class MergeAnalyzer(Protocol):
    """Produces evidence for or against merging two nodes."""
    name: str
    def analyze(self) -> list[MergeEvidence]: ...
```

Evidence objects carry a confidence score and signal-specific data
(cluster assignments, distance measurements, etc.) that the detector
uses to weigh and combine.

### 3.3 Analyzer Families

Analyzers fall into three families. The power is in their
**correlation** — geometric clustering confirmed by structural community
detection is much higher confidence than either alone.

**Geometric** (embedding space):
Operate on vector representations in Milvus. This is where novel
discovery happens — patterns that no single text revealed but emerge
from the accumulated embedding geometry.

**Structural** (graph topology):
Operate on Neo4j graph structure. Ground geometric discoveries — a
cluster that also forms a structural community is more likely real.

**Statistical** (traditional NLP/KG):
Operate on node properties and aggregate statistics. Cheap, fast
pre-filtering before expensive geometric or structural analysis.

### 3.4 Initial Analyzers

Start small. These are the highest-value analyzers to implement first:

**For SplitDetector:**
- Variance analyzer (geometric) — high internal variance suggests
  divergence. Already partially implemented in TypeEmergenceDetector.
- Community analyzer (structural) — dense subgraphs within a type
  suggest natural subdivisions.

**For MergeDetector:**
- Centroid proximity analyzer (geometric) — two type centroids close
  together suggest redundancy.
- Name similarity analyzer (statistical) — "INHABITS" and "LIVES_IN"
  are probably the same thing.
- Shared neighbor analyzer (structural) — two entities with heavily
  overlapping edge profiles.

**For TypeCreateDetector:**
- Orphan cluster analyzer (geometric) — coherent cluster of `object`
  nodes that don't fit any existing type centroid.
- Frequency analyzer (statistical) — `object` type has grown too large.

**For EdgeCreateDetector:**
- Cross-type proximity analyzer (geometric) — node embedding is close
  to nodes of a different type, suggesting an unstated relationship.
- Structural gap analyzer (structural) — two nodes share many indirect
  paths but no direct edge.

### 3.5 Analyzer Roadmap

Future analyzers to implement as patterns are observed in real data:

| Analyzer | Family | Detector(s) | Signal |
|----------|--------|-------------|--------|
| Elongation | Geometric | Split | Cluster stretched along an axis |
| Bridge detection | Geometric | Split | Thin neck between dense sub-regions |
| Centroid drift | Geometric | Split, Merge | Type centroid moving over time |
| Shell structure | Geometric | Split | Hollow cluster with sparse center |
| Vector field coherence | Geometric | Split (edges) | Edge vectors of same relation aren't parallel |
| Relation axis analysis | Geometric | Merge (edges) | Synonymous/antonymous relation types |
| Cross-type proximity | Geometric | Reclassify | Node closer to another type's centroid |
| Connectivity pattern | Structural | Split, Merge | Nodes with similar edge profiles |
| Degree anomaly | Structural | Reclassify | Unusual connectivity for its type |
| Source/target type pattern | Structural | Split (edges) | Edges always connecting same type pairs |
| Bidirectionality pattern | Structural | Split (edges) | Edges always one-way vs always both |
| Co-occurrence | Statistical | Merge, Edge create | Nodes always appearing in same proposals |
| Temporal clustering | Statistical | Type create | Nodes created in bursts |
| Property overlap | Statistical | Merge | Nodes with similar property sets |

This table is a roadmap, not a requirement. Analyzers are added
incrementally as the system encounters real data and we observe which
patterns matter.

---

## 4. Enactment

### 4.1 Sophia–Hermes Mutation Flow

Sophia doesn't write Cypher or directly execute graph mutations for
maintenance operations. She describes what she wants; Hermes produces
the mutations; Sophia evaluates and decides.

```
Detector → Candidate
  → Sophia builds structured mutation request
  → Hermes returns mutation proposal (structured JSON / Cypher)
  → Sophia evaluates: accept all, accept some, reject, or modify
  → Sophia executes accepted primitives via HCGClient
```

This mirrors the existing ingestion flow — Hermes proposes, Sophia
decides — but in the opposite direction. For ingestion, Hermes initiates
with extracted entities. For maintenance, Sophia initiates with detected
patterns.

### 4.2 Mutation Request Contract

Sophia sends Hermes a structured request describing the intended
operation and the evidence that triggered it:

```json
{
  "operation": "split",
  "subject_uuid": "type_uuid_123",
  "clusters": [
    {"member_uuids": ["a", "b", "c"], "centroid": [0.1, ...]},
    {"member_uuids": ["d", "e", "f"], "centroid": [0.7, ...]}
  ],
  "evidence_summary": "variance 0.82, k-means produced two clusters with internal variance 0.15 and 0.18"
}
```

Sophia tells Hermes **what** she wants to happen. Hermes decides **how**
to express it — naming the new types, generating the specific node
properties, producing the mutation steps.

### 4.3 Mutation Proposal Response

Hermes returns a structured proposal that Sophia can evaluate:

```json
{
  "mutations": [
    {"op": "create", "node": {"type": "concept", "name": "vehicle", "properties": {...}}},
    {"op": "create", "node": {"type": "concept", "name": "furniture", "properties": {...}}},
    {"op": "update", "uuid": "a", "changes": {"type_assignment": "vehicle"}},
    {"op": "update", "uuid": "d", "changes": {"type_assignment": "furniture"}},
    {"op": "create", "edge": {"source": "vehicle", "target": "type_uuid_123", "relation": "IS_A"}},
    ...
  ]
}
```

Each mutation maps directly to a primitive (Create, Update, Delete).
Sophia can:
- **Accept all** — execute the full proposal
- **Accept partially** — execute some mutations, reject others
- **Reject** — discard the proposal entirely
- **Modify** — adjust parameters before executing (e.g., accept the
  split but change which members go where)

### 4.4 Execution

Accepted mutations are executed through existing HCGClient methods
(`add_node`, `update_node`, `add_edge`) and HCGMilvusSync for embedding
operations. The execution happens within Sophia — she owns the database
connections.

### 4.5 Pub/Sub Notification

Every executed mutation publishes an ontology change event so that:
- Hermes updates its cached type/edge-type lists (#501)
- The maintenance scheduler can react to cascading changes
- Downstream consumers (Apollo, logging) stay informed

### 4.6 Idempotency and Safety

- All operations use MERGE-style Cypher (existing HCGClient pattern)
- Each proposal carries its evidence chain for auditability
- Rate limiting prevents runaway cascades (no splitting a type created
  in the last N minutes)
- Sophia's evaluation step is the safety gate — no mutation happens
  without her approval

---

## 5. Consequence Propagation

When any degree of freedom changes on a node, connected nodes may need
re-evaluation. This is not special-cased per node type — it follows
from a general principle:

**Any mutation to a node requires examining all edges connected to that
node to determine if they are still valid.**

### 5.1 Propagation Rules

| Change | What to re-evaluate |
|--------|---------------------|
| **Node deleted** | All edges to/from this node must be re-pointed, redistributed, or deleted. If the node was a type: all members' IS_A edges are orphaned. |
| **Node type assignment changed** | Edges may no longer make semantic sense in the new type context. Each edge's relation should be checked against the new type. |
| **Node properties changed** | If name/identity changed significantly, edges based on the old identity may need review. Confidence changes may affect connected edges' confidence. |
| **Node embedding changed** | Centroid of the node's type may drift. Cross-type proximity may change — node might now be closer to a different type. |
| **Edge source/target changed** | The nodes that gained or lost a connection may need structural re-evaluation (new neighbor patterns, changed connectivity). |
| **Edge relation changed** | The semantic meaning of the connection between source and target has changed. Source and target nodes may need re-evaluation in light of the new relationship. |
| **Edge deleted** | A gap in the graph may trigger relationship inference — was this edge load-bearing? Do the source and target still need to be connected? |

### 5.2 Cascade Control

A mutation that triggers re-evaluation may itself produce new mutations,
which trigger further re-evaluation. To prevent runaway cascades:

- **Depth limit**: configurable maximum cascade depth before pausing
- **Rate limiting**: don't re-evaluate nodes that were just mutated
- **Batch boundaries**: collect all triggered re-evaluations from one
  mutation, execute them as a batch, then evaluate the next round
- **Confidence decay**: each cascade step reduces the confidence of
  triggered proposals, so deeper cascades require stronger evidence

---

## 6. Integration with Epic #499

### Current Status

| # | Story | Status |
|---|-------|--------|
| #500 | Centralized Redis & Pub/Sub Infrastructure | DONE |
| #502 | Hermes NER Quality Improvements | DONE |
| #508 | Maintenance Scheduler | DONE (metrics/observability remaining) |
| #501 | Ontology Pub/Sub Distribution | TODO |
| #503 | Entity Resolution — Alias Detection and Merging | TODO |
| #504 | Type Correction — Centroid-Based Reclassification | TODO |
| #505 | Ontology Evolution — Emergent Type Discovery | TODO |
| #506 | Relationship Inference — Taxonomic Scaffolding | TODO |
| #507 | Competing Edges & Confidence Model | TODO |

### New Story Required

**Mutation Primitives, Pipeline & Propagation** — the shared foundation
that all compound operations build on:

- Primitive operations on all degrees of freedom (section 1)
- Unified Sophia→Hermes mutation request/response contract (section 4)
- Detector framework with pluggable analyzers (section 3)
- Consequence propagation and cascade control (section 5)

This story depends on #500 and #508. It blocks #501, #503–#507.

### Revised Execution Order

| Order | Story | Notes |
|-------|-------|-------|
| 1 | #500 — Redis Infrastructure | DONE |
| 2 | #502 — Hermes NER Improvements | DONE |
| 3 | #508 — Maintenance Scheduler | DONE (close after metrics) |
| 4 | **NEW — Mutation Primitives, Pipeline & Propagation** | Foundation for everything below |
| 5 | #507 — Confidence Model | Confidence is a degree of freedom the primitives need. Moved up. |
| 6 | #501 — Ontology Pub/Sub (expanded) | Event schema expanded to cover all primitive operations |
| 7 | #504 — Type Correction | Simplest compound operation — reclassify built on primitives |
| 8 | #503 — Entity Resolution | Merge built on primitives |
| 9 | #505 — Ontology Evolution | Split/create built on primitives. Add edge-type evolution. Remove parent/child topology assumption. |
| 10 | #506 — Relationship Inference | Edge create built on primitives |

### Updates to Existing Stories

- **#501**: Expand event schema to cover all primitive operations
  (not just type added/removed)
- **#505**: Add edge-type evolution (relation types can split/merge).
  Remove assumption that splits are always parent/child — topology is
  proposed by Hermes, evaluated by Sophia.
- **#508**: Check off completed tasks, keep metrics/observability as
  remaining work.
- **#503–#506**: Each removes its bespoke Hermes endpoint and mutation
  logic in favor of the unified contract from the new foundation story.

## 7. Open Questions

1. **Confidence model**: What's the initial confidence for a newly created
   node? How does reinforcement accumulate? Linear, logarithmic, Bayesian?
2. **Cascade depth**: If reclassifying a node triggers a structural change
   that triggers another signal — how deep do we let cascades go before
   pausing?
3. **Human-in-the-loop**: Should high-impact operations (e.g., merging two
   large types) require human approval? If so, how is that surfaced?
4. **Hermes naming failures**: What happens if Hermes can't name a cluster?
   Temporary placeholder? Retry? Leave as subtype of parent?
5. **Edge vector field analysis**: How exactly do we compute "parallel" vs
   "orthogonal" for relation types? Cosine similarity of mean edge vectors?
   PCA on the edge embedding set?
6. **Reserved types**: Which types are immune to evolution? Currently
   prefixed `reserved_*` — is that sufficient?
7. **Delete vs deprecate**: Should Delete mean hard removal or soft
   deprecation? Provenance argues for soft, but stale deprecated nodes
   could accumulate. Policy needed.
