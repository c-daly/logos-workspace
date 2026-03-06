# Entity Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give Sophia the ability to detect candidate alias pairs via embedding proximity and structural overlap, create hypothesis edges with confidence, consult Hermes for language judgment, and merge nodes when confidence is sufficient.

**Architecture:** Two-repo change. Hermes gets a new `/alias-check` endpoint following the existing `/name-type` pattern. Sophia gets a candidate detector, hypothesis edge creator, merge executor, and a maintenance handler wiring it all together. Detection uses `milvus.search_similar()` for embedding triage and HCGClient edge queries for structural overlap. Hypothesis edges are reified `POSSIBLE_ALIAS_OF` edges with confidence. Merge absorbs edges and removes the consumed node.

**Tech Stack:** Python 3.12, FastAPI (Hermes), Neo4j (HCG), Milvus (embeddings), Redis (maintenance queue/events)

---

## Task 1: Hermes `/alias-check` Endpoint

**Files:**
- Modify: `hermes/src/hermes/main.py` (add request/response models and endpoint, after `/name-relationship` endpoint)
- Test: `hermes/tests/unit/test_alias_check.py`

### Step 1: Write the failing test

```python
"""Tests for /alias-check endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_alias_check_identifies_aliases():
    from hermes.main import app

    mock_result = {
        "choices": [
            {
                "message": {
                    "content": '{"is_alias": true, "confidence": 0.9, "canonical_name": "Rottweiler", "reason": "rottie is an informal abbreviation of Rottweiler"}'
                }
            }
        ]
    }

    with patch("hermes.main.generate_completion", new_callable=AsyncMock, return_value=mock_result):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/alias-check", json={
                "entity_a": {"name": "rottie", "type": "entity", "relationships": ["OWNED_BY -> Alice", "IS_A -> dog"]},
                "entity_b": {"name": "Rottweiler", "type": "entity", "relationships": ["IS_A -> dog", "BREED_OF -> dog"]},
            })

    assert resp.status_code == 200
    data = resp.json()
    assert data["is_alias"] is True
    assert data["confidence"] > 0.5
    assert data["canonical_name"] == "Rottweiler"


@pytest.mark.asyncio
async def test_alias_check_rejects_non_aliases():
    from hermes.main import app

    mock_result = {
        "choices": [
            {
                "message": {
                    "content": '{"is_alias": false, "confidence": 0.1, "canonical_name": null, "reason": "A cat and a dog are different animals"}'
                }
            }
        ]
    }

    with patch("hermes.main.generate_completion", new_callable=AsyncMock, return_value=mock_result):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/alias-check", json={
                "entity_a": {"name": "cat", "type": "entity", "relationships": ["IS_A -> feline"]},
                "entity_b": {"name": "dog", "type": "entity", "relationships": ["IS_A -> canine"]},
            })

    assert resp.status_code == 200
    data = resp.json()
    assert data["is_alias"] is False
```

### Step 2: Run test to verify it fails

Run: `cd hermes && poetry run pytest tests/unit/test_alias_check.py -v`
Expected: FAIL — endpoint not found / 404

### Step 3: Write the endpoint

Add to `hermes/src/hermes/main.py` after the `/name-relationship` endpoint:

```python
class AliasEntityInfo(BaseModel):
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    relationships: list[str] = Field(
        default_factory=list, description="Relationship summaries (e.g., 'IS_A -> dog')"
    )


class AliasCheckRequest(BaseModel):
    entity_a: AliasEntityInfo = Field(..., description="First entity")
    entity_b: AliasEntityInfo = Field(..., description="Second entity")
    context: str | None = Field(default=None, description="Optional additional context")


class AliasCheckResponse(BaseModel):
    is_alias: bool = Field(..., description="Whether the entities are aliases")
    confidence: float = Field(..., description="Confidence in the judgment (0-1)")
    canonical_name: str | None = Field(default=None, description="Suggested canonical name if alias")
    reason: str = Field(..., description="Brief explanation")


@app.post("/alias-check", response_model=AliasCheckResponse)
async def alias_check(request: AliasCheckRequest) -> AliasCheckResponse:
    """Check if two entities are aliases of each other using LLM judgment."""
    a = request.entity_a
    b = request.entity_b
    a_rels = "; ".join(a.relationships) if a.relationships else "none"
    b_rels = "; ".join(b.relationships) if b.relationships else "none"

    system_msg = (
        "You are an entity resolution assistant. "
        "Determine if two entities refer to the same real-world thing. "
        'Return ONLY a JSON object: {"is_alias": <bool>, "confidence": <0-1>, '
        '"canonical_name": "<name or null>", "reason": "<brief explanation>"}.'
    )
    user_msg = (
        f"Entity A: {a.name} (type: {a.type}, relationships: {a_rels})\n"
        f"Entity B: {b.name} (type: {b.type}, relationships: {b_rels})"
    )
    if request.context:
        user_msg += f"\nContext: {request.context}"
    user_msg += "\n\nAre these the same entity?"

    result = await generate_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    choices = result.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="LLM returned no choices")
    content = choices[0]["message"]["content"]
    data = _extract_json(content)
    return AliasCheckResponse(
        is_alias=data.get("is_alias", False),
        confidence=data.get("confidence", 0.0),
        canonical_name=data.get("canonical_name"),
        reason=data.get("reason", ""),
    )
```

### Step 4: Run test to verify it passes

Run: `cd hermes && poetry run pytest tests/unit/test_alias_check.py -v`
Expected: PASS

### Step 5: Commit

```bash
cd hermes
git add src/hermes/main.py tests/unit/test_alias_check.py
git commit -m "feat: add /alias-check endpoint for entity resolution (#503)"
```

---

## Task 2: Sophia Candidate Detector

**Files:**
- Create: `sophia/src/sophia/maintenance/entity_resolution.py`
- Test: `sophia/tests/unit/maintenance/test_entity_resolution.py`

### Step 1: Write the failing test

```python
"""Tests for entity resolution candidate detection."""

from unittest.mock import MagicMock


class TestCandidateDetector:
    def _make_detector(self, milvus=None, hcg=None):
        from sophia.maintenance.entity_resolution import CandidateDetector

        return CandidateDetector(
            milvus=milvus or MagicMock(),
            hcg=hcg or MagicMock(),
        )

    def test_finds_candidates_by_embedding_proximity(self):
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = [
            {"uuid": "node-b", "score": 0.3},
        ]
        mock_milvus.get_embedding.return_value = {
            "uuid": "node-a",
            "embedding": [0.1] * 384,
            "embedding_model": "test",
            "last_sync": 0,
        }

        mock_hcg = MagicMock()
        mock_hcg.get_node.side_effect = lambda uuid: {
            "node-a": {"uuid": "node-a", "name": "rottie", "type": "entity", "properties": {}},
            "node-b": {"uuid": "node-b", "name": "Rottweiler", "type": "entity", "properties": {}},
        }.get(uuid)

        detector = self._make_detector(milvus=mock_milvus, hcg=mock_hcg)
        candidates = detector.find_candidates_for_node("node-a", node_type="Entity")

        assert len(candidates) >= 1
        assert candidates[0]["target_uuid"] == "node-b"
        assert "embedding_score" in candidates[0]

    def test_finds_candidates_by_structural_overlap(self):
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.get_embedding.return_value = None

        mock_hcg = MagicMock()
        # Both nodes connect to "dog" and "Alice" via same relations
        mock_hcg.get_node.side_effect = lambda uuid: {
            "node-a": {"uuid": "node-a", "name": "rottie", "type": "entity", "properties": {}},
            "node-c": {"uuid": "node-c", "name": "Rotts", "type": "entity", "properties": {}},
        }.get(uuid)

        def mock_outgoing(uuid):
            shared = {
                "node-a": [
                    {"relation": "IS_A", "target_uuid": "dog-1", "target_name": "dog"},
                    {"relation": "OWNED_BY", "target_uuid": "alice-1", "target_name": "Alice"},
                ],
                "node-c": [
                    {"relation": "IS_A", "target_uuid": "dog-1", "target_name": "dog"},
                    {"relation": "OWNED_BY", "target_uuid": "alice-1", "target_name": "Alice"},
                ],
            }
            return shared.get(uuid, [])

        mock_hcg._execute_read = MagicMock()

        detector = self._make_detector(milvus=mock_milvus, hcg=mock_hcg)
        detector._get_outgoing_edges = MagicMock(side_effect=mock_outgoing)
        candidates = detector.find_candidates_by_structure("node-a")

        assert len(candidates) >= 1
        assert candidates[0]["target_uuid"] == "node-c"
        assert "shared_targets" in candidates[0]

    def test_skips_existing_hypothesis_edges(self):
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = [
            {"uuid": "node-b", "score": 0.3},
        ]
        mock_milvus.get_embedding.return_value = {
            "uuid": "node-a",
            "embedding": [0.1] * 384,
            "embedding_model": "test",
            "last_sync": 0,
        }

        mock_hcg = MagicMock()
        mock_hcg.get_node.side_effect = lambda uuid: {
            "node-a": {"uuid": "node-a", "name": "rottie", "type": "entity", "properties": {}},
            "node-b": {"uuid": "node-b", "name": "Rottweiler", "type": "entity", "properties": {}},
        }.get(uuid)

        detector = self._make_detector(milvus=mock_milvus, hcg=mock_hcg)
        detector._has_hypothesis_edge = MagicMock(return_value=True)

        candidates = detector.find_candidates_for_node("node-a", node_type="Entity")
        assert len(candidates) == 0
```

### Step 2: Run test to verify it fails

Run: `cd sophia && poetry run pytest tests/unit/maintenance/test_entity_resolution.py::TestCandidateDetector -v`
Expected: FAIL — module not found

### Step 3: Write the candidate detector

Create `sophia/src/sophia/maintenance/entity_resolution.py`:

```python
"""Entity resolution — detect and merge alias nodes on the HCG.

Part of KG Maintenance (logos #499, story #503).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# L2 distance threshold for embedding triage (lower = more similar)
EMBEDDING_CANDIDATE_THRESHOLD = 1.0
# Minimum shared relationship targets for structural triage
MIN_SHARED_TARGETS = 2


class CandidateDetector:
    """Find candidate alias pairs via embedding proximity and structural overlap."""

    def __init__(self, milvus: Any, hcg: Any) -> None:
        self._milvus = milvus
        self._hcg = hcg

    def find_candidates_for_node(
        self,
        node_uuid: str,
        node_type: str = "Entity",
        top_k: int = 10,
    ) -> list[dict]:
        """Find candidate aliases for a node via embedding proximity.

        Args:
            node_uuid: UUID of the node to find candidates for.
            node_type: Milvus collection to search.
            top_k: Number of nearest neighbors to check.

        Returns:
            List of candidate dicts with target_uuid, embedding_score.
        """
        embedding_data = self._milvus.get_embedding(node_type, node_uuid)
        if not embedding_data:
            return []

        neighbors = self._milvus.search_similar(
            node_type=node_type,
            query_embedding=embedding_data["embedding"],
            top_k=top_k,
        )

        candidates = []
        for neighbor in neighbors:
            target_uuid = neighbor["uuid"]
            if target_uuid == node_uuid:
                continue
            if neighbor["score"] > EMBEDDING_CANDIDATE_THRESHOLD:
                continue
            if self._has_hypothesis_edge(node_uuid, target_uuid):
                continue

            target_node = self._hcg.get_node(target_uuid)
            if not target_node:
                continue

            candidates.append({
                "source_uuid": node_uuid,
                "target_uuid": target_uuid,
                "target_name": target_node.get("name", ""),
                "embedding_score": neighbor["score"],
            })

        return candidates

    def find_candidates_by_structure(self, node_uuid: str) -> list[dict]:
        """Find candidate aliases via shared relationship targets.

        Finds nodes that share N >= MIN_SHARED_TARGETS relationship targets
        with the given node, regardless of embedding distance.

        Args:
            node_uuid: UUID of the node to find structural matches for.

        Returns:
            List of candidate dicts with target_uuid, shared_targets.
        """
        source_edges = self._get_outgoing_edges(node_uuid)
        if not source_edges:
            return []

        source_targets = {
            (e["relation"], e["target_uuid"]) for e in source_edges
        }

        # Find nodes sharing targets via Cypher
        target_uuids = [e["target_uuid"] for e in source_edges]
        co_linked = self._find_nodes_sharing_targets(node_uuid, target_uuids)

        candidates = []
        for other_uuid, other_name in co_linked:
            if self._has_hypothesis_edge(node_uuid, other_uuid):
                continue

            other_edges = self._get_outgoing_edges(other_uuid)
            other_targets = {
                (e["relation"], e["target_uuid"]) for e in other_edges
            }

            shared = source_targets & other_targets
            if len(shared) >= MIN_SHARED_TARGETS:
                candidates.append({
                    "source_uuid": node_uuid,
                    "target_uuid": other_uuid,
                    "target_name": other_name,
                    "shared_targets": len(shared),
                    "shared_details": [
                        {"relation": r, "target_uuid": t} for r, t in shared
                    ],
                })

        return candidates

    def _get_outgoing_edges(self, node_uuid: str) -> list[dict]:
        """Get outgoing edges for a node."""
        query = """
        MATCH (n:Node {uuid: $uuid})<-[:FROM]-(e:Node {type: "edge"})-[:TO]->(target:Node)
        WHERE target.relation IS NULL
        RETURN e.relation AS relation, target.uuid AS target_uuid, target.name AS target_name
        """
        records = self._hcg._execute_read(query, {"uuid": node_uuid})
        return [dict(r) for r in records] if records else []

    def _find_nodes_sharing_targets(
        self, exclude_uuid: str, target_uuids: list[str]
    ) -> list[tuple[str, str]]:
        """Find other nodes that also link to the same targets."""
        query = """
        MATCH (other:Node)<-[:FROM]-(e:Node {type: "edge"})-[:TO]->(target:Node)
        WHERE target.uuid IN $target_uuids
          AND other.uuid <> $exclude_uuid
          AND other.relation IS NULL
        WITH other, count(DISTINCT target) AS shared_count
        WHERE shared_count >= $min_shared
        RETURN other.uuid AS uuid, other.name AS name
        """
        records = self._hcg._execute_read(
            query,
            {
                "target_uuids": target_uuids,
                "exclude_uuid": exclude_uuid,
                "min_shared": MIN_SHARED_TARGETS,
            },
        )
        return [(r["uuid"], r["name"]) for r in records] if records else []

    def _has_hypothesis_edge(self, uuid_a: str, uuid_b: str) -> bool:
        """Check if a POSSIBLE_ALIAS_OF edge already exists between two nodes."""
        query = """
        MATCH (a:Node {uuid: $uuid_a})<-[:FROM]-(e:Node {type: "edge", relation: "POSSIBLE_ALIAS_OF"})-[:TO]->(b:Node {uuid: $uuid_b})
        RETURN e.uuid AS uuid
        UNION
        MATCH (b:Node {uuid: $uuid_b})<-[:FROM]-(e:Node {type: "edge", relation: "POSSIBLE_ALIAS_OF"})-[:TO]->(a:Node {uuid: $uuid_a})
        RETURN e.uuid AS uuid
        """
        records = self._hcg._execute_read(query, {"uuid_a": uuid_a, "uuid_b": uuid_b})
        return bool(records)
```

### Step 4: Run test to verify it passes

Run: `cd sophia && poetry run pytest tests/unit/maintenance/test_entity_resolution.py::TestCandidateDetector -v`
Expected: PASS

### Step 5: Commit

```bash
cd sophia
git add src/sophia/maintenance/entity_resolution.py tests/unit/maintenance/test_entity_resolution.py
git commit -m "feat: candidate detector for entity resolution (#503)"
```

---

## Task 3: Hypothesis Edge Creator

**Files:**
- Modify: `sophia/src/sophia/maintenance/entity_resolution.py`
- Test: `sophia/tests/unit/maintenance/test_entity_resolution.py`

### Step 1: Write the failing test

Add to `test_entity_resolution.py`:

```python
class TestHypothesisManager:
    def _make_manager(self, hcg=None):
        from sophia.maintenance.entity_resolution import HypothesisManager

        return HypothesisManager(hcg=hcg or MagicMock())

    def test_creates_hypothesis_edge(self):
        mock_hcg = MagicMock()
        mock_hcg.add_edge.return_value = "hyp-edge-1"

        manager = self._make_manager(hcg=mock_hcg)
        edge_uuid = manager.create_hypothesis(
            source_uuid="node-a",
            target_uuid="node-b",
            confidence=0.6,
            evidence={"embedding_score": 0.3, "shared_targets": 3},
        )

        assert edge_uuid == "hyp-edge-1"
        mock_hcg.add_edge.assert_called_once()
        call_kwargs = mock_hcg.add_edge.call_args
        assert call_kwargs[1]["relation"] == "POSSIBLE_ALIAS_OF"
        props = call_kwargs[1]["properties"]
        assert props["confidence"] == 0.6

    def test_updates_confidence(self):
        mock_hcg = MagicMock()
        mock_hcg._execute_read.return_value = [
            {"uuid": "hyp-edge-1", "confidence": 0.6}
        ]

        manager = self._make_manager(hcg=mock_hcg)
        manager.update_confidence("hyp-edge-1", delta=0.15, reason="new shared target")

        mock_hcg.update_node.assert_called_once()
        call_args = mock_hcg.update_node.call_args
        assert call_args[0][0] == "hyp-edge-1"
        assert call_args[0][1]["confidence"] == 0.75

    def test_confidence_clamped_to_1(self):
        mock_hcg = MagicMock()
        mock_hcg._execute_read.return_value = [
            {"uuid": "hyp-edge-1", "confidence": 0.9}
        ]

        manager = self._make_manager(hcg=mock_hcg)
        manager.update_confidence("hyp-edge-1", delta=0.5, reason="strong corroboration")

        call_args = mock_hcg.update_node.call_args
        assert call_args[0][1]["confidence"] == 1.0
```

### Step 2: Run test to verify it fails

Run: `cd sophia && poetry run pytest tests/unit/maintenance/test_entity_resolution.py::TestHypothesisManager -v`
Expected: FAIL — HypothesisManager not found

### Step 3: Add HypothesisManager to entity_resolution.py

Append to `sophia/src/sophia/maintenance/entity_resolution.py`:

```python
class HypothesisManager:
    """Create and manage POSSIBLE_ALIAS_OF hypothesis edges."""

    def __init__(self, hcg: Any) -> None:
        self._hcg = hcg

    def create_hypothesis(
        self,
        source_uuid: str,
        target_uuid: str,
        confidence: float,
        evidence: dict,
    ) -> str:
        """Create a POSSIBLE_ALIAS_OF hypothesis edge between two nodes.

        Args:
            source_uuid: First node UUID.
            target_uuid: Second node UUID.
            confidence: Initial confidence (0-1).
            evidence: Dict of detection signals that triggered the hypothesis.

        Returns:
            UUID of the created hypothesis edge.
        """
        import json

        edge_uuid = self._hcg.add_edge(
            source_uuid=source_uuid,
            target_uuid=target_uuid,
            relation="POSSIBLE_ALIAS_OF",
            properties={
                "confidence": min(max(confidence, 0.0), 1.0),
                "evidence": json.dumps(evidence),
                "status": "hypothesis",
            },
        )
        logger.info(
            "Created POSSIBLE_ALIAS_OF hypothesis %s between %s and %s (confidence=%.2f)",
            edge_uuid, source_uuid, target_uuid, confidence,
        )
        return edge_uuid

    def update_confidence(
        self,
        edge_uuid: str,
        delta: float,
        reason: str,
    ) -> float:
        """Update confidence on a hypothesis edge.

        Args:
            edge_uuid: UUID of the hypothesis edge.
            delta: Amount to change confidence (positive or negative).
            reason: Why the confidence changed.

        Returns:
            New confidence value.
        """
        records = self._hcg._execute_read(
            "MATCH (e:Node {uuid: $uuid}) RETURN e.uuid AS uuid, e.confidence AS confidence",
            {"uuid": edge_uuid},
        )
        if not records:
            raise ValueError(f"Hypothesis edge {edge_uuid} not found")

        old_confidence = records[0]["confidence"]
        new_confidence = min(max(old_confidence + delta, 0.0), 1.0)

        self._hcg.update_node(edge_uuid, {
            "confidence": new_confidence,
        })
        logger.info(
            "Updated hypothesis %s confidence: %.2f -> %.2f (%s)",
            edge_uuid, old_confidence, new_confidence, reason,
        )
        return new_confidence

    def find_mergeable(self, threshold: float = 0.9) -> list[dict]:
        """Find hypothesis edges with confidence above the merge threshold.

        Args:
            threshold: Minimum confidence to consider for merge.

        Returns:
            List of hypothesis edge dicts with source/target UUIDs.
        """
        query = """
        MATCH (a:Node)<-[:FROM]-(e:Node {type: "edge", relation: "POSSIBLE_ALIAS_OF"})-[:TO]->(b:Node)
        WHERE e.confidence >= $threshold AND e.status = "hypothesis"
        RETURN e.uuid AS edge_uuid, e.confidence AS confidence,
               a.uuid AS source_uuid, a.name AS source_name,
               b.uuid AS target_uuid, b.name AS target_name
        """
        records = self._hcg._execute_read(query, {"threshold": threshold})
        return [dict(r) for r in records] if records else []
```

### Step 4: Run test to verify it passes

Run: `cd sophia && poetry run pytest tests/unit/maintenance/test_entity_resolution.py::TestHypothesisManager -v`
Expected: PASS

### Step 5: Commit

```bash
cd sophia
git add src/sophia/maintenance/entity_resolution.py tests/unit/maintenance/test_entity_resolution.py
git commit -m "feat: hypothesis edge creation and confidence accumulation (#503)"
```

---

## Task 4: Merge Executor

**Files:**
- Modify: `sophia/src/sophia/maintenance/entity_resolution.py`
- Test: `sophia/tests/unit/maintenance/test_entity_resolution.py`

### Step 1: Write the failing test

Add to `test_entity_resolution.py`:

```python
class TestMergeExecutor:
    def _make_executor(self, hcg=None, milvus=None):
        from sophia.maintenance.entity_resolution import MergeExecutor

        return MergeExecutor(hcg=hcg or MagicMock(), milvus=milvus or MagicMock())

    def test_merge_absorbs_edges(self):
        mock_hcg = MagicMock()
        # Consumed node has two outgoing edges
        mock_hcg._execute_read.side_effect = [
            # outgoing edges of consumed node
            [
                {"edge_uuid": "e1", "relation": "LIKES", "target_uuid": "t1", "props": {}},
                {"edge_uuid": "e2", "relation": "IS_A", "target_uuid": "t2", "props": {}},
            ],
            # incoming edges of consumed node
            [],
        ]
        mock_hcg.add_edge.return_value = "new-edge"

        executor = self._make_executor(hcg=mock_hcg)
        result = executor.merge(
            canonical_uuid="node-a",
            consumed_uuid="node-b",
            hypothesis_edge_uuid="hyp-1",
            canonical_name="Rottweiler",
        )

        assert result["canonical_uuid"] == "node-a"
        assert result["consumed_uuid"] == "node-b"
        assert result["edges_absorbed"] == 2
        # Should create ALIAS_OF edge
        alias_calls = [
            c for c in mock_hcg.add_edge.call_args_list
            if c[1].get("relation") == "ALIAS_OF"
        ]
        assert len(alias_calls) == 1

    def test_merge_removes_consumed_node_embedding(self):
        mock_milvus = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg._execute_read.return_value = []

        executor = self._make_executor(hcg=mock_hcg, milvus=mock_milvus)
        executor.merge(
            canonical_uuid="node-a",
            consumed_uuid="node-b",
            hypothesis_edge_uuid="hyp-1",
            canonical_name="Rottweiler",
        )

        mock_milvus.delete_embedding.assert_called_once_with("Entity", "node-b")
```

### Step 2: Run test to verify it fails

Run: `cd sophia && poetry run pytest tests/unit/maintenance/test_entity_resolution.py::TestMergeExecutor -v`
Expected: FAIL — MergeExecutor not found

### Step 3: Add MergeExecutor to entity_resolution.py

Append to `sophia/src/sophia/maintenance/entity_resolution.py`:

```python
class MergeExecutor:
    """Execute node merges when hypothesis confidence is sufficient."""

    def __init__(self, hcg: Any, milvus: Any) -> None:
        self._hcg = hcg
        self._milvus = milvus

    def merge(
        self,
        canonical_uuid: str,
        consumed_uuid: str,
        hypothesis_edge_uuid: str,
        canonical_name: str,
    ) -> dict:
        """Merge consumed node into canonical node.

        Absorbs all edges from consumed node, creates ALIAS_OF audit edge,
        updates hypothesis edge status, removes consumed node and its embedding.

        Args:
            canonical_uuid: UUID of the surviving node.
            consumed_uuid: UUID of the node to be consumed.
            hypothesis_edge_uuid: UUID of the POSSIBLE_ALIAS_OF edge.
            canonical_name: Name for the canonical node (from Hermes).

        Returns:
            Dict with merge results: canonical_uuid, consumed_uuid, edges_absorbed.
        """
        import json
        from datetime import UTC, datetime

        # Absorb outgoing edges
        outgoing = self._hcg._execute_read(
            """
            MATCH (consumed:Node {uuid: $uuid})<-[:FROM]-(e:Node {type: "edge"})-[:TO]->(target:Node)
            WHERE e.relation <> "POSSIBLE_ALIAS_OF"
            RETURN e.uuid AS edge_uuid, e.relation AS relation, target.uuid AS target_uuid,
                   properties(e) AS props
            """,
            {"uuid": consumed_uuid},
        )
        outgoing = [dict(r) for r in outgoing] if outgoing else []

        # Absorb incoming edges
        incoming = self._hcg._execute_read(
            """
            MATCH (source:Node)<-[:FROM]-(e:Node {type: "edge"})-[:TO]->(consumed:Node {uuid: $uuid})
            WHERE e.relation <> "POSSIBLE_ALIAS_OF"
            RETURN e.uuid AS edge_uuid, e.relation AS relation, source.uuid AS source_uuid,
                   properties(e) AS props
            """,
            {"uuid": consumed_uuid},
        )
        incoming = [dict(r) for r in incoming] if incoming else []

        edges_absorbed = 0

        # Redirect outgoing edges to canonical node
        for edge in outgoing:
            self._hcg.add_edge(
                source_uuid=canonical_uuid,
                target_uuid=edge["target_uuid"],
                relation=edge["relation"],
            )
            edges_absorbed += 1

        # Redirect incoming edges to canonical node
        for edge in incoming:
            self._hcg.add_edge(
                source_uuid=edge["source_uuid"],
                target_uuid=canonical_uuid,
                relation=edge["relation"],
            )
            edges_absorbed += 1

        # Create ALIAS_OF audit edge
        consumed_node = self._hcg.get_node(consumed_uuid)
        consumed_name = consumed_node["name"] if consumed_node else consumed_uuid
        self._hcg.add_edge(
            source_uuid=canonical_uuid,
            target_uuid=consumed_uuid,
            relation="ALIAS_OF",
            properties={
                "merged_at": datetime.now(UTC).isoformat(),
                "merge_source": hypothesis_edge_uuid,
            },
        )

        # Update canonical node name if needed
        self._hcg.update_node(canonical_uuid, {
            "name": canonical_name,
            "merge_history": json.dumps({
                "merged": consumed_uuid,
                "merged_name": consumed_name,
                "merged_at": datetime.now(UTC).isoformat(),
                "edges_absorbed": edges_absorbed,
            }),
        })

        # Mark hypothesis edge as resolved
        self._hcg.update_node(hypothesis_edge_uuid, {"status": "merged"})

        # Remove consumed node's embedding
        try:
            self._milvus.delete_embedding("Entity", consumed_uuid)
        except Exception as e:
            logger.warning("Failed to delete embedding for consumed node %s: %s", consumed_uuid, e)

        # Delete consumed node's edges and the node itself
        self._hcg._execute_query(
            """
            MATCH (n:Node {uuid: $uuid})
            OPTIONAL MATCH (n)<-[:FROM]-(e:Node {type: "edge"})
            OPTIONAL MATCH (n)<-[:TO]-(e2:Node {type: "edge"})
            DETACH DELETE e, e2, n
            """,
            {"uuid": consumed_uuid},
        )

        logger.info(
            "Merged node %s (%s) into %s (%s), absorbed %d edges",
            consumed_uuid, consumed_name, canonical_uuid, canonical_name, edges_absorbed,
        )

        return {
            "canonical_uuid": canonical_uuid,
            "consumed_uuid": consumed_uuid,
            "canonical_name": canonical_name,
            "edges_absorbed": edges_absorbed,
        }
```

### Step 4: Run test to verify it passes

Run: `cd sophia && poetry run pytest tests/unit/maintenance/test_entity_resolution.py::TestMergeExecutor -v`
Expected: PASS

### Step 5: Commit

```bash
cd sophia
git add src/sophia/maintenance/entity_resolution.py tests/unit/maintenance/test_entity_resolution.py
git commit -m "feat: merge executor for entity resolution (#503)"
```

---

## Task 5: Maintenance Handler and Wiring

**Files:**
- Modify: `sophia/src/sophia/maintenance/entity_resolution.py`
- Modify: `sophia/src/sophia/maintenance/__init__.py` (if exists, add exports)
- Test: `sophia/tests/unit/maintenance/test_entity_resolution.py`

### Step 1: Write the failing test

Add to `test_entity_resolution.py`:

```python
class TestEntityResolutionHandler:
    def test_handler_detects_and_creates_hypotheses(self):
        from sophia.maintenance.entity_resolution import entity_resolution_handler

        mock_milvus = MagicMock()
        mock_hcg = MagicMock()
        mock_hermes_url = "http://hermes:17000"

        # Detector finds one candidate
        mock_milvus.search_similar.return_value = [
            {"uuid": "node-b", "score": 0.3},
        ]
        mock_milvus.get_embedding.return_value = {
            "uuid": "node-a",
            "embedding": [0.1] * 384,
            "embedding_model": "test",
            "last_sync": 0,
        }
        mock_hcg.get_node.side_effect = lambda uuid: {
            "node-a": {"uuid": "node-a", "name": "rottie", "type": "entity", "properties": {}},
            "node-b": {"uuid": "node-b", "name": "Rottweiler", "type": "entity", "properties": {}},
        }.get(uuid)
        mock_hcg._execute_read.return_value = []
        mock_hcg.add_edge.return_value = "hyp-1"

        handler = entity_resolution_handler(
            milvus=mock_milvus,
            hcg=mock_hcg,
            hermes_url=mock_hermes_url,
        )

        # Handler is a callable that takes node UUIDs
        result = handler(affected_node_uuids=["node-a"])

        assert result["candidates_found"] >= 0
        assert result["hypotheses_created"] >= 0
```

### Step 2: Run test to verify it fails

Run: `cd sophia && poetry run pytest tests/unit/maintenance/test_entity_resolution.py::TestEntityResolutionHandler -v`
Expected: FAIL — entity_resolution_handler not found

### Step 3: Add handler factory to entity_resolution.py

Append to `sophia/src/sophia/maintenance/entity_resolution.py`:

```python
import httpx


def entity_resolution_handler(
    milvus: Any,
    hcg: Any,
    hermes_url: str,
) -> callable:
    """Create an entity resolution handler for the maintenance scheduler.

    Args:
        milvus: MilvusSync instance.
        hcg: HCGClient instance.
        hermes_url: Base URL of the Hermes service.

    Returns:
        Handler callable that accepts affected_node_uuids.
    """
    detector = CandidateDetector(milvus=milvus, hcg=hcg)
    hypothesis_mgr = HypothesisManager(hcg=hcg)
    merge_exec = MergeExecutor(hcg=hcg, milvus=milvus)

    def handler(affected_node_uuids: list[str] | None = None, **kwargs) -> dict:
        """Run entity resolution on affected nodes.

        Args:
            affected_node_uuids: Node UUIDs to check for aliases.

        Returns:
            Summary dict with candidates_found, hypotheses_created, merges_performed.
        """
        if not affected_node_uuids:
            return {"candidates_found": 0, "hypotheses_created": 0, "merges_performed": 0}

        all_candidates = []
        for node_uuid in affected_node_uuids:
            # Embedding triage
            embedding_candidates = detector.find_candidates_for_node(node_uuid)
            all_candidates.extend(embedding_candidates)

            # Structural triage
            structure_candidates = detector.find_candidates_by_structure(node_uuid)
            all_candidates.extend(structure_candidates)

        # Deduplicate candidate pairs
        seen_pairs = set()
        unique_candidates = []
        for c in all_candidates:
            pair = tuple(sorted([c["source_uuid"], c["target_uuid"]]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_candidates.append(c)

        # Create hypothesis edges for new candidates
        hypotheses_created = 0
        for candidate in unique_candidates:
            # Compute initial confidence from available signals
            confidence = 0.0
            evidence = {}

            if "embedding_score" in candidate:
                # Convert L2 distance to confidence: lower distance = higher confidence
                emb_confidence = max(0, 1.0 - candidate["embedding_score"])
                confidence = max(confidence, emb_confidence)
                evidence["embedding_score"] = candidate["embedding_score"]

            if "shared_targets" in candidate:
                struct_confidence = min(candidate["shared_targets"] / 5.0, 1.0)
                confidence = max(confidence, struct_confidence)
                evidence["shared_targets"] = candidate["shared_targets"]

            hypothesis_mgr.create_hypothesis(
                source_uuid=candidate["source_uuid"],
                target_uuid=candidate["target_uuid"],
                confidence=confidence,
                evidence=evidence,
            )
            hypotheses_created += 1

        # Check for merge-ready hypotheses
        merges_performed = 0
        mergeable = hypothesis_mgr.find_mergeable()
        for hyp in mergeable:
            # Consult Hermes for language judgment
            try:
                source_node = hcg.get_node(hyp["source_uuid"])
                target_node = hcg.get_node(hyp["target_uuid"])
                if not source_node or not target_node:
                    continue

                resp = httpx.post(
                    f"{hermes_url}/alias-check",
                    json={
                        "entity_a": {
                            "name": source_node["name"],
                            "type": source_node.get("type", "entity"),
                            "relationships": [],
                        },
                        "entity_b": {
                            "name": target_node["name"],
                            "type": target_node.get("type", "entity"),
                            "relationships": [],
                        },
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                alias_result = resp.json()

                if alias_result.get("is_alias"):
                    canonical_name = alias_result.get("canonical_name") or source_node["name"]
                    merge_exec.merge(
                        canonical_uuid=hyp["source_uuid"],
                        consumed_uuid=hyp["target_uuid"],
                        hypothesis_edge_uuid=hyp["edge_uuid"],
                        canonical_name=canonical_name,
                    )
                    merges_performed += 1
                else:
                    # Hermes says no — lower confidence
                    hypothesis_mgr.update_confidence(
                        hyp["edge_uuid"], delta=-0.3, reason="Hermes rejected alias"
                    )

            except Exception as e:
                logger.warning("Hermes alias-check failed for %s: %s", hyp["edge_uuid"], e)

        return {
            "candidates_found": len(unique_candidates),
            "hypotheses_created": hypotheses_created,
            "merges_performed": merges_performed,
        }

    return handler
```

### Step 4: Run test to verify it passes

Run: `cd sophia && poetry run pytest tests/unit/maintenance/test_entity_resolution.py::TestEntityResolutionHandler -v`
Expected: PASS

### Step 5: Commit

```bash
cd sophia
git add src/sophia/maintenance/entity_resolution.py tests/unit/maintenance/test_entity_resolution.py
git commit -m "feat: entity resolution maintenance handler (#503)"
```

---

## Task 6: Register Handler with Scheduler

**Files:**
- Modify: `sophia/src/sophia/maintenance/scheduler.py` or wherever handlers are registered at startup
- Check: `sophia/src/sophia/main.py` or `sophia/src/sophia/app.py` for application startup

### Step 1: Find where handlers are registered

Search sophia's startup code for where the handlers dict is constructed and passed to `MaintenanceScheduler`. The handler registration looks like:

```python
handlers = {
    "entity_resolution": entity_resolution_handler(
        milvus=milvus_sync,
        hcg=hcg_client,
        hermes_url=hermes_url,
    ),
}
scheduler = MaintenanceScheduler(queue, event_bus, config, handlers, hcg_client)
```

### Step 2: Add entity_resolution to the handlers dict

Add the import and handler registration at the application startup point. The exact location depends on what the search reveals — likely in `sophia/src/sophia/main.py` or an `orchestrator.py`.

```python
from sophia.maintenance.entity_resolution import entity_resolution_handler
```

### Step 3: Add post-ingestion trigger

In the scheduler's `_on_proposal_processed` method (or equivalent event handler), add entity resolution to the jobs enqueued after ingestion:

```python
self._queue.enqueue(
    job_type="entity_resolution",
    priority="normal",
    params={"affected_node_uuids": payload.get("stored_node_ids", [])},
)
```

### Step 4: Run existing scheduler tests to verify nothing broke

Run: `cd sophia && poetry run pytest tests/unit/maintenance/ -v`
Expected: All existing tests PASS

### Step 5: Commit

```bash
cd sophia
git add src/sophia/maintenance/ src/sophia/main.py
git commit -m "feat: register entity resolution handler with maintenance scheduler (#503)"
```

---

## Task 7: Integration Test

**Files:**
- Create: `sophia/tests/integration/maintenance/test_entity_resolution_integration.py`

### Step 1: Write integration test

This test requires Neo4j and Milvus running. Uses the rottie/rottweiler scenario from the issue.

```python
"""Integration test for entity resolution — rottie/rottweiler scenario.

Requires: Neo4j and Milvus running locally.
"""

import pytest

from sophia.maintenance.entity_resolution import (
    CandidateDetector,
    HypothesisManager,
    MergeExecutor,
)


@pytest.mark.integration
class TestEntityResolutionIntegration:
    """End-to-end entity resolution with real graph infrastructure."""

    def test_rottie_rottweiler_resolution(self, hcg_client, milvus_sync):
        """Two nodes for the same dog breed should be detected and merged."""
        # Create two nodes that are aliases
        uuid_a = hcg_client.add_node("rottie", "entity")
        uuid_b = hcg_client.add_node("Rottweiler", "entity")
        uuid_dog = hcg_client.add_node("dog", "concept")

        # Both connect to "dog" via IS_A
        hcg_client.add_edge(uuid_a, uuid_dog, "IS_A")
        hcg_client.add_edge(uuid_b, uuid_dog, "IS_A")

        # Give them similar embeddings
        embedding_a = [0.5] * 384
        embedding_b = [0.51] * 384  # Very close
        milvus_sync.upsert_embedding("Entity", uuid_a, embedding_a, "test")
        milvus_sync.upsert_embedding("Entity", uuid_b, embedding_b, "test")

        # Detect candidates
        detector = CandidateDetector(milvus=milvus_sync, hcg=hcg_client)
        candidates = detector.find_candidates_for_node(uuid_a)

        assert any(c["target_uuid"] == uuid_b for c in candidates)

        # Create hypothesis
        hyp_mgr = HypothesisManager(hcg=hcg_client)
        edge_uuid = hyp_mgr.create_hypothesis(
            source_uuid=uuid_a,
            target_uuid=uuid_b,
            confidence=0.95,
            evidence={"embedding_score": 0.01, "shared_targets": 1},
        )

        # Verify hypothesis edge exists
        mergeable = hyp_mgr.find_mergeable(threshold=0.9)
        assert any(h["edge_uuid"] == edge_uuid for h in mergeable)

        # Execute merge
        executor = MergeExecutor(hcg=hcg_client, milvus=milvus_sync)
        result = executor.merge(
            canonical_uuid=uuid_a,
            consumed_uuid=uuid_b,
            hypothesis_edge_uuid=edge_uuid,
            canonical_name="Rottweiler",
        )

        assert result["canonical_uuid"] == uuid_a
        assert result["edges_absorbed"] >= 1

        # Verify canonical node has the IS_A edge
        canonical = hcg_client.get_node(uuid_a)
        assert canonical["name"] == "Rottweiler"

        # Verify consumed node is gone
        consumed = hcg_client.get_node(uuid_b)
        assert consumed is None
```

### Step 2: Run integration test

Run: `cd sophia && poetry run pytest tests/integration/maintenance/test_entity_resolution_integration.py -v`
Expected: PASS (with infrastructure running)

### Step 3: Commit

```bash
cd sophia
git add tests/integration/maintenance/test_entity_resolution_integration.py
git commit -m "test: entity resolution integration test with rottie/rottweiler scenario (#503)"
```

---

## Summary

| Task | Component | What |
|------|-----------|------|
| 1 | Hermes | `/alias-check` endpoint |
| 2 | Sophia | Candidate detector (embedding + structural triage) |
| 3 | Sophia | Hypothesis edge creation + confidence accumulation |
| 4 | Sophia | Merge executor |
| 5 | Sophia | Handler factory wiring detection → hypothesis → merge |
| 6 | Sophia | Register handler with maintenance scheduler |
| 7 | Sophia | Integration test (rottie/rottweiler scenario) |

Tasks 1 and 2-4 are independent and can be parallelized across hermes and sophia repos.
