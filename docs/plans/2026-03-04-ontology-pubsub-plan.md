# Ontology Pub/Sub Distribution (#501) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire EventBus into sophia's proposal processor to publish batch events, write type snapshots to Redis, and build a TypeRegistry in hermes that stays in sync via pub/sub.

**Architecture:** ProposalProcessor gets an EventBus injected. After each `process()` call, it publishes a `proposal_processed` batch event and writes the full type list to a Redis key. Hermes reads that key on boot and subscribes to the channel for real-time updates via a new TypeRegistry class.

**Tech Stack:** logos_events.EventBus, redis-py, pydantic, threading

**Design Doc:** `docs/plans/2026-03-04-ontology-pubsub-and-scheduler-design.md`

---

## Phase 1: Sophia Publisher (branch: `feat/sophia-501-pubsub-publisher`)

### Task 1: Add EventBus to ProposalProcessor constructor

**Files:**
- Modify: `sophia/src/sophia/ingestion/proposal_processor.py:94-101`
- Modify: `sophia/src/sophia/api/app.py:396-399`
- Test: `sophia/tests/ingestion/test_proposal_processor.py`

**Step 1: Write the failing test**

Add to `sophia/tests/ingestion/test_proposal_processor.py`:

```python
from unittest.mock import MagicMock

def test_proposal_processor_accepts_event_bus():
    """ProposalProcessor accepts optional event_bus parameter."""
    mock_hcg = MagicMock()
    mock_milvus = MagicMock()
    mock_event_bus = MagicMock()
    processor = ProposalProcessor(mock_hcg, mock_milvus, event_bus=mock_event_bus)
    assert processor._event_bus is mock_event_bus


def test_proposal_processor_event_bus_defaults_to_none():
    """ProposalProcessor works without event_bus."""
    mock_hcg = MagicMock()
    mock_milvus = MagicMock()
    processor = ProposalProcessor(mock_hcg, mock_milvus)
    assert processor._event_bus is None
```

**Step 2: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/ingestion/test_proposal_processor.py::test_proposal_processor_accepts_event_bus -v`
Expected: FAIL — TypeError (unexpected keyword argument 'event_bus')

**Step 3: Write minimal implementation**

In `sophia/src/sophia/ingestion/proposal_processor.py`, modify the constructor (lines 94-101):

```python
def __init__(
    self,
    hcg_client: Any,
    milvus_sync: Any,
    event_bus: Any | None = None,
    redis_client: Any | None = None,
) -> None:
    self._hcg = hcg_client
    self._milvus = milvus_sync
    self._classifier = TypeClassifier(milvus=milvus_sync, hcg=hcg_client)
    self._event_bus = event_bus
    self._redis = redis_client
```

Add import at top of file:

```python
import json as _json
```

**Step 4: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/ingestion/test_proposal_processor.py -v -k "event_bus"`
Expected: PASS

**Step 5: Commit**

```bash
cd sophia && git add src/sophia/ingestion/proposal_processor.py tests/ingestion/test_proposal_processor.py
git commit -m "feat: add event_bus and redis_client params to ProposalProcessor"
```

---

### Task 2: Publish batch event after process() completes

**Files:**
- Modify: `sophia/src/sophia/ingestion/proposal_processor.py:448-452`
- Test: `sophia/tests/ingestion/test_proposal_processor.py`

**Step 1: Write the failing test**

```python
from unittest.mock import MagicMock, patch, call
import json


def test_process_publishes_batch_event(mock_proposal_fixtures):
    """process() publishes a proposal_processed event via EventBus."""
    mock_hcg = MagicMock()
    mock_milvus = MagicMock()
    mock_event_bus = MagicMock()
    mock_redis = MagicMock()

    # Configure mocks to simulate successful node creation
    mock_hcg.add_node.return_value = "node-uuid-1"
    mock_hcg.get_node.return_value = {"uuid": "type-uuid-1", "name": "person", "member_count": 1}
    mock_hcg.find_nodes_by_names.return_value = {}
    mock_milvus.search_similar.return_value = []
    mock_milvus.batch_upsert_embeddings.return_value = None

    processor = ProposalProcessor(mock_hcg, mock_milvus, event_bus=mock_event_bus, redis_client=mock_redis)

    # Minimal proposal with one node
    proposal = {
        "proposed_nodes": [{"name": "Alice", "node_type": "person", "properties": {}}],
        "proposed_edges": [],
        "document_embedding": [0.1] * 384,
        "raw_text": "test",
    }

    processor.process(proposal)

    # Verify event was published
    mock_event_bus.publish.assert_called_once()
    channel, event = mock_event_bus.publish.call_args[0]
    assert channel == "logos:sophia:proposal_processed"
    assert event["event_type"] == "proposal_processed"
    assert "payload" in event
    assert "affected_node_uuids" in event["payload"]
```

Note: This test may need adjustment based on how `mock_proposal_fixtures` are structured in the existing test file. Check existing test patterns first.

**Step 2: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/ingestion/test_proposal_processor.py::test_process_publishes_batch_event -v`
Expected: FAIL — publish not called (no event publishing code yet)

**Step 3: Write minimal implementation**

Add a `_publish_batch_event` method to ProposalProcessor and call it at the end of `process()`.

After the return statement area (line ~448), but before the return, add event publishing. Modify the end of `process()`:

```python
    def _publish_batch_event(
        self,
        stored_node_ids: list[str],
        stored_edge_ids: list[str],
        new_types: list[str],
        updated_types: list[str],
        affected_node_uuids: list[str],
    ) -> None:
        """Publish a batch event summarizing the proposal processing."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(
                "logos:sophia:proposal_processed",
                {
                    "event_type": "proposal_processed",
                    "source": "sophia",
                    "payload": {
                        "new_types": new_types,
                        "updated_types": updated_types,
                        "stored_node_ids": stored_node_ids,
                        "stored_edge_ids": stored_edge_ids,
                        "affected_node_uuids": affected_node_uuids,
                    },
                },
            )
        except Exception:
            logger.exception("Failed to publish proposal_processed event")
```

In the `process()` method, track new/updated types as nodes are processed. Before the return dict, call `_publish_batch_event()`. This requires:

1. Initialize tracking lists at the start of `process()`:
```python
        new_types: list[str] = []
        updated_types: list[str] = []
        affected_node_uuids: list[str] = []
```

2. When a node is created (after `add_node()` at line ~253), append to `affected_node_uuids`.

3. When a type_definition is created vs updated (around lines 267-288), track in `new_types` or `updated_types`.

4. Before the return dict (line ~448), call:
```python
        self._publish_batch_event(
            stored_node_ids=stored_node_ids,
            stored_edge_ids=stored_edge_ids,
            new_types=new_types,
            updated_types=updated_types,
            affected_node_uuids=affected_node_uuids,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/ingestion/test_proposal_processor.py -v -k "batch_event"`
Expected: PASS

**Step 5: Commit**

```bash
cd sophia && git add src/sophia/ingestion/proposal_processor.py tests/ingestion/test_proposal_processor.py
git commit -m "feat: publish proposal_processed batch event via EventBus"
```

---

### Task 3: Write type snapshot to Redis after processing

**Files:**
- Modify: `sophia/src/sophia/ingestion/proposal_processor.py`
- Test: `sophia/tests/ingestion/test_proposal_processor.py`

**Step 1: Write the failing test**

```python
def test_process_writes_type_snapshot_to_redis():
    """process() writes full type list to Redis key logos:ontology:types."""
    mock_hcg = MagicMock()
    mock_milvus = MagicMock()
    mock_event_bus = MagicMock()
    mock_redis = MagicMock()

    # Mock get_type_definitions to return current types
    mock_hcg.get_type_definitions.return_value = [
        {"uuid": "t1", "name": "person", "member_count": 10},
        {"uuid": "t2", "name": "location", "member_count": 5},
    ]
    mock_hcg.add_node.return_value = "node-uuid-1"
    mock_hcg.get_node.return_value = {"uuid": "t1", "name": "person", "member_count": 10}
    mock_hcg.find_nodes_by_names.return_value = {}
    mock_milvus.search_similar.return_value = []
    mock_milvus.batch_upsert_embeddings.return_value = None

    processor = ProposalProcessor(mock_hcg, mock_milvus, event_bus=mock_event_bus, redis_client=mock_redis)

    proposal = {
        "proposed_nodes": [{"name": "Alice", "node_type": "person", "properties": {}}],
        "proposed_edges": [],
        "document_embedding": [0.1] * 384,
        "raw_text": "test",
    }

    processor.process(proposal)

    # Verify Redis write
    mock_redis.set.assert_called_once()
    key, value = mock_redis.set.call_args[0]
    assert key == "logos:ontology:types"
    snapshot = json.loads(value)
    assert "person" in snapshot
    assert "location" in snapshot
```

**Step 2: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/ingestion/test_proposal_processor.py::test_process_writes_type_snapshot_to_redis -v`
Expected: FAIL — set not called

**Step 3: Write minimal implementation**

Add `_write_type_snapshot` method to ProposalProcessor:

```python
    def _write_type_snapshot(self) -> None:
        """Write full type list to Redis for Hermes initial sync."""
        if self._redis is None:
            return
        try:
            type_defs = self._hcg.get_type_definitions()
            snapshot = {
                td["name"]: {
                    "uuid": td["uuid"],
                    "member_count": td.get("member_count", 0),
                }
                for td in type_defs
            }
            self._redis.set("logos:ontology:types", _json.dumps(snapshot))
        except Exception:
            logger.exception("Failed to write type snapshot to Redis")
```

Call it after `_publish_batch_event()` in `process()`:

```python
        self._write_type_snapshot()
```

Note: Check if `hcg_client` has a `get_type_definitions()` method. If not, we need to add a query that fetches all nodes with `node_type == "type_definition"`. The HCGClient likely has something like `query_nodes()` that can be used. Check `logos/logos_hcg/client.py` for available methods and adapt accordingly.

**Step 4: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/ingestion/test_proposal_processor.py -v -k "type_snapshot"`
Expected: PASS

**Step 5: Commit**

```bash
cd sophia && git add src/sophia/ingestion/proposal_processor.py tests/ingestion/test_proposal_processor.py
git commit -m "feat: write ontology type snapshot to Redis after proposal processing"
```

---

### Task 4: Initialize EventBus in sophia app.py and pass to ProposalProcessor

**Files:**
- Modify: `sophia/src/sophia/api/app.py:396-399`

**Step 1: Add EventBus initialization in lifespan**

In `sophia/src/sophia/api/app.py`, add after the existing imports (around line 33):

```python
from logos_events import EventBus
```

Add a global variable (around line 242):

```python
_event_bus: EventBus | None = None
```

In the lifespan startup section, before ProposalProcessor creation (around line 396), add:

```python
        # Initialize EventBus for pub/sub
        from logos_config import RedisConfig
        _redis_config = RedisConfig()
        _event_bus = EventBus(_redis_config)
        _redis_direct = redis.from_url(_redis_config.url)
        logger.info("EventBus initialized for pub/sub")
```

Add `import redis` to imports if not already there.

Modify the ProposalProcessor construction (lines 396-399) to pass event_bus and redis_client:

```python
        _proposal_processor = ProposalProcessor(
            hcg_client=_hcg_client,
            milvus_sync=_milvus_sync,
            event_bus=_event_bus,
            redis_client=_redis_direct,
        )
```

In shutdown (after line 456), add cleanup:

```python
        if _event_bus is not None:
            _event_bus.close()
            logger.info("EventBus closed")
```

**Step 2: Run existing tests to verify nothing breaks**

Run: `cd sophia && poetry run pytest tests/ -x --no-infra -q`
Expected: ALL PASS

**Step 3: Commit**

```bash
cd sophia && git add src/sophia/api/app.py
git commit -m "feat: initialize EventBus and pass to ProposalProcessor"
```

---

## Phase 2: Hermes TypeRegistry (branch: `feat/hermes-501-type-registry`)

### Task 5: Create TypeRegistry class

**Files:**
- Create: `hermes/src/hermes/type_registry.py`
- Create: `hermes/tests/test_type_registry.py`

**Step 1: Write the failing test**

Create `hermes/tests/test_type_registry.py`:

```python
"""Tests for hermes.type_registry.TypeRegistry."""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock

import pytest

from hermes.type_registry import TypeRegistry


class TestTypeRegistryInit:
    """Tests for TypeRegistry initialization from Redis snapshot."""

    def test_loads_types_from_redis_on_init(self):
        """TypeRegistry reads logos:ontology:types from Redis on init."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({
            "person": {"uuid": "t1", "member_count": 10},
            "location": {"uuid": "t2", "member_count": 5},
        })

        registry = TypeRegistry(mock_redis)

        mock_redis.get.assert_called_with("logos:ontology:types")
        assert registry.get_type_names() == sorted(["person", "location"])

    def test_empty_registry_when_no_snapshot(self):
        """TypeRegistry starts empty if no Redis snapshot exists."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        registry = TypeRegistry(mock_redis)

        assert registry.get_type_names() == []

    def test_get_type_returns_type_dict(self):
        """get_type() returns the type properties dict."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({
            "person": {"uuid": "t1", "member_count": 10},
        })

        registry = TypeRegistry(mock_redis)

        result = registry.get_type("person")
        assert result == {"uuid": "t1", "member_count": 10}

    def test_get_type_returns_none_for_unknown(self):
        """get_type() returns None for unknown types."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({})

        registry = TypeRegistry(mock_redis)

        assert registry.get_type("unknown") is None

    def test_format_for_prompt(self):
        """format_for_prompt() returns formatted type list string."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({
            "person": {"uuid": "t1", "member_count": 10},
            "location": {"uuid": "t2", "member_count": 5},
        })

        registry = TypeRegistry(mock_redis)
        prompt = registry.format_for_prompt()

        assert "person" in prompt
        assert "location" in prompt


class TestTypeRegistryUpdate:
    """Tests for TypeRegistry updates via events."""

    def test_update_from_event(self):
        """on_proposal_processed updates type list from event payload."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({})

        registry = TypeRegistry(mock_redis)

        # Simulate receiving an event
        event = {
            "event_type": "proposal_processed",
            "source": "sophia",
            "timestamp": "2026-03-04T00:00:00Z",
            "payload": {
                "new_types": ["vehicle"],
                "updated_types": [],
            },
        }
        registry.on_proposal_processed(event)

        # After event, registry should re-read from Redis
        mock_redis.get.assert_called_with("logos:ontology:types")
```

**Step 2: Run test to verify it fails**

Run: `cd hermes && poetry run pytest tests/test_type_registry.py -v`
Expected: FAIL — ModuleNotFoundError (type_registry doesn't exist)

**Step 3: Write minimal implementation**

Create `hermes/src/hermes/type_registry.py`:

```python
"""TypeRegistry — live ontology type cache backed by Redis + pub/sub.

Reads the full type list from Redis on boot. Subscribes to
logos:sophia:proposal_processed events to stay in sync as Sophia
evolves the ontology.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class TypeRegistry:
    """Thread-safe registry of ontology types, synced from Redis."""

    REDIS_KEY = "logos:ontology:types"

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._lock = threading.Lock()
        self._types: dict[str, dict] = {}
        self._load_from_redis()

    def _load_from_redis(self) -> None:
        """Load type snapshot from Redis."""
        try:
            raw = self._redis.get(self.REDIS_KEY)
            if raw is not None:
                data = json.loads(raw)
                with self._lock:
                    self._types = data
                logger.info("TypeRegistry loaded %d types from Redis", len(data))
            else:
                logger.info("TypeRegistry: no snapshot in Redis, starting empty")
        except Exception:
            logger.exception("TypeRegistry: failed to load from Redis")

    def get_type_names(self) -> list[str]:
        """Return sorted list of known type names."""
        with self._lock:
            return sorted(self._types.keys())

    def get_type(self, name: str) -> dict | None:
        """Return type properties dict, or None if unknown."""
        with self._lock:
            return self._types.get(name)

    def format_for_prompt(self) -> str:
        """Format type list for injection into NER prompt."""
        with self._lock:
            if not self._types:
                return "No ontology types available."
            lines = []
            for name in sorted(self._types):
                info = self._types[name]
                count = info.get("member_count", 0)
                lines.append(f"- {name} ({count} members)")
            return "Known entity types:\n" + "\n".join(lines)

    def on_proposal_processed(self, event: dict) -> None:
        """Handle proposal_processed event — reload types from Redis.

        This is the EventBus callback. Rather than incrementally updating,
        we re-read the full snapshot from Redis. This is simple and
        guarantees consistency even if events are missed.
        """
        logger.info("TypeRegistry: proposal_processed event, reloading types")
        self._load_from_redis()
```

**Step 4: Run test to verify it passes**

Run: `cd hermes && poetry run pytest tests/test_type_registry.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd hermes && git add src/hermes/type_registry.py tests/test_type_registry.py
git commit -m "feat: add TypeRegistry class for live ontology type sync"
```

---

### Task 6: Wire TypeRegistry into hermes startup

**Files:**
- Modify: `hermes/src/hermes/main.py`

**Step 1: Add TypeRegistry initialization**

In `hermes/src/hermes/main.py`, add import (around line 32):

```python
from hermes.type_registry import TypeRegistry
```

Add global variable (near other globals):

```python
_type_registry: TypeRegistry | None = None
```

Add a function to lazily create the TypeRegistry (near `_get_context_cache()` at line 159):

```python
def _get_type_registry() -> TypeRegistry | None:
    """Return (and lazily create) the module-level TypeRegistry."""
    global _type_registry
    if _type_registry is None:
        try:
            from logos_config import RedisConfig
            import redis
            _redis = redis.from_url(RedisConfig().url)
            _type_registry = TypeRegistry(_redis)
            logger.info("TypeRegistry initialized")
        except Exception:
            logger.exception("Failed to initialize TypeRegistry")
    return _type_registry
```

**Step 2: Add EventBus subscription for live updates**

In the lifespan or startup section, after TypeRegistry is created, subscribe to events:

```python
        # Subscribe TypeRegistry to ontology changes
        registry = _get_type_registry()
        if registry is not None:
            from logos_events import EventBus
            from logos_config import RedisConfig
            _event_bus = EventBus(RedisConfig())
            _event_bus.subscribe(
                "logos:sophia:proposal_processed",
                registry.on_proposal_processed,
            )
            _listener_thread = threading.Thread(
                target=_event_bus.listen, daemon=True, name="type-registry-listener"
            )
            _listener_thread.start()
            logger.info("TypeRegistry subscribed to ontology changes")
```

Note: Check hermes's startup pattern. If hermes has a lifespan context manager, put this there. If not, the lazy init via `_get_type_registry()` with the EventBus subscription may need to be in the first request handler or an `@app.on_event("startup")` handler.

**Step 3: Run existing tests to verify nothing breaks**

Run: `cd hermes && poetry run pytest tests/ -x --no-infra -q`
Expected: ALL PASS

**Step 4: Commit**

```bash
cd hermes && git add src/hermes/main.py
git commit -m "feat: wire TypeRegistry into hermes startup with EventBus subscription"
```

---

### Task 7: Integration test for pub/sub flow

**Files:**
- Create: `sophia/tests/integration/test_pubsub_flow.py`

**Step 1: Write integration test**

```python
"""Integration test: sophia publishes events that hermes can subscribe to.

Requires: Redis running on localhost:6379
"""

from __future__ import annotations

import json
import threading
import time

import pytest
import redis

from logos_config import RedisConfig
from logos_events import EventBus

REDIS_AVAILABLE = False
try:
    r = redis.from_url("redis://localhost:6379/0")
    r.ping()
    REDIS_AVAILABLE = True
except Exception:
    pass

pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")


class TestPubSubFlow:
    """Test the full pub/sub event flow."""

    def test_proposal_processed_event_received(self):
        """Published proposal_processed event is received by subscriber."""
        config = RedisConfig()
        received: list[dict] = []

        # Subscriber
        sub_bus = EventBus(config)
        sub_bus.subscribe("logos:sophia:proposal_processed", lambda e: received.append(e))
        listener = threading.Thread(target=sub_bus.listen, daemon=True)
        listener.start()
        time.sleep(0.2)

        # Publisher
        pub_bus = EventBus(config)
        pub_bus.publish("logos:sophia:proposal_processed", {
            "event_type": "proposal_processed",
            "source": "sophia",
            "payload": {
                "new_types": ["vehicle"],
                "updated_types": ["person"],
                "stored_node_ids": ["n1"],
                "stored_edge_ids": ["e1"],
                "affected_node_uuids": ["n1"],
            },
        })
        pub_bus.close()

        time.sleep(0.3)
        sub_bus.stop()

        assert len(received) == 1
        assert received[0]["event_type"] == "proposal_processed"
        assert received[0]["payload"]["new_types"] == ["vehicle"]

    def test_type_snapshot_written_and_readable(self):
        """Type snapshot written to Redis is readable."""
        config = RedisConfig()
        r = redis.from_url(config.url)

        snapshot = {
            "person": {"uuid": "t1", "member_count": 10},
            "location": {"uuid": "t2", "member_count": 5},
        }
        r.set("logos:ontology:types", json.dumps(snapshot))

        raw = r.get("logos:ontology:types")
        loaded = json.loads(raw)
        assert loaded == snapshot

        # Cleanup
        r.delete("logos:ontology:types")
        r.close()
```

**Step 2: Run integration test**

Run: `cd sophia && poetry run pytest tests/integration/test_pubsub_flow.py -v`
Expected: ALL PASS (or SKIP if Redis not running)

**Step 3: Commit**

```bash
cd sophia && git add tests/integration/test_pubsub_flow.py
git commit -m "test: add integration tests for pub/sub ontology flow"
```

---

## Verification Checklist

- [ ] `cd sophia && poetry run pytest tests/ -x --no-infra -q` — ALL PASS
- [ ] `cd hermes && poetry run pytest tests/ -x --no-infra -q` — ALL PASS
- [ ] `cd sophia && poetry run pytest tests/integration/test_pubsub_flow.py -v` — PASS (with Redis)
- [ ] ProposalProcessor publishes event after process()
- [ ] Type snapshot written to `logos:ontology:types` in Redis
- [ ] TypeRegistry loads snapshot on boot
- [ ] TypeRegistry updates on proposal_processed event
- [ ] format_for_prompt() returns usable type list string
