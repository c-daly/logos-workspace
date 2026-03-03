# Centralized Redis & Pub/Sub Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Promote Redis to shared LOGOS infrastructure (alongside Neo4j/Milvus), add RedisConfig to logos_config, create an EventBus abstraction, and refactor Sophia/Hermes to use centralized config with no graceful fallback.

**Architecture:** Redis becomes required infrastructure. `RedisConfig` joins `Neo4jConfig`/`MilvusConfig` in `logos_config`. A thin `EventBus` class in a new `logos_events` package wraps Redis pub/sub. Sophia and Hermes refactor to use `RedisConfig` and drop optional-Redis code paths. Docker-compose files across all repos get a Redis service.

**Tech Stack:** Redis 7, redis-py >=5.0.0, pydantic-settings, Docker Compose

**Design Doc:** `docs/plans/2026-03-03-centralized-redis-pubsub-design.md`

---

## Phase 1: logos foundry (must merge & tag before Phase 2/3)

### Task 1: Add RedisConfig to logos_config

**Files:**
- Modify: `logos_config/settings.py`
- Modify: `logos_config/__init__.py`
- Modify: `tests/logos_config/test_settings.py`

**Step 1: Write the failing test**

Add to `tests/logos_config/test_settings.py`:

```python
from logos_config.settings import RedisConfig


class TestRedisConfig:
    """Tests for RedisConfig."""

    def test_default_values(self) -> None:
        """Default values are set correctly."""
        for var in ["REDIS_HOST", "REDIS_PORT", "REDIS_DB"]:
            os.environ.pop(var, None)

        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0

    def test_url_property(self) -> None:
        """URL property returns correct redis:// connection string."""
        for var in ["REDIS_HOST", "REDIS_PORT", "REDIS_DB"]:
            os.environ.pop(var, None)

        config = RedisConfig()
        assert config.url == "redis://localhost:6379/0"

        config = RedisConfig(host="redis-server", port=6380, db=2)
        assert config.url == "redis://redis-server:6380/2"

    def test_env_var_override(self) -> None:
        """Environment variables override defaults."""
        with mock.patch.dict(
            os.environ,
            {"REDIS_HOST": "redis-prod", "REDIS_PORT": "6380", "REDIS_DB": "3"},
        ):
            config = RedisConfig()
            assert config.host == "redis-prod"
            assert config.port == 6380
            assert config.db == 3
            assert config.url == "redis://redis-prod:6380/3"
```

**Step 2: Run test to verify it fails**

Run: `cd logos && poetry run pytest tests/logos_config/test_settings.py::TestRedisConfig -v`
Expected: FAIL — `ImportError: cannot import name 'RedisConfig'`

**Step 3: Write minimal implementation**

Add to `logos_config/settings.py` after `MilvusConfig`:

```python
class RedisConfig(BaseSettings):
    """Redis connection configuration.

    Env vars: REDIS_HOST, REDIS_PORT, REDIS_DB
    """

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)

    @property
    def url(self) -> str:
        """Return the redis:// connection URL."""
        return f"redis://{self.host}:{self.port}/{self.db}"
```

Add to `logos_config/__init__.py`:
- Import: `from logos_config.settings import RedisConfig` (add to existing import line)
- Add `"RedisConfig"` to `__all__` list under the `# settings` comment

**Step 4: Run test to verify it passes**

Run: `cd logos && poetry run pytest tests/logos_config/test_settings.py -v`
Expected: ALL PASS (existing + new tests)

**Step 5: Commit**

```bash
cd logos && git add logos_config/settings.py logos_config/__init__.py tests/logos_config/test_settings.py
git commit -m "feat(logos_config): add RedisConfig for shared Redis infrastructure"
```

---

### Task 2: Add Redis port to logos_config/ports.py

**Files:**
- Modify: `logos_config/ports.py`

**Step 1: Update port table documentation**

In the module docstring's port table, add a `Redis` column:

```
| Repo   | Neo4j HTTP | Neo4j Bolt | Milvus gRPC | Milvus Metrics | Redis | API   |
|--------|------------|------------|-------------|----------------|-------|-------|
| hermes | 7474       | 7687       | 19530       | 9091           | 6379  | 17000 |
| apollo | 7474       | 7687       | 19530       | 9091           | 6379  | 27000 |
| logos  | 7474       | 7687       | 19530       | 9091           | 6379  | 37000 |
| sophia | 7474       | 7687       | 19530       | 9091           | 6379  | 47000 |
| talos  | 7474       | 7687       | 19530       | 9091           | 6379  | 57000 |
```

Add `redis` field to `RepoPorts`:

```python
class RepoPorts(NamedTuple):
    """Port configuration for a repo."""
    neo4j_http: int
    neo4j_bolt: int
    milvus_grpc: int
    milvus_metrics: int
    redis: int
    api: int
```

Update all port constants to include redis=6379:

```python
HERMES_PORTS = RepoPorts(7474, 7687, 19530, 9091, 6379, 17000)
APOLLO_PORTS = RepoPorts(7474, 7687, 19530, 9091, 6379, 27000)
LOGOS_PORTS = RepoPorts(7474, 7687, 19530, 9091, 6379, 37000)
SOPHIA_PORTS = RepoPorts(7474, 7687, 19530, 9091, 6379, 47000)
TALOS_PORTS = RepoPorts(7474, 7687, 19530, 9091, 6379, 57000)
```

**Step 2: Run existing tests to verify nothing breaks**

Run: `cd logos && poetry run pytest tests/logos_config/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
cd logos && git add logos_config/ports.py
git commit -m "feat(logos_config): add Redis port to RepoPorts"
```

---

### Task 3: Add redis-py dependency to logos-foundry

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add redis dependency**

Add `redis = ">=5.0.0"` to `[tool.poetry.dependencies]` in `logos/pyproject.toml`.

**Step 2: Lock and verify**

Run: `cd logos && poetry lock --no-update && poetry install`
Expected: redis-py resolves (sophia and hermes already depend on it, so no conflict)

**Step 3: Commit**

```bash
cd logos && git add pyproject.toml poetry.lock
git commit -m "feat: add redis-py dependency to logos-foundry"
```

---

### Task 4: Create logos_events package with EventBus

**Files:**
- Create: `logos_events/__init__.py`
- Create: `logos_events/event_bus.py`
- Create: `tests/logos_events/__init__.py`
- Create: `tests/logos_events/test_event_bus.py`
- Modify: `pyproject.toml` (add package to packages list)

**Step 1: Write the failing tests**

Create `tests/logos_events/__init__.py` (empty).

Create `tests/logos_events/test_event_bus.py`:

```python
"""Tests for logos_events.EventBus."""

from __future__ import annotations

import json
import threading
import time

import pytest
import redis

from logos_config.settings import RedisConfig
from logos_events.event_bus import EventBus

REDIS_AVAILABLE = False
try:
    r = redis.from_url("redis://localhost:6379/0")
    r.ping()
    REDIS_AVAILABLE = True
except Exception:
    pass

pytestmark = pytest.mark.skipif(
    not REDIS_AVAILABLE, reason="Redis not available"
)


class TestEventBus:
    """Tests for EventBus."""

    def setup_method(self) -> None:
        self.config = RedisConfig()
        self.bus = EventBus(self.config)

    def teardown_method(self) -> None:
        self.bus.close()

    def test_publish_and_subscribe(self) -> None:
        """Published events are received by subscribers."""
        received: list[dict] = []

        def on_event(event: dict) -> None:
            received.append(event)

        self.bus.subscribe("logos:test:ping", on_event)

        # Start listener in background thread
        listener = threading.Thread(target=self.bus.listen, daemon=True)
        listener.start()

        # Give subscriber time to register
        time.sleep(0.1)

        # Publish from a separate connection
        pub_bus = EventBus(self.config)
        pub_bus.publish("logos:test:ping", {
            "event_type": "ping",
            "source": "test",
            "payload": {"value": 42},
        })
        pub_bus.close()

        # Wait for delivery
        time.sleep(0.2)
        self.bus.stop()

        assert len(received) == 1
        assert received[0]["event_type"] == "ping"
        assert received[0]["source"] == "test"
        assert received[0]["payload"] == {"value": 42}
        assert "timestamp" in received[0]

    def test_publish_envelope_format(self) -> None:
        """Published events include standard envelope fields."""
        received: list[dict] = []

        def on_event(event: dict) -> None:
            received.append(event)

        self.bus.subscribe("logos:test:envelope", on_event)
        listener = threading.Thread(target=self.bus.listen, daemon=True)
        listener.start()
        time.sleep(0.1)

        pub_bus = EventBus(self.config)
        pub_bus.publish("logos:test:envelope", {
            "event_type": "test_event",
            "source": "sophia",
            "payload": {},
        })
        pub_bus.close()

        time.sleep(0.2)
        self.bus.stop()

        assert len(received) == 1
        event = received[0]
        assert "event_type" in event
        assert "source" in event
        assert "timestamp" in event
        assert "payload" in event

    def test_close_is_idempotent(self) -> None:
        """Closing multiple times does not raise."""
        self.bus.close()
        self.bus.close()  # Should not raise
```

**Step 2: Run test to verify it fails**

Run: `cd logos && poetry run pytest tests/logos_events/test_event_bus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'logos_events'`

**Step 3: Write minimal implementation**

Create `logos_events/__init__.py`:

```python
"""logos_events - Redis pub/sub event bus for LOGOS services."""

from logos_events.event_bus import EventBus

__all__ = ["EventBus"]
```

Create `logos_events/event_bus.py`:

```python
"""Redis pub/sub event bus for LOGOS services.

Provides a thin wrapper over redis-py pub/sub for inter-service
event communication. Events use a standard envelope format with
event_type, source, timestamp, and payload.

Channel naming convention: logos:<service>:<event_type>
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

import redis

from logos_config.settings import RedisConfig

logger = logging.getLogger(__name__)


class EventBus:
    """Redis pub/sub event bus for LOGOS services."""

    def __init__(self, redis_config: RedisConfig) -> None:
        self._redis = redis.from_url(redis_config.url)
        self._pubsub = self._redis.pubsub()
        self._callbacks: dict[str, Callable[[dict], None]] = {}
        self._running = False

    def publish(self, channel: str, event: dict) -> None:
        """Publish an event to a channel.

        The event dict should contain event_type, source, and payload.
        A timestamp is added automatically.
        """
        envelope = {
            "event_type": event.get("event_type", "unknown"),
            "source": event.get("source", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": event.get("payload", {}),
        }
        self._redis.publish(channel, json.dumps(envelope))

    def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        """Subscribe to a channel with a callback.

        The callback receives the parsed event dict (envelope).
        """
        self._callbacks[channel] = callback

        def _handler(message: dict[str, Any]) -> None:
            try:
                data = json.loads(message["data"])
                callback(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to parse event on %s: %s", channel, e)

        self._pubsub.subscribe(**{channel: _handler})

    def listen(self) -> None:
        """Blocking listen loop. Run in a background thread.

        Call stop() from another thread to terminate.
        """
        self._running = True
        for message in self._pubsub.listen():
            if not self._running:
                break

    def stop(self) -> None:
        """Signal the listen loop to stop."""
        self._running = False
        self._pubsub.unsubscribe()

    def close(self) -> None:
        """Close connections. Idempotent."""
        self.stop()
        try:
            self._pubsub.close()
        except Exception:
            pass
        try:
            self._redis.close()
        except Exception:
            pass
```

Add `logos_events` to `pyproject.toml` packages list (find `packages = [` and add `{ include = "logos_events" }`).

**Step 4: Run test to verify it passes**

Run: `cd logos && poetry run pytest tests/logos_events/ -v`
Expected: ALL PASS (or SKIP if Redis not running — these are infra tests)

**Step 5: Commit**

```bash
cd logos && git add logos_events/ tests/logos_events/ pyproject.toml
git commit -m "feat: add logos_events package with EventBus pub/sub abstraction"
```

---

### Task 5: Add Redis to docker-compose files

**Files:**
- Modify: `logos/infra/test_stack/docker-compose.test.yml`
- Modify: `logos/infra/sophia/docker-compose.test.yml`
- Modify: `logos/infra/hermes/docker-compose.test.yml`
- Modify: `logos/infra/apollo/docker-compose.test.yml`
- Modify: `logos/infra/talos/docker-compose.test.yml`

**Step 1: Add Redis service to each docker-compose**

Add the following service block and volume to each file. Use the same naming pattern as the existing services (prefix varies per file):

For `logos/infra/test_stack/docker-compose.test.yml` (prefix: `logos-test-`):

```yaml
# Under volumes: add
  logos_test_redis_data: {}

# Under services: add
  redis:
    container_name: logos-test-redis
    image: redis:7-alpine
    ports:
    - 6379:6379
    volumes:
    - logos_test_redis_data:/data
    healthcheck:
      test:
      - CMD
      - redis-cli
      - ping
      interval: 5s
      timeout: 3s
      retries: 3
    networks:
    - default
```

For each repo-specific compose (sophia, hermes, apollo, talos), use the same block but with the appropriate container name prefix (e.g., `sophia-test-redis`, `hermes-test-redis`) and volume prefix (e.g., `sophia_test_redis_data`).

**Step 2: Verify compose files parse**

Run: `cd logos && docker compose -f infra/test_stack/docker-compose.test.yml config --quiet`
Expected: No errors

**Step 3: Commit**

```bash
cd logos && git add infra/
git commit -m "infra: add Redis service to all docker-compose test stacks"
```

---

### Task 6: Bump foundry version and tag

**Step 1: Bump version**

Update `pyproject.toml`: change `version = "0.6.0"` to `version = "0.7.0"`

**Step 2: Run full test suite**

Run: `cd logos && poetry run pytest tests/logos_config/ tests/logos_events/ -v`
Expected: ALL PASS

**Step 3: Commit and tag**

```bash
cd logos && git add pyproject.toml
git commit -m "chore: bump foundry version to v0.7.0"
git tag v0.7.0
git push origin main --tags
```

---

## Phase 2: sophia (after logos v0.7.0 is tagged and pushed)

### Task 7: Update sophia to logos v0.7.0

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update foundry dependency**

Change `tag = "v0.6.0"` to `tag = "v0.7.0"` for the `logos-foundry` dependency.

**Step 2: Lock and install**

Run: `cd sophia && poetry lock --no-update && poetry install`
Expected: Resolves with logos-foundry v0.7.0

**Step 3: Run existing tests**

Run: `cd sophia && poetry run pytest tests/ -x --no-infra -q`
Expected: ALL PASS (no behavioral change yet)

**Step 4: Commit**

```bash
cd sophia && git add pyproject.toml poetry.lock
git commit -m "chore: bump logos-foundry to v0.7.0"
```

---

### Task 8: Refactor sophia to use RedisConfig

**Files:**
- Modify: `src/sophia/feedback/config.py`
- Modify: `src/sophia/api/app.py`
- Modify: `src/sophia/feedback/proposal_queue.py`
- Modify: `src/sophia/feedback/queue.py`

**Step 1: Simplify FeedbackConfig**

In `src/sophia/feedback/config.py`, replace `redis_url` with a reference to `RedisConfig`:

```python
"""Configuration for feedback emission system."""

from pydantic import Field
from pydantic_settings import BaseSettings

from logos_config import RedisConfig


class FeedbackConfig(BaseSettings):
    """Configuration for feedback emission."""

    enabled: bool = Field(
        default=True,
        description="Enable/disable feedback emission",
    )
    redis: RedisConfig = Field(default_factory=RedisConfig)
    hermes_url: str = Field(
        default="http://localhost:18000",
        description="Hermes base URL",
    )
    worker_timeout: float = Field(
        default=10.0,
        description="HTTP timeout for Hermes requests",
    )

    model_config = {"env_prefix": "SOPHIA_FEEDBACK_"}
```

**Step 2: Update app.py to use RedisConfig**

In `src/sophia/api/app.py`, update the startup code that creates Redis-backed queues. Change:

```python
feedback_queue = FeedbackQueue(feedback_config.redis_url)
```
to:
```python
feedback_queue = FeedbackQueue(feedback_config.redis.url)
```

And change:
```python
proposal_queue = ProposalQueue(feedback_config.redis_url)
```
to:
```python
proposal_queue = ProposalQueue(feedback_config.redis.url)
```

**Remove the graceful fallback try/except** around Redis initialization. The block around lines 275-294 that catches `RedisConnectionError` and falls back to `FeedbackDispatcher(None, enabled=False)` — replace it so Redis failure is a hard error (let the exception propagate, or log and re-raise).

**Step 3: Run tests**

Run: `cd sophia && poetry run pytest tests/ -x --no-infra -q`
Expected: ALL PASS

**Step 4: Commit**

```bash
cd sophia && git add src/sophia/feedback/config.py src/sophia/api/app.py
git commit -m "refactor: use RedisConfig from logos_config, remove Redis fallback"
```

---

### Task 9: Update sophia docker-compose

**Files:**
- Modify: `sophia/containers/docker-compose.test.yml`

**Step 1: Update Redis service**

Change the existing Redis service (lines 116-130) to use port `6379:6379` instead of `46379:6379`. Add a volume for persistence consistency. The container name can stay `sophia-test-redis` or change to match the logos infra naming — either way, the port should be the standard 6379.

**Step 2: Verify compose parses**

Run: `cd sophia && docker compose -f containers/docker-compose.test.yml config --quiet`
Expected: No errors

**Step 3: Run tests with infra**

Run: `cd sophia && docker compose -f containers/docker-compose.test.yml up -d && poetry run pytest tests/ -x -q`
Expected: ALL PASS

**Step 4: Commit**

```bash
cd sophia && git add containers/docker-compose.test.yml
git commit -m "infra: update Redis to standard port 6379"
```

---

## Phase 3: hermes (after logos v0.7.0 is tagged and pushed)

### Task 10: Update hermes to logos v0.7.0

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update foundry dependency**

Change `tag = "v0.6.0"` to `tag = "v0.7.0"` for logos-foundry (appears twice in hermes pyproject.toml — both the PEP 508 line and the poetry source).

**Step 2: Lock and install**

Run: `cd hermes && poetry lock --no-update && poetry install`
Expected: Resolves with logos-foundry v0.7.0

**Step 3: Commit**

```bash
cd hermes && git add pyproject.toml poetry.lock
git commit -m "chore: bump logos-foundry to v0.7.0"
```

---

### Task 11: Refactor hermes ContextCache to use RedisConfig

**Files:**
- Modify: `src/hermes/context_cache.py`
- Modify: `src/hermes/main.py`

**Step 1: Simplify ContextCache**

Replace the entire `__init__` method to remove graceful fallback. Remove the `_available` property. Redis is now required — if it's not up, let the exception propagate.

```python
"""Redis-backed cache for Sophia context and proposal queue."""

import json
import logging
import uuid
from datetime import UTC, datetime

import redis

from logos_config import RedisConfig

logger = logging.getLogger(__name__)


class ContextCache:
    """Redis-backed cache for Sophia context and proposal queue."""

    QUEUE_KEY = "sophia:proposals:pending"
    CONTEXT_PREFIX = "sophia:context:"

    def __init__(self, redis_config: RedisConfig) -> None:
        self._redis = redis.from_url(redis_config.url)
        self._redis.ping()
        logger.info("Context cache connected to Redis")

    def get_context(self, conversation_id: str) -> list[dict]:
        """Return cached Sophia context for *conversation_id*, or ``[]``."""
        raw = self._redis.get(f"{self.CONTEXT_PREFIX}{conversation_id}")
        if raw:
            result: list[dict] = json.loads(raw)
            return result
        return []

    def enqueue_proposal(
        self, proposal: dict, conversation_id: str | None = None
    ) -> None:
        """Push a proposal onto the Redis queue for background Sophia processing."""
        message = json.dumps(
            {
                "id": f"pq-{uuid.uuid4()}",
                "payload": proposal,
                "conversation_id": conversation_id,
                "attempts": 0,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        self._redis.lpush(self.QUEUE_KEY, message)
```

**Step 2: Update main.py**

Change `_get_context_cache()` in `src/hermes/main.py` (~line 159):

```python
def _get_context_cache() -> ContextCache:
    """Return (and lazily create) the module-level ContextCache."""
    global _context_cache
    if _context_cache is None:
        from logos_config import RedisConfig
        _context_cache = ContextCache(RedisConfig())
    return _context_cache
```

Update the type annotation of `_context_cache` from `ContextCache | None` to `ContextCache | None` (keep it nullable for lazy init, but the function now always returns a `ContextCache`, never `None`).

Update `_get_sophia_context()` to remove the `cache.available` checks — just call `cache.get_context()` directly. Remove the `if cache is not None and cache.available:` guard.

**Step 3: Run tests**

Run: `cd hermes && poetry run pytest tests/ -x --no-infra -q`
Expected: ALL PASS

**Step 4: Commit**

```bash
cd hermes && git add src/hermes/context_cache.py src/hermes/main.py
git commit -m "refactor: use RedisConfig, remove Redis graceful fallback"
```

---

## Verification Checklist

After all phases:

- [ ] `cd logos && poetry run pytest tests/logos_config/ tests/logos_events/ -v` — ALL PASS
- [ ] `cd sophia && poetry run pytest tests/ -x -q` — ALL PASS
- [ ] `cd hermes && poetry run pytest tests/ -x -q` — ALL PASS
- [ ] `docker compose -f logos/infra/test_stack/docker-compose.test.yml up -d` starts Redis alongside Neo4j/Milvus
- [ ] Sophia starts and connects to Redis without fallback warnings
- [ ] Hermes starts and connects to Redis without fallback warnings
- [ ] Proposal queue/dequeue cycle works end-to-end
- [ ] Context cache read/write works end-to-end
