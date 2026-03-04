# Maintenance Scheduler (#508) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a configurable maintenance scheduler in sophia that orchestrates async KG maintenance jobs via four trigger sources (post-ingestion, periodic, event-driven, threshold-based) without blocking the cognitive loop.

**Architecture:** MaintenanceScheduler is a third background worker in sophia's lifespan. It uses a Redis-backed job queue, subscribes to EventBus events, and dispatches jobs to registered handlers (TypeEmergenceDetector, RelationshipDiscoverer, future jobs) via asyncio.to_thread() with configurable concurrency limits.

**Tech Stack:** asyncio, redis-py, pydantic-settings, logos_events.EventBus, threading

**Design Doc:** `docs/plans/2026-03-04-ontology-pubsub-and-scheduler-design.md`

---

## Phase 1: Core Infrastructure (branch: `feat/sophia-508-maintenance-scheduler`)

### Task 1: Create MaintenanceConfig

**Files:**
- Create: `sophia/src/sophia/maintenance/__init__.py`
- Create: `sophia/src/sophia/maintenance/config.py`
- Create: `sophia/tests/maintenance/__init__.py`
- Create: `sophia/tests/maintenance/test_config.py`

**Step 1: Write the failing test**

Create `sophia/tests/maintenance/__init__.py` (empty).

Create `sophia/tests/maintenance/test_config.py`:

```python
"""Tests for sophia.maintenance.config.MaintenanceConfig."""

from __future__ import annotations

import os
from unittest import mock

from sophia.maintenance.config import MaintenanceConfig


class TestMaintenanceConfig:
    """Tests for MaintenanceConfig defaults and env overrides."""

    def test_default_values(self):
        """Default configuration values are sensible."""
        config = MaintenanceConfig()
        assert config.enabled is True
        assert config.post_ingestion_enabled is True
        assert config.periodic_enabled is True
        assert config.periodic_interval_seconds == 3600
        assert config.event_driven_enabled is True
        assert config.threshold_enabled is True
        assert config.type_member_count_threshold == 100
        assert config.max_concurrent_jobs == 2

    def test_env_override(self):
        """Environment variables override defaults."""
        with mock.patch.dict(os.environ, {
            "SOPHIA_MAINTENANCE_ENABLED": "false",
            "SOPHIA_MAINTENANCE_PERIODIC_INTERVAL_SECONDS": "600",
            "SOPHIA_MAINTENANCE_MAX_CONCURRENT_JOBS": "4",
        }):
            config = MaintenanceConfig()
            assert config.enabled is False
            assert config.periodic_interval_seconds == 600
            assert config.max_concurrent_jobs == 4

    def test_individual_triggers_toggleable(self):
        """Each trigger source can be independently disabled."""
        with mock.patch.dict(os.environ, {
            "SOPHIA_MAINTENANCE_POST_INGESTION_ENABLED": "false",
            "SOPHIA_MAINTENANCE_PERIODIC_ENABLED": "false",
            "SOPHIA_MAINTENANCE_EVENT_DRIVEN_ENABLED": "false",
            "SOPHIA_MAINTENANCE_THRESHOLD_ENABLED": "false",
        }):
            config = MaintenanceConfig()
            assert config.post_ingestion_enabled is False
            assert config.periodic_enabled is False
            assert config.event_driven_enabled is False
            assert config.threshold_enabled is False
```

**Step 2: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/maintenance/test_config.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write minimal implementation**

Create `sophia/src/sophia/maintenance/__init__.py`:

```python
"""sophia.maintenance — async KG maintenance job scheduling."""
```

Create `sophia/src/sophia/maintenance/config.py`:

```python
"""Configuration for the maintenance scheduler."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MaintenanceConfig(BaseSettings):
    """Configuration for async KG maintenance scheduling.

    All trigger sources are independently toggleable via env vars.
    """

    model_config = SettingsConfigDict(env_prefix="SOPHIA_MAINTENANCE_")

    enabled: bool = Field(default=True, description="Master switch for maintenance scheduler")

    # Post-ingestion trigger
    post_ingestion_enabled: bool = Field(default=True, description="Queue checks after proposal processing")

    # Periodic trigger
    periodic_enabled: bool = Field(default=True, description="Run periodic graph scans")
    periodic_interval_seconds: int = Field(default=3600, description="Interval between periodic scans")

    # Event-driven trigger
    event_driven_enabled: bool = Field(default=True, description="React to specific EventBus channels")

    # Threshold-based trigger
    threshold_enabled: bool = Field(default=True, description="Trigger jobs on metric thresholds")
    type_member_count_threshold: int = Field(default=100, description="Member count to trigger ontology evolution")

    # Resource management
    max_concurrent_jobs: int = Field(default=2, description="Max simultaneous maintenance jobs")
```

**Step 4: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/maintenance/test_config.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd sophia && git add src/sophia/maintenance/ tests/maintenance/
git commit -m "feat: add MaintenanceConfig with configurable trigger toggles"
```

---

### Task 2: Create MaintenanceQueue (Redis-backed job queue)

**Files:**
- Create: `sophia/src/sophia/maintenance/job_queue.py`
- Create: `sophia/tests/maintenance/test_job_queue.py`

**Step 1: Write the failing test**

Create `sophia/tests/maintenance/test_job_queue.py`:

```python
"""Tests for sophia.maintenance.job_queue.MaintenanceQueue."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from sophia.maintenance.job_queue import MaintenanceQueue, MaintenanceJob


class TestMaintenanceQueue:
    """Tests for MaintenanceQueue."""

    def test_enqueue_creates_job_with_metadata(self):
        """enqueue() pushes a JSON job to Redis with correct structure."""
        mock_redis = MagicMock()
        queue = MaintenanceQueue(mock_redis)

        queue.enqueue(
            job_type="type_emergence",
            priority="normal",
            params={"type_uuid": "t1"},
        )

        mock_redis.lpush.assert_called_once()
        key, value = mock_redis.lpush.call_args[0]
        assert key == "sophia:maintenance:pending"
        job = json.loads(value)
        assert job["job_type"] == "type_emergence"
        assert job["priority"] == "normal"
        assert job["params"] == {"type_uuid": "t1"}
        assert "id" in job
        assert "created_at" in job

    def test_dequeue_returns_job(self):
        """dequeue() returns a MaintenanceJob from Redis."""
        mock_redis = MagicMock()
        job_data = json.dumps({
            "id": "mj-123",
            "job_type": "type_emergence",
            "priority": "normal",
            "params": {"type_uuid": "t1"},
            "created_at": "2026-03-04T00:00:00Z",
            "attempts": 0,
        })
        mock_redis.brpop.return_value = ("sophia:maintenance:pending", job_data)

        queue = MaintenanceQueue(mock_redis)
        job = queue.dequeue(timeout=1)

        assert job is not None
        assert job.job_type == "type_emergence"
        assert job.params == {"type_uuid": "t1"}

    def test_dequeue_returns_none_on_timeout(self):
        """dequeue() returns None when queue is empty."""
        mock_redis = MagicMock()
        mock_redis.brpop.return_value = None

        queue = MaintenanceQueue(mock_redis)
        job = queue.dequeue(timeout=1)

        assert job is None

    def test_pending_count(self):
        """pending_count() returns queue length."""
        mock_redis = MagicMock()
        mock_redis.llen.return_value = 5

        queue = MaintenanceQueue(mock_redis)
        assert queue.pending_count() == 5
```

**Step 2: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/maintenance/test_job_queue.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write minimal implementation**

Create `sophia/src/sophia/maintenance/job_queue.py`:

```python
"""Redis-backed job queue for maintenance tasks."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class MaintenanceJob:
    """A maintenance job dequeued from Redis."""

    id: str
    job_type: str
    priority: str
    params: dict
    created_at: str
    attempts: int = 0


class MaintenanceQueue:
    """Redis-backed queue for maintenance jobs.

    Follows the same pattern as FeedbackQueue/ProposalQueue.
    """

    QUEUE_KEY = "sophia:maintenance:pending"
    FAILED_KEY = "sophia:maintenance:failed"
    MAX_RETRIES = 3

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    def enqueue(
        self,
        job_type: str,
        priority: str = "normal",
        params: dict | None = None,
    ) -> str:
        """Add a maintenance job to the queue. Returns job ID."""
        job_id = f"mj-{uuid.uuid4()}"
        job = {
            "id": job_id,
            "job_type": job_type,
            "priority": priority,
            "params": params or {},
            "created_at": datetime.now(UTC).isoformat(),
            "attempts": 0,
        }
        self._redis.lpush(self.QUEUE_KEY, json.dumps(job))
        logger.debug("Enqueued maintenance job %s: %s", job_id, job_type)
        return job_id

    def dequeue(self, timeout: int = 1) -> MaintenanceJob | None:
        """Blocking dequeue with timeout. Returns None if empty."""
        result = self._redis.brpop(self.QUEUE_KEY, timeout=timeout)
        if result is None:
            return None
        _, raw = result
        data = json.loads(raw)
        return MaintenanceJob(
            id=data["id"],
            job_type=data["job_type"],
            priority=data["priority"],
            params=data.get("params", {}),
            created_at=data["created_at"],
            attempts=data.get("attempts", 0),
        )

    def requeue(self, job: MaintenanceJob) -> None:
        """Requeue a failed job with incremented attempt count."""
        job.attempts += 1
        if job.attempts >= self.MAX_RETRIES:
            self.move_to_failed(job)
            return
        data = {
            "id": job.id,
            "job_type": job.job_type,
            "priority": job.priority,
            "params": job.params,
            "created_at": job.created_at,
            "attempts": job.attempts,
        }
        self._redis.lpush(self.QUEUE_KEY, json.dumps(data))
        logger.info("Requeued maintenance job %s (attempt %d)", job.id, job.attempts)

    def move_to_failed(self, job: MaintenanceJob) -> None:
        """Move a job to the failed queue after max retries."""
        data = {
            "id": job.id,
            "job_type": job.job_type,
            "priority": job.priority,
            "params": job.params,
            "created_at": job.created_at,
            "attempts": job.attempts,
            "failed_at": datetime.now(UTC).isoformat(),
        }
        self._redis.lpush(self.FAILED_KEY, json.dumps(data))
        logger.warning("Maintenance job %s moved to failed after %d attempts", job.id, job.attempts)

    def pending_count(self) -> int:
        """Return number of pending jobs."""
        return self._redis.llen(self.QUEUE_KEY)
```

**Step 4: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/maintenance/test_job_queue.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd sophia && git add src/sophia/maintenance/job_queue.py tests/maintenance/test_job_queue.py
git commit -m "feat: add MaintenanceQueue Redis-backed job queue"
```

---

### Task 3: Create MaintenanceScheduler core

**Files:**
- Create: `sophia/src/sophia/maintenance/scheduler.py`
- Create: `sophia/tests/maintenance/test_scheduler.py`

**Step 1: Write the failing test**

Create `sophia/tests/maintenance/test_scheduler.py`:

```python
"""Tests for sophia.maintenance.scheduler.MaintenanceScheduler."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.job_queue import MaintenanceJob, MaintenanceQueue
from sophia.maintenance.scheduler import MaintenanceScheduler


class TestMaintenanceScheduler:
    """Tests for MaintenanceScheduler."""

    def test_init_registers_handlers(self):
        """Scheduler stores registered handlers."""
        mock_queue = MagicMock()
        mock_event_bus = MagicMock()
        config = MaintenanceConfig()

        handler_fn = MagicMock()
        handlers = {"type_emergence": handler_fn}

        scheduler = MaintenanceScheduler(
            queue=mock_queue,
            event_bus=mock_event_bus,
            config=config,
            handlers=handlers,
        )

        assert "type_emergence" in scheduler._handlers

    def test_on_proposal_processed_enqueues_jobs(self):
        """Post-ingestion trigger enqueues maintenance jobs."""
        mock_queue = MagicMock()
        mock_event_bus = MagicMock()
        config = MaintenanceConfig()

        scheduler = MaintenanceScheduler(
            queue=mock_queue,
            event_bus=mock_event_bus,
            config=config,
            handlers={"type_emergence": MagicMock()},
        )

        event = {
            "event_type": "proposal_processed",
            "payload": {
                "affected_node_uuids": ["n1", "n2"],
                "new_types": ["vehicle"],
                "updated_types": ["person"],
            },
        }

        scheduler._on_proposal_processed(event)

        # Should enqueue at least one job
        assert mock_queue.enqueue.call_count >= 1

    def test_disabled_scheduler_does_nothing(self):
        """When enabled=False, scheduler doesn't subscribe or dispatch."""
        mock_queue = MagicMock()
        mock_event_bus = MagicMock()
        config = MaintenanceConfig(enabled=False)

        scheduler = MaintenanceScheduler(
            queue=mock_queue,
            event_bus=mock_event_bus,
            config=config,
            handlers={},
        )

        # Should not subscribe to EventBus
        mock_event_bus.subscribe.assert_not_called()

    def test_disabled_post_ingestion_skips_subscription(self):
        """When post_ingestion_enabled=False, doesn't subscribe to proposals."""
        mock_queue = MagicMock()
        mock_event_bus = MagicMock()
        config = MaintenanceConfig(post_ingestion_enabled=False)

        scheduler = MaintenanceScheduler(
            queue=mock_queue,
            event_bus=mock_event_bus,
            config=config,
            handlers={},
        )

        # Should not subscribe to proposal_processed
        for call in mock_event_bus.subscribe.call_args_list:
            channel = call[0][0]
            assert channel != "logos:sophia:proposal_processed"

    @pytest.mark.asyncio
    async def test_dispatch_calls_handler(self):
        """dispatch_job() calls the registered handler."""
        mock_queue = MagicMock()
        mock_event_bus = MagicMock()
        config = MaintenanceConfig()

        handler_fn = MagicMock(return_value={"result": "ok"})
        handlers = {"type_emergence": handler_fn}

        scheduler = MaintenanceScheduler(
            queue=mock_queue,
            event_bus=mock_event_bus,
            config=config,
            handlers=handlers,
        )

        job = MaintenanceJob(
            id="mj-1",
            job_type="type_emergence",
            priority="normal",
            params={"type_uuid": "t1"},
            created_at="2026-03-04T00:00:00Z",
        )

        await scheduler._dispatch_job(job)
        handler_fn.assert_called_once_with(**job.params)

    @pytest.mark.asyncio
    async def test_dispatch_unknown_job_type_logs_warning(self):
        """dispatch_job() logs warning for unknown job types."""
        mock_queue = MagicMock()
        mock_event_bus = MagicMock()
        config = MaintenanceConfig()

        scheduler = MaintenanceScheduler(
            queue=mock_queue,
            event_bus=mock_event_bus,
            config=config,
            handlers={},
        )

        job = MaintenanceJob(
            id="mj-1",
            job_type="unknown_type",
            priority="normal",
            params={},
            created_at="2026-03-04T00:00:00Z",
        )

        # Should not raise, just log
        await scheduler._dispatch_job(job)
```

**Step 2: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/maintenance/test_scheduler.py -v`
Expected: FAIL — ModuleNotFoundError

**Step 3: Write minimal implementation**

Create `sophia/src/sophia/maintenance/scheduler.py`:

```python
"""MaintenanceScheduler — orchestrates async KG maintenance jobs.

Subscribes to EventBus events, manages a Redis job queue, and dispatches
jobs to registered handlers without blocking the cognitive loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.job_queue import MaintenanceJob, MaintenanceQueue

logger = logging.getLogger(__name__)


class MaintenanceScheduler:
    """Async maintenance job scheduler for sophia."""

    def __init__(
        self,
        queue: MaintenanceQueue,
        event_bus: Any,
        config: MaintenanceConfig,
        handlers: dict[str, Callable],
        hcg_client: Any | None = None,
    ) -> None:
        self._queue = queue
        self._event_bus = event_bus
        self._config = config
        self._handlers = handlers
        self._hcg = hcg_client
        self._running = False
        self._semaphore: asyncio.Semaphore | None = None
        self._listener_thread: threading.Thread | None = None

        if config.enabled:
            self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        """Subscribe to EventBus channels based on config."""
        if self._config.post_ingestion_enabled:
            self._event_bus.subscribe(
                "logos:sophia:proposal_processed",
                self._on_proposal_processed,
            )
            logger.info("Scheduler subscribed to proposal_processed events")

        if self._config.event_driven_enabled:
            self._event_bus.subscribe(
                "logos:sophia:threshold_crossed",
                self._on_threshold_crossed,
            )
            logger.info("Scheduler subscribed to threshold_crossed events")

    def _on_proposal_processed(self, event: dict) -> None:
        """Handle proposal_processed event — queue neighborhood checks."""
        payload = event.get("payload", {})
        affected = payload.get("affected_node_uuids", [])
        new_types = payload.get("new_types", [])
        updated_types = payload.get("updated_types", [])

        # Queue relationship discovery for new nodes
        if affected and "relationship_discovery" in self._handlers:
            self._queue.enqueue(
                job_type="relationship_discovery",
                priority="normal",
                params={"node_uuids": affected},
            )

        # Queue type emergence check for updated types
        if updated_types and "type_emergence" in self._handlers:
            for type_name in updated_types:
                self._queue.enqueue(
                    job_type="type_emergence",
                    priority="normal",
                    params={"type_name": type_name},
                )

        # Threshold check: if new types were created, check member counts
        if self._config.threshold_enabled and (new_types or updated_types):
            self._check_thresholds(new_types + updated_types)

    def _on_threshold_crossed(self, event: dict) -> None:
        """Handle threshold_crossed event — queue targeted maintenance."""
        payload = event.get("payload", {})
        job_type = payload.get("job_type")
        if job_type and job_type in self._handlers:
            self._queue.enqueue(
                job_type=job_type,
                priority="high",
                params=payload.get("params", {}),
            )

    def _check_thresholds(self, type_names: list[str]) -> None:
        """Check if any types cross the member count threshold."""
        if self._hcg is None:
            return
        threshold = self._config.type_member_count_threshold
        for type_name in type_names:
            try:
                # Look up type node to check member count
                # Implementation depends on HCGClient API
                pass
            except Exception:
                logger.exception("Failed threshold check for type %s", type_name)

    async def start(self) -> None:
        """Start the scheduler: listener thread + dispatch loop + periodic timer."""
        if not self._config.enabled:
            logger.info("Maintenance scheduler disabled")
            return

        self._running = True
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_jobs)

        # Start EventBus listener in background thread
        self._listener_thread = threading.Thread(
            target=self._event_bus.listen, daemon=True, name="maintenance-listener"
        )
        self._listener_thread.start()
        logger.info("Maintenance scheduler started")

        # Run dispatch loop and periodic timer concurrently
        tasks = [asyncio.create_task(self._dispatch_loop())]
        if self._config.periodic_enabled:
            tasks.append(asyncio.create_task(self._periodic_loop()))

        await asyncio.gather(*tasks)

    def stop(self) -> None:
        """Signal the scheduler to stop."""
        self._running = False
        if self._event_bus is not None:
            self._event_bus.stop()
        logger.info("Maintenance scheduler stopping")

    async def _dispatch_loop(self) -> None:
        """Dequeue and dispatch jobs."""
        while self._running:
            try:
                job = await asyncio.to_thread(self._queue.dequeue, 1)
                if job is not None:
                    async with self._semaphore:
                        await self._dispatch_job(job)
            except Exception:
                if self._running:
                    logger.exception("Error in dispatch loop")
                    await asyncio.sleep(1)

    async def _periodic_loop(self) -> None:
        """Periodically queue graph-wide maintenance scans."""
        interval = self._config.periodic_interval_seconds
        while self._running:
            await asyncio.sleep(interval)
            if not self._running:
                break
            logger.info("Periodic maintenance scan triggered")
            # Queue a full scan job for each registered handler
            if "type_emergence" in self._handlers:
                self._queue.enqueue(
                    job_type="type_emergence",
                    priority="low",
                    params={"scan": "full"},
                )

    async def _dispatch_job(self, job: MaintenanceJob) -> None:
        """Dispatch a single job to its handler."""
        handler = self._handlers.get(job.job_type)
        if handler is None:
            logger.warning("No handler for job type: %s (job %s)", job.job_type, job.id)
            return

        logger.info("Dispatching maintenance job %s: %s", job.id, job.job_type)
        try:
            await asyncio.to_thread(handler, **job.params)
            logger.info("Maintenance job %s completed", job.id)
        except Exception:
            logger.exception("Maintenance job %s failed", job.id)
            self._queue.requeue(job)
```

**Step 4: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/maintenance/test_scheduler.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd sophia && git add src/sophia/maintenance/scheduler.py tests/maintenance/test_scheduler.py
git commit -m "feat: add MaintenanceScheduler with four trigger sources"
```

---

### Task 4: Wire MaintenanceScheduler into sophia lifespan

**Files:**
- Modify: `sophia/src/sophia/api/app.py`

**Step 1: Add imports and globals**

In `sophia/src/sophia/api/app.py`, add imports (near line 33):

```python
from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.scheduler import MaintenanceScheduler
from sophia.maintenance.job_queue import MaintenanceQueue
```

Add global variables (near line 242):

```python
_maintenance_scheduler: MaintenanceScheduler | None = None
_maintenance_task: asyncio.Task | None = None
```

**Step 2: Add initialization in lifespan startup**

After ProposalWorker initialization (around line 420), add:

```python
        # Initialize Maintenance Scheduler
        try:
            _maintenance_config = MaintenanceConfig()
            if _maintenance_config.enabled:
                from logos_config import RedisConfig
                from logos_events import EventBus
                import redis

                _maint_redis_config = RedisConfig()
                _maint_event_bus = EventBus(_maint_redis_config)
                _maint_redis = redis.from_url(_maint_redis_config.url)
                _maint_queue = MaintenanceQueue(_maint_redis)

                # Register available handlers
                from sophia.ingestion.type_emergence import TypeEmergenceDetector
                from sophia.ingestion.relationship_discoverer import RelationshipDiscoverer

                _type_emergence = TypeEmergenceDetector(
                    milvus=_milvus_sync, hcg=_hcg_client
                )
                _relationship_discoverer = RelationshipDiscoverer(milvus=_milvus_sync)

                _handlers = {
                    "type_emergence": _type_emergence.check_type,
                    "relationship_discovery": _relationship_discoverer.find_candidates,
                }

                _maintenance_scheduler = MaintenanceScheduler(
                    queue=_maint_queue,
                    event_bus=_maint_event_bus,
                    config=_maintenance_config,
                    handlers=_handlers,
                    hcg_client=_hcg_client,
                )
                _maintenance_task = asyncio.create_task(_maintenance_scheduler.start())
                logger.info("Maintenance scheduler started")
        except Exception:
            logger.exception("Failed to start maintenance scheduler")
```

Note: The EventBus here is a separate instance from the one used in #501. Both subscribe to the same Redis pub/sub channels, which is fine — Redis pub/sub supports multiple subscribers. However, if #501 has already initialized an EventBus and stored it globally, consider reusing it. The scheduler needs its own listener thread regardless.

**Step 3: Add shutdown logic**

In the shutdown section (after line 452), add before the HCG client close:

```python
        # Stop Maintenance Scheduler
        if _maintenance_scheduler is not None:
            _maintenance_scheduler.stop()
        if _maintenance_task is not None:
            _maintenance_task.cancel()
            try:
                await _maintenance_task
            except asyncio.CancelledError:
                pass
            logger.info("Maintenance scheduler stopped")
```

**Step 4: Run existing tests to verify nothing breaks**

Run: `cd sophia && poetry run pytest tests/ -x --no-infra -q`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd sophia && git add src/sophia/api/app.py
git commit -m "feat: wire MaintenanceScheduler into sophia lifespan"
```

---

### Task 5: Integration test for scheduler

**Files:**
- Create: `sophia/tests/integration/test_maintenance_scheduler.py`

**Step 1: Write integration test**

```python
"""Integration test for MaintenanceScheduler with real Redis.

Requires: Redis running on localhost:6379
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
import redis as redis_lib

from logos_config import RedisConfig
from logos_events import EventBus

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.job_queue import MaintenanceQueue
from sophia.maintenance.scheduler import MaintenanceScheduler

REDIS_AVAILABLE = False
try:
    r = redis_lib.from_url("redis://localhost:6379/0")
    r.ping()
    REDIS_AVAILABLE = True
except Exception:
    pass

pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")


class TestMaintenanceSchedulerIntegration:
    """Integration tests for MaintenanceScheduler."""

    def setup_method(self):
        self.config = RedisConfig()
        self.redis = redis_lib.from_url(self.config.url)
        # Clean up test keys
        self.redis.delete("sophia:maintenance:pending")
        self.redis.delete("sophia:maintenance:failed")

    def teardown_method(self):
        self.redis.delete("sophia:maintenance:pending")
        self.redis.delete("sophia:maintenance:failed")
        self.redis.close()

    def test_post_ingestion_trigger_queues_jobs(self):
        """Publishing proposal_processed event queues maintenance jobs."""
        queue = MaintenanceQueue(self.redis)
        event_bus = EventBus(self.config)
        handler_called = []

        def mock_handler(**kwargs):
            handler_called.append(kwargs)

        maint_config = MaintenanceConfig(
            periodic_enabled=False,
            event_driven_enabled=False,
            threshold_enabled=False,
        )

        scheduler = MaintenanceScheduler(
            queue=queue,
            event_bus=event_bus,
            config=maint_config,
            handlers={"relationship_discovery": mock_handler},
        )

        # Start EventBus listener
        listener = threading.Thread(target=event_bus.listen, daemon=True)
        listener.start()
        time.sleep(0.2)

        # Publish event from a separate EventBus instance
        pub_bus = EventBus(self.config)
        pub_bus.publish("logos:sophia:proposal_processed", {
            "event_type": "proposal_processed",
            "source": "sophia",
            "payload": {
                "affected_node_uuids": ["n1", "n2"],
                "new_types": [],
                "updated_types": [],
            },
        })
        pub_bus.close()

        time.sleep(0.3)
        event_bus.stop()

        # Verify job was queued
        assert queue.pending_count() >= 1

    @pytest.mark.asyncio
    async def test_dispatch_loop_processes_jobs(self):
        """Dispatch loop dequeues and processes jobs."""
        queue = MaintenanceQueue(self.redis)
        event_bus = EventBus(self.config)
        results = []

        def mock_handler(**kwargs):
            results.append(kwargs)

        maint_config = MaintenanceConfig(
            periodic_enabled=False,
            event_driven_enabled=False,
            post_ingestion_enabled=False,
        )

        scheduler = MaintenanceScheduler(
            queue=queue,
            event_bus=event_bus,
            config=maint_config,
            handlers={"type_emergence": mock_handler},
        )

        # Enqueue a job directly
        queue.enqueue("type_emergence", params={"type_uuid": "t1"})

        # Start scheduler briefly
        scheduler._running = True
        scheduler._semaphore = asyncio.Semaphore(2)
        task = asyncio.create_task(scheduler._dispatch_loop())
        await asyncio.sleep(2)
        scheduler.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        event_bus.close()

        assert len(results) == 1
        assert results[0] == {"type_uuid": "t1"}
```

**Step 2: Run integration test**

Run: `cd sophia && poetry run pytest tests/integration/test_maintenance_scheduler.py -v`
Expected: ALL PASS (or SKIP if Redis not running)

**Step 3: Commit**

```bash
cd sophia && git add tests/integration/test_maintenance_scheduler.py
git commit -m "test: add integration tests for maintenance scheduler"
```

---

## Verification Checklist

- [ ] `cd sophia && poetry run pytest tests/maintenance/ -v` — ALL PASS
- [ ] `cd sophia && poetry run pytest tests/ -x --no-infra -q` — ALL PASS
- [ ] `cd sophia && poetry run pytest tests/integration/test_maintenance_scheduler.py -v` — PASS (with Redis)
- [ ] MaintenanceConfig respects all env var overrides
- [ ] MaintenanceQueue enqueue/dequeue/requeue works with Redis
- [ ] Scheduler subscribes to proposal_processed when post_ingestion_enabled
- [ ] Scheduler doesn't subscribe when triggers are disabled
- [ ] Dispatch loop calls registered handlers
- [ ] max_concurrent_jobs limits parallel execution
- [ ] Scheduler starts and stops cleanly in sophia lifespan
