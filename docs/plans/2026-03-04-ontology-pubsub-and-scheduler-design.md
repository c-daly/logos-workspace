# Design: Ontology Pub/Sub Distribution (#501) & Maintenance Scheduler (#508)

**Date:** 2026-03-04
**Epic:** logos #499 (KG Maintenance)
**Issues:** logos #501, logos #508
**Depends on:** logos #500 (complete — EventBus, RedisConfig in foundry v0.7.0)

---

## Overview

Two features built in parallel on top of the EventBus infrastructure:

1. **#501**: Sophia publishes batch events after proposal processing. Hermes subscribes and maintains a live TypeRegistry for NER classification.
2. **#508**: A MaintenanceScheduler in sophia orchestrates async maintenance jobs (entity resolution, type correction, etc.) via four configurable trigger sources.

These are independent — #501 wires up event publishing and Hermes consumption, #508 wires up event subscription and job dispatch within sophia. Both use the same EventBus.

---

## #501: Ontology Pub/Sub Distribution

### Event Schema

One batch event per proposal, published after `ProposalProcessor.process()` completes:

- **Channel:** `logos:sophia:proposal_processed`
- **Event type:** `proposal_processed`
- **Payload:**

```json
{
  "proposal_id": "uuid",
  "new_types": ["person", "vehicle"],
  "updated_types": ["location"],
  "new_nodes": [{"uuid": "...", "name": "...", "type": "..."}],
  "new_edges": [{"uuid": "...", "relation": "...", "source": "...", "target": "..."}],
  "affected_node_uuids": ["uuid1", "uuid2"]
}
```

### Sophia Publisher

- `ProposalProcessor` receives an `EventBus` instance via constructor injection.
- After `process()` completes, it calls `EventBus.publish()` with the batch summary.
- After publishing, it writes the full type list to Redis key `logos:ontology:types` as a JSON snapshot (dict of type_name -> type properties including uuid, member_count, centroid).

### Hermes TypeRegistry

New class in hermes: `TypeRegistry`.

- **On boot:** reads `logos:ontology:types` from Redis for initial sync.
- **At runtime:** subscribes to `logos:sophia:proposal_processed`, updates internal state when new/updated types appear.
- **Interface:**
  - `get_type_names() -> list[str]`
  - `get_type(name: str) -> dict | None`
  - `get_edge_types() -> list[str]`
  - `format_for_prompt() -> str` — returns formatted type list for NER prompt injection
- **Thread safety:** internal `threading.Lock` around the type dict.
- **Background:** `EventBus.listen()` runs in a daemon thread, started during hermes init.

### Redis Key

- Key: `logos:ontology:types`
- Value: JSON dict `{type_name: {uuid, member_count, centroid, ...}}`
- Written by sophia after each proposal batch.
- Read by hermes on boot (initial sync).
- No TTL — always current as long as sophia is running.

---

## #508: Maintenance Scheduler

### Architecture

MaintenanceScheduler is a third background worker in sophia's lifespan, alongside FeedbackWorker and ProposalWorker.

### Four Trigger Sources (independently configurable)

1. **Post-ingestion** — subscribes to `logos:sophia:proposal_processed` via EventBus. Queues neighborhood checks for `affected_node_uuids`.
2. **Periodic** — configurable interval timer. Scans graph partitions for accumulated drift (type variance, orphan nodes, etc.).
3. **Event-driven** — subscribes to specific EventBus channels for targeted triggers (e.g. `logos:sophia:threshold_crossed`).
4. **Threshold-based** — checks metrics after post-ingestion events (e.g. type member_count exceeds threshold -> queue ontology evolution).

### MaintenanceConfig

Pydantic settings class:

```python
class MaintenanceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SOPHIA_MAINTENANCE_")

    enabled: bool = True

    # Post-ingestion
    post_ingestion_enabled: bool = True

    # Periodic
    periodic_enabled: bool = True
    periodic_interval_seconds: int = 3600  # 1 hour

    # Event-driven
    event_driven_enabled: bool = True

    # Threshold-based
    threshold_enabled: bool = True
    type_member_count_threshold: int = 100  # trigger ontology evolution

    # Resource management
    max_concurrent_jobs: int = 2
```

### Job Queue

- Redis-backed, key `sophia:maintenance:pending`.
- Follows FeedbackQueue/ProposalQueue pattern (brpop with timeout, asyncio.to_thread).
- Job payload includes: `job_type`, `priority`, `params` (e.g. affected UUIDs, type name), `created_at`.
- Priority levels: `critical` > `high` > `normal` > `low`.

### Job Handlers

Registered at startup as a `dict[str, Callable]`:

```python
handlers = {
    "type_emergence": type_emergence_detector.check,
    "relationship_discovery": relationship_discoverer.find_candidates,
    "entity_resolution": ...,   # future
    "type_correction": ...,     # future
}
```

TypeEmergenceDetector and RelationshipDiscoverer already exist in sophia but are unwired. The scheduler wires them in as handlers.

### Job Dispatch

- Worker loop dequeues from Redis, dispatches to handler via `asyncio.to_thread()`.
- Respects `max_concurrent_jobs` — uses `asyncio.Semaphore`.
- Failed jobs are requeued with exponential backoff (same pattern as FeedbackQueue).
- Observability: logs job start/complete/fail, tracks queue depth.

### Lifecycle

- Started as `asyncio.create_task()` in sophia's lifespan.
- `stop()` signals shutdown, awaits in-flight jobs.
- EventBus listener thread stopped and joined.

---

## File Changes Summary

### logos (foundry) — no changes needed
EventBus and RedisConfig already exist in v0.7.0.

### sophia (#501)
- Modify: `src/sophia/ingestion/proposal_processor.py` — accept EventBus, publish batch event + write type snapshot to Redis
- Modify: `src/sophia/api/app.py` — initialize EventBus, pass to ProposalProcessor

### sophia (#508)
- Create: `src/sophia/maintenance/__init__.py`
- Create: `src/sophia/maintenance/config.py` — MaintenanceConfig
- Create: `src/sophia/maintenance/scheduler.py` — MaintenanceScheduler
- Create: `src/sophia/maintenance/job_queue.py` — MaintenanceQueue (Redis-backed)
- Create: `tests/maintenance/` — tests for scheduler and queue
- Modify: `src/sophia/api/app.py` — initialize and start MaintenanceScheduler in lifespan

### hermes (#501)
- Create: `src/hermes/type_registry.py` — TypeRegistry class
- Modify: `src/hermes/main.py` — initialize TypeRegistry with EventBus, use in NER
- Create: `tests/test_type_registry.py` — tests

---

## Conflict Risk

Both #501 and #508 modify `sophia/src/sophia/api/app.py` (lifespan initialization). Minimal overlap — #501 adds EventBus init and passes it to ProposalProcessor, #508 adds MaintenanceScheduler init. Trivial merge if needed.
