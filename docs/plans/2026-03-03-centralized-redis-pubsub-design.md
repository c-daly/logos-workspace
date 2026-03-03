# Centralized Redis & Pub/Sub Infrastructure

**Date**: 2026-03-03
**Status**: Approved
**Related Issues**: logos #500, #501, #499 (parent epic)

## Context

Sophia and Hermes already use Redis for proposal queuing and context caching, but the infrastructure is ad-hoc:

- Redis container only exists in sophia's test docker-compose (`redis:7-alpine`, port 46379)
- No `RedisConfig` in `logos_config` — configuration is scattered across `FeedbackConfig` and `ContextCache` constructors
- Hermes has graceful fallback code (`_available` flags) treating Redis as optional
- No pub/sub usage anywhere — all current patterns are queue-based (LPUSH/BRPOP)
- Redis is not in the shared infrastructure stack alongside Neo4j and Milvus

The cognitive loop needs async maintenance jobs (entity resolution, type correction, ontology evolution — see #499). These jobs must run post-response, not inline with proposal processing, otherwise the loop gets slower with every new capability. Pub/sub is the mechanism for that async dispatch.

## Decision

**Redis becomes required shared infrastructure**, on par with Neo4j and Milvus. No graceful fallback — if Redis isn't up, the system doesn't start, same as the other infra services.

**Event bus lives in logos foundry**. A thin `EventBus` wrapper over Redis pub/sub, co-located with `RedisConfig` in the foundry packages. Both Sophia and Hermes import and use the same abstraction.

**Event-driven trigger model**. The proposal processor publishes events (e.g. `proposal_processed`). A future maintenance scheduler subscribes and decides what jobs to run. This keeps the proposal processor decoupled from maintenance job knowledge.

## Design

### 1. Infrastructure — Redis in the Shared Stack

Add a `redis` service to all docker-compose files in `logos/infra/`:

- **Image**: `redis:7-alpine`
- **Port**: `6379:6379` (standard default, consistent with Neo4j/Milvus using their defaults)
- **Healthcheck**: `redis-cli ping`
- **Named volume**: Optional, for persistence consistency with other services

Docker-compose files to update:
- `logos/infra/test_stack/docker-compose.test.yml` (shared test stack)
- `logos/infra/sophia/docker-compose.test.yml`
- `logos/infra/hermes/docker-compose.test.yml`
- `logos/infra/apollo/docker-compose.test.yml`
- `logos/infra/talos/docker-compose.test.yml`
- `sophia/containers/docker-compose.test.yml` (remove local Redis, use shared)

### 2. logos_config — RedisConfig

Add to `logos_config/settings.py`, following the `Neo4jConfig`/`MilvusConfig` pattern:

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

Add Redis port (6379) to `logos_config/ports.py` port table documentation.

### 3. Event Bus — `logos_events`

A new module in the logos foundry providing a thin pub/sub wrapper:

```python
class EventBus:
    """Redis pub/sub event bus for LOGOS services."""

    def __init__(self, redis_config: RedisConfig):
        self._redis = redis.from_url(redis_config.url)
        self._pubsub = self._redis.pubsub()

    def publish(self, channel: str, event: dict) -> None:
        """Publish an event to a channel."""
        self._redis.publish(channel, json.dumps({
            "event_type": event.get("event_type", "unknown"),
            "source": event.get("source", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": event.get("payload", {}),
        }))

    def subscribe(self, channel: str, callback: Callable) -> None:
        """Subscribe to a channel with a callback."""
        self._pubsub.subscribe(**{channel: callback})

    def listen(self) -> None:
        """Blocking listen loop. Run in a background thread."""
        for message in self._pubsub.listen():
            if message["type"] == "message":
                pass  # callbacks invoked by pubsub

    def close(self) -> None:
        self._pubsub.close()
        self._redis.close()
```

**Channel naming convention**: `logos:<service>:<event_type>`
- `logos:sophia:proposal_processed`
- `logos:sophia:ontology_changed`
- `logos:sophia:node_created`

**Event envelope**: Every event carries a standard header:
```json
{
    "event_type": "proposal_processed",
    "source": "sophia",
    "timestamp": "2026-03-03T12:00:00+00:00",
    "payload": { "...": "event-specific data" }
}
```

For this work: the `EventBus` class is built and tested. No events are published or subscribed to yet — that's future work (#501, #508).

### 4. Sophia Refactoring

- Replace `FeedbackConfig.redis_url` with `RedisConfig` from logos_config
- Remove graceful fallback in `ProposalQueue`, `FeedbackQueue`, `FeedbackDispatcher` initialization
- Update `sophia/containers/docker-compose.test.yml` to remove local Redis (use shared infra Redis)
- Verify: proposal queue/dequeue, context cache, feedback dispatch all work unchanged

### 5. Hermes Refactoring

- Refactor `ContextCache` to use `RedisConfig` from logos_config
- Remove `_available` flag and graceful degradation — Redis is required
- Verify: context caching works unchanged

## Execution Order

The dependency chain requires strict ordering:

1. **logos** — Add Redis to docker-compose files, add `RedisConfig` to `logos_config`, add `logos_events` with `EventBus`, tests, bump to v0.7.0, merge & distribute
2. **sophia** — Pick up logos v0.7.0, refactor to use `RedisConfig`, remove fallbacks, update docker-compose, verify
3. **hermes** — Pick up logos v0.7.0, refactor `ContextCache`, remove fallbacks, verify

## Verification

"Everything still works the same" means:
- Existing proposal queue/dequeue cycle works
- Context cache read/write works
- Feedback dispatch works
- All existing tests pass in all three repos
- No behavioral change — infrastructure moved, code simplified

## Non-Goals (This Work)

- No events are published or subscribed to yet
- No maintenance scheduler
- No ontology pub/sub distribution (that's #501)
- No new Hermes endpoints for maintenance queries

## Future Work

This infrastructure enables:
- **#501**: Ontology pub/sub distribution (Sophia publishes, Hermes subscribes)
- **#508**: Maintenance scheduler (subscribes to `proposal_processed`, dispatches jobs)
- **#503-507**: Individual maintenance jobs (entity resolution, type correction, ontology evolution, relationship inference, competing edges)
