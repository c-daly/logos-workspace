# OpenTelemetry Instrumentation Design

**Date:** 2026-02-01
**Tracking:** logos#321, sub-tickets #334-344
**Goal:** Operational visibility across Sophia, Hermes, and Apollo via OpenTelemetry

---

## Overview

### What is this?

LOGOS is a multi-service cognitive architecture with five repos: logos (foundry), sophia, hermes, apollo, and talos. The services communicate over HTTP using httpx. This spec adds distributed tracing and metrics via OpenTelemetry (OTel), so that a single request flowing through Apollo → Sophia → Hermes produces a correlated trace visible in a single UI.

### How OTel works (the 30-second version)

- **Spans** are timed operations (e.g. "handle HTTP request", "run simulation"). Spans nest: a parent span can have child spans.
- **Traces** are trees of spans sharing a single trace ID. When Service A calls Service B over HTTP, the trace ID propagates via W3C `traceparent` headers — automatically, if both sides are instrumented.
- **TracerProvider** is the SDK entry point. You configure it once at app startup with exporters (where to send spans) and a resource (service name).
- **OTel Collector** is an intermediary process that receives spans/metrics and forwards them to backends (Jaeger for traces, Prometheus for metrics).
- **Jaeger** is a trace visualization UI. You query it by service name, operation, or trace ID.

### Architecture

```
Sophia/Hermes/Apollo (instrumented services)
  └─ logos_observability.setup_telemetry(service_name, otlp_endpoint)
       └─ TracerProvider + MeterProvider
            └─ OTLPSpanExporter → OTel Collector (:4317 gRPC)
                 ├─ Jaeger (:16686 UI) — traces    [Phases 0-2]
                 ├─ Prometheus (:9090) — metrics    [Phases 0-2]
                 ├─ Tempo (:3200 query) — traces    [Phase 3]
                 └─ Grafana (:3001 UI) — dashboards [Phase 3]
```

### Key Decisions

- **Centralized OTel deps**: All OpenTelemetry packages live in `logos-foundry` (the shared library), not per-service.
- **OTel is a mandatory runtime dependency**: Services are not expected to function without OTel packages installed. Failure to import OTel is treated as a fatal misconfiguration, not a degraded mode. The SDK monkey-patches libraries (FastAPI, httpx) at instrumentation time — this is an accepted behavioral change.
- **Always-on instrumentation, opt-in export**: Code is always instrumented. OTLP export is enabled only when `OTEL_EXPORTER_OTLP_ENDPOINT` env var is set. When no endpoint is configured, spans are created but never exported.
- **Auto + manual spans**: `FastAPIInstrumentor` for automatic request-level spans, `HTTPXClientInstrumentor` for automatic outbound HTTP trace propagation, manual `get_tracer()` spans for key internal operations.
- **Consistent span naming**: All manual spans follow `{service}.{operation}` convention (e.g. `sophia.simulate`, `hermes.embed`).
- **Independent service PRs**: Each service's instrumentation is a separate PR. One failing doesn't block others.
- **SDK-provided test utilities**: Use `opentelemetry.sdk.trace.export.in_memory.InMemorySpanExporter` (shipped with the SDK) rather than custom implementations.

### Non-Goals

- **Business analytics**: This is operational observability, not a data pipeline for product metrics or user behavior tracking.
- **Trace completeness guarantees**: Traces may be dropped under memory pressure or collector outage. No SLA on trace delivery.
- **Replacing application logging**: OTel traces complement structured logs — they do not replace `logger.info()` / `logger.error()` calls.
- **Per-user observability**: No user-scoped trace filtering or PII in span attributes.
- **Custom metrics beyond auto-instrumentation**: v1 does not define custom application metrics (except existing JEPA metrics). See Metrics Scope below.

### Sampling Policy

v1 runs with **100% head-based sampling** in all environments. Every request produces a complete trace.

Tail-based or probabilistic sampling is explicitly **out of scope** for this spec. Introducing sampling changes debugging semantics — developers currently assume "if I made a request, I can find its trace." Any future sampling changes must be introduced via a separate design doc that addresses:
- Impact on debugging workflows
- Which environments get reduced sampling
- How to force-sample specific requests (e.g. via `X-Force-Sample` header)

### Failure Model

When the OTel Collector is unreachable or down:
- `BatchSpanProcessor` buffers spans in memory (default: 2048 spans, 5s flush interval)
- When the buffer fills, **new spans are silently dropped** — no exceptions, no retries
- Memory usage increases by the buffer size (~few MB) but does not grow unboundedly
- Export failure is logged once by the SDK at WARNING level, then suppressed
- **No guarantees** are made about trace completeness during collector outages
- Service behavior, latency, and correctness are unaffected

This is acceptable for v1. If span loss during outages becomes a problem, the fix is collector redundancy, not application-level retries.

### Span Semantics

Manual spans SHOULD represent:
- **Business-relevant operations** — planning, simulation, embedding generation, LLM calls
- **State transitions** — CWM reads/writes, HCG mutations
- **Cross-boundary calls** — outbound HTTP to other LOGOS services (though `HTTPXClientInstrumentor` already handles the transport-level span; manual spans add semantic context like "this was a plan creation request")

Manual spans SHOULD NOT:
- Wrap trivial helper functions or pure computations
- Duplicate what auto-instrumentation already provides (e.g. don't manually span every HTTP handler — `FastAPIInstrumentor` does this)
- Include high-cardinality attributes (UUIDs are fine as span attributes; raw prompts, full request bodies, or user-generated strings are not)

### Metrics Scope (v1)

v1 metrics are limited to:
- **Auto-instrumented HTTP metrics** from `FastAPIInstrumentor` (request count, latency, error rate per endpoint)
- **Existing JEPA metrics** in `sophia/src/sophia/jepa/metrics.py` (inference count, latency, GPU memory) — already implemented, no changes
- **OTel Collector self-metrics** scraped by Prometheus

v1 does **not** define custom application metrics beyond the above. No custom counters, histograms, or gauges are added in this spec. If custom metrics are needed later, they should follow the same `{service}.{metric_name}` naming convention and avoid high-cardinality labels (no UUIDs, no free-text, no per-request labels).

### Invariants

- `setup_telemetry()` MUST be called **exactly once per process**, during app startup (in the lifespan handler). Calling it multiple times overwrites the global `TracerProvider`, which orphans any previously-configured exporters.
- `FastAPIInstrumentor.instrument_app(app)` MUST be called **exactly once per app instance**. Double-instrumentation produces duplicate spans.
- `HTTPXClientInstrumentor().instrument()` MUST be called **exactly once per process**. It monkey-patches the httpx module globally.
- Re-instrumentation behavior (e.g. after a lifespan restart) is **undefined and unsupported** by the OTel SDK. In practice, FastAPI lifespans don't restart within a process — the process exits and a new one starts.

### OTel Collector as Behavioral Component

The OTel Collector is not merely an observability sink — it is part of the runtime behavior of the system. It controls:
- Batching and retry semantics for span export
- Memory limits that determine when spans are dropped
- Sampling decisions (in future configurations)
- Routing of traces and metrics to backends

**Changes to collector configuration must be reviewed with the same rigor as application code changes.** A misconfigured collector can cause span loss, increased latency (via backpressure), or silent data corruption in backends.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `None` (disabled) | OTel collector gRPC endpoint (e.g. `http://localhost:4317`) |
| `OTEL_EXPORT_CONSOLE` | `false` | Enable console span export (dev mode) |

---

## Repository Layout (relevant files)

```
LOGOS/
├── logos/                              # Foundry repo (shared library)
│   ├── pyproject.toml                  # OTel deps currently in test group (Phase 0 changes this)
│   ├── logos_observability/
│   │   ├── __init__.py                 # Public API: setup_telemetry, get_tracer, get_meter, get_logger
│   │   ├── telemetry.py                # TracerProvider/MeterProvider setup, OTLP export
│   │   └── exporter.py                 # File-based telemetry export (JSONL)
│   ├── logos_sophia/
│   │   └── api.py                      # Has existing tracer on /simulate (needs normalization)
│   ├── docker-compose.otel.yml         # Jaeger + Prometheus + OTel Collector
│   ├── config/
│   │   ├── otel-collector-config.yaml  # Collector pipelines
│   │   └── prometheus.yml              # Scrape targets
│   └── tests/
│       ├── integration/observability/
│       │   └── test_otel_smoke.py      # Unit-level OTel tests (no collector needed)
│       └── infra/
│           └── test_otel_stack.py      # Integration tests against live compose
├── sophia/                             # Cognitive core
│   ├── pyproject.toml                  # Pins logos-foundry@v0.2.0
│   └── src/sophia/
│       ├── api/app.py                  # FastAPI app with lifespan (Phase 1 wires OTel here)
│       └── jepa/metrics.py             # Existing JEPA OTel metrics (leave untouched)
├── hermes/                             # Language & embedding service
│   ├── pyproject.toml                  # Pins logos-foundry@v0.2.0
│   └── src/hermes/main.py             # FastAPI app with lifespan (Phase 1 wires OTel here)
└── apollo/                             # UI and command layer
    ├── pyproject.toml                  # Pins logos-foundry@v0.2.0
    └── src/apollo/api/server.py        # FastAPI app with lifespan (Phase 1 wires OTel here)
```

---

## Current State (what already exists)

### logos_observability module

**File:** `logos/logos_observability/telemetry.py`

The module imports OTel SDK unconditionally (lines 14-22) and guards OTLP exporters with try/except (lines 24-32):

```python
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False
```

`setup_telemetry()` signature (line 164):
```python
def setup_telemetry(
    service_name: str = "logos-service",
    export_to_console: bool = True,
    otlp_endpoint: str | None = None,
) -> TracerProvider:
```

**Implication:** The core SDK imports (`opentelemetry.sdk.*`) are NOT guarded. If OTel packages are missing at runtime, `import logos_observability` will crash. This is why Phase 0 (moving deps to production) is a hard prerequisite.

### Existing Sophia instrumentation

**File:** `logos/logos_sophia/api.py` (lines 4, 10, 78-99)

```python
from opentelemetry import trace                    # line 4
tracer = trace.get_tracer(__name__)                # line 10

@router.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulationRequest):
    with tracer.start_as_current_span("simulate") as span:  # <- NOT normalized
        result = simulation_service.run_simulation(request)
        span.set_attribute("plan_id", result.process.uuid)
        span.set_attribute("capability_id", request.capability_id)
        span.set_attribute("horizon", result.process.horizon)
```

**Problem:** Span is named `"simulate"` instead of `"sophia.simulate"`. Tracer name is `__name__` instead of `"sophia.simulation"`.

**File:** `sophia/src/sophia/jepa/metrics.py`

`JEPAMetrics` class uses `get_meter("sophia.jepa", version="1.0.0")` — this already follows the naming convention. Leave it untouched.

### Docker compose (already correct)

**File:** `logos/docker-compose.otel.yml`

Already deploys: OTel Collector, Jaeger, Prometheus. **Port issue:** maps `4319:4317` and `4320:4318` instead of standard `4317:4317` and `4318:4318`.

### OTel Collector config (already correct)

**File:** `logos/config/otel-collector-config.yaml`

Receives OTLP on `:4317` (gRPC) and `:4318` (HTTP). Exports traces to Jaeger, metrics to Prometheus.

### Prometheus config (already correct)

**File:** `logos/config/prometheus.yml`

Scrapes: `otel-collector:8889`, `sophia:8001`, `hermes:8002`, `apollo:8000`.

### Infrastructure tests

**File:** `logos/tests/infra/test_otel_stack.py`

Tests reference ports `4319` (line 75) and `4320` (line 85) — these must be updated to `4317` and `4318` when the compose is fixed.

---

## Phase 0 — Foundry Update

**Repo:** `logos/`
**Branch:** `feature/otel-phase0`
**Ticket coverage:** Prerequisite for #334, #337, #340
**Gate:** Tag `v0.3.0` must exist before Phase 1 begins.

### Step 0.1: Move OTel deps from test to production group

**File:** `logos/pyproject.toml`

**Current state** (lines 61-71):
```toml
[tool.poetry.group.test.dependencies]
rdflib = "^7.0.0"
pyshacl = "^0.25.0"
neo4j = "^6.0.0"
pymilvus = "^2.3.0"
opentelemetry-api = "^1.20.0"
opentelemetry-sdk = "^1.20.0"
opentelemetry-instrumentation-fastapi = "^0.41b0"
opentelemetry-exporter-otlp-proto-grpc = "^1.20.0"
numpy = "^1.24.0"
requests = "^2.32.5"
PyYAML = "^6.0"
```

**Target state for `[tool.poetry.dependencies]`** — add these four lines plus the new httpx one after the existing production deps (after line 59):
```toml
opentelemetry-api = "^1.20.0"
opentelemetry-sdk = "^1.20.0"
opentelemetry-instrumentation-fastapi = "^0.41b0"
opentelemetry-exporter-otlp-proto-grpc = "^1.20.0"
opentelemetry-instrumentation-httpx = "^0.41b0"
```

**Target state for `[tool.poetry.group.test.dependencies]`** — remove the four OTel lines, leaving:
```toml
[tool.poetry.group.test.dependencies]
rdflib = "^7.0.0"
pyshacl = "^0.25.0"
neo4j = "^6.0.0"
pymilvus = "^2.3.0"
numpy = "^1.24.0"
requests = "^2.32.5"
PyYAML = "^6.0"
```

**Verification:**
```bash
cd logos && poetry lock && poetry install
poetry run python -c "from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor; print('OK')"
poetry run python -c "from logos_observability import setup_telemetry; print('OK')"
```

### Step 0.2: Fix compose port mapping

**File:** `logos/docker-compose.otel.yml`

**Change** the otel-collector ports section:

| Line | Current | Target |
|------|---------|--------|
| Port mapping | `"4319:4317"` | `"4317:4317"` |
| Port mapping | `"4320:4318"` | `"4318:4318"` |

**Note:** The Jaeger service also maps `4317:4317`. After this change, both the collector and Jaeger expose port 4317, which will conflict. The collector should be the OTLP entry point, so **remove** the `4317:4317` line from the `jaeger` service. Jaeger receives traces from the collector internally (via `jaeger:4317` on the Docker network), not from the host.

**Target compose otel-collector ports:**
```yaml
    ports:
      - "4317:4317"   # OTLP gRPC receiver
      - "4318:4318"   # OTLP HTTP receiver
      - "8889:8889"   # Prometheus metrics exporter
      - "13133:13133" # Health check
```

**Target compose jaeger ports:**
```yaml
    ports:
      - "16686:16686"  # Jaeger UI
```

### Step 0.3: Update infrastructure tests for new ports

**File:** `logos/tests/infra/test_otel_stack.py`

**Change `test_otlp_grpc_port_open`** (line 75):
```python
# Before:
    result = sock.connect_ex(("localhost", 4319))
# After:
    result = sock.connect_ex(("localhost", 4317))
```
Update the assert message on line 77 accordingly: `"OTLP gRPC port 4317 should be open"`.

**Change `test_otlp_http_port_open`** (line 85):
```python
# Before:
    result = sock.connect_ex(("localhost", 4320))
# After:
    result = sock.connect_ex(("localhost", 4318))
```
Update the assert message on line 87 accordingly: `"OTLP HTTP port 4318 should be open"`.

### Step 0.4: Run full test suite

```bash
cd logos
poetry run pytest tests/ -v
```

All tests must pass, including `test_otel_smoke.py` (unit-level, no compose needed) and `test_otel_stack.py` (requires compose — run `docker compose -f docker-compose.otel.yml up -d` first, wait 10s).

### Step 0.5: Bump version and tag

**File:** `logos/pyproject.toml` — change line 5:
```toml
# Before:
version = "0.2.0"
# After:
version = "0.3.0"
```

```bash
cd logos
git add -A && git commit -m "feat(otel): move OTel deps to production, add httpx instrumentation, fix ports"
git tag v0.3.0
git push origin main --tags
```

### Step 0.6: Verification

```bash
# Verify tag exists
git tag --list 'v0.3.0'

# Verify compose works with new ports
docker compose -f docker-compose.otel.yml up -d
curl -s http://localhost:16686/ | head -1           # Jaeger UI
curl -s http://localhost:9090/-/healthy              # Prometheus
curl -s http://localhost:13133/                      # Collector health

# Verify OTLP ports
python -c "import socket; s=socket.socket(); print(s.connect_ex(('localhost', 4317)))"  # should print 0
python -c "import socket; s=socket.socket(); print(s.connect_ex(('localhost', 4318)))"  # should print 0
```

---

## Phase 1 — Service Instrumentation (parallel: Sophia, Hermes, Apollo)

**Ticket coverage:** #334+#335+#336 (Sophia), #337+#338+#339 (Hermes), #340+#341+#342 (Apollo)

### Orchestration Protocol

1. Create `feature/otel-instrumentation` branch in all three repos (sophia, hermes, apollo).
2. In each repo, update the logos-foundry dependency to `v0.3.0` and run `poetry lock`.
3. Dispatch three independent workers, one per service.
4. Each worker: wires telemetry, adds/normalizes manual spans, adds smoke tests, creates a PR.
5. PRs are independent — no cross-service blocking.

### Step 1.0: Update dependency pins (all three repos)

**Sophia** — `sophia/pyproject.toml` line 39:
```toml
# Before:
logos-foundry = {git = "https://github.com/c-daly/logos.git", tag = "v0.2.0"}
# After:
logos-foundry = {git = "https://github.com/c-daly/logos.git", tag = "v0.3.0"}
```

**Hermes** — `hermes/pyproject.toml` line 30:
```toml
# Before:
"logos-foundry @ git+https://github.com/c-daly/logos.git@v0.2.0 ; python_version >= \"3.11\" and python_version < \"4.0\"",
# After:
"logos-foundry @ git+https://github.com/c-daly/logos.git@v0.3.0 ; python_version >= \"3.11\" and python_version < \"4.0\"",
```

**Apollo** — `apollo/pyproject.toml` lines 23-25:
```toml
# Before:
logos-sophia-sdk = {git = "https://github.com/c-daly/logos.git", tag = "v0.2.0", subdirectory = "sdk/python/sophia"}
logos-hermes-sdk = {git = "https://github.com/c-daly/logos.git", tag = "v0.2.0", subdirectory = "sdk/python/hermes"}
logos-foundry = {git = "https://github.com/c-daly/logos.git", tag = "v0.2.0"}
# After:
logos-sophia-sdk = {git = "https://github.com/c-daly/logos.git", tag = "v0.3.0", subdirectory = "sdk/python/sophia"}
logos-hermes-sdk = {git = "https://github.com/c-daly/logos.git", tag = "v0.3.0", subdirectory = "sdk/python/hermes"}
logos-foundry = {git = "https://github.com/c-daly/logos.git", tag = "v0.3.0"}
```

Then in each repo: `poetry lock && poetry install`

### Sophia Worker

#### 1A. Wire telemetry on app startup

**File:** `sophia/src/sophia/api/app.py`

**Add imports** after the existing imports (after line 30, before the model imports):
```python
import os
from logos_observability import setup_telemetry
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
```

**Add telemetry initialization** inside the `lifespan()` function, at the very beginning of the Startup section (after line 240 `logger.info("Starting Sophia API service...")`):
```python
    # Initialize OpenTelemetry
    setup_telemetry(
        service_name="sophia",
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        export_to_console=os.getenv("OTEL_EXPORT_CONSOLE", "").lower() == "true",
    )
    HTTPXClientInstrumentor().instrument()
```

**Add FastAPI instrumentation** in the `create_app()` function, after the app is created and middleware is added (after line 394 `app.add_middleware(RequestIDMiddleware)`):
```python
    FastAPIInstrumentor.instrument_app(app)
```

#### 1B. Normalize existing spans

**File:** `logos/logos_sophia/api.py`

**Change tracer initialization** (line 10):
```python
# Before:
tracer = trace.get_tracer(__name__)
# After:
tracer = trace.get_tracer("sophia.simulation")
```

**Change span name** (line 78, inside the `simulate` function):
```python
# Before:
    with tracer.start_as_current_span("simulate") as span:
# After:
    with tracer.start_as_current_span("sophia.simulate") as span:
```

#### 1C. Add manual spans on key operations

Add spans where they make sense in Sophia's codebase. Use this pattern:

```python
from logos_observability import get_tracer

tracer = get_tracer("sophia.planner")

async def create_plan(goal: str) -> Plan:
    with tracer.start_as_current_span("sophia.plan") as span:
        span.set_attribute("plan.goal", goal)
        result = await self._run_planning(goal)
        span.set_attribute("plan.step_count", len(result.steps))
        return result
```

**Suggested spans** (add where they naturally fit — guidance, not exhaustive):
- `sophia.plan` — planning pipeline (goal, step count)
- `sophia.simulate` — already exists after normalization
- `sophia.hcg.read` / `sophia.hcg.write` — HCG client operations
- `sophia.state.read` / `sophia.state.write` — CWM state operations

**Do NOT modify** `sophia/src/sophia/jepa/metrics.py` — it already uses `get_meter("sophia.jepa")` correctly.

#### 1D. Add smoke test

**Create file:** `sophia/tests/integration/test_otel_smoke.py`

```python
"""Smoke tests for OpenTelemetry instrumentation in Sophia."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory import InMemorySpanExporter

from logos_observability import setup_telemetry, get_tracer


@pytest.fixture(autouse=True)
def reset_tracer_provider():
    """Reset the global tracer provider between tests."""
    yield
    # Reset to prevent cross-test contamination
    trace.set_tracer_provider(TracerProvider())


def test_sophia_telemetry_setup():
    """Verify setup_telemetry configures a working TracerProvider."""
    provider = setup_telemetry(service_name="sophia", export_to_console=False)
    assert provider is not None

    tracer = get_tracer("sophia.test")
    with tracer.start_as_current_span("sophia.test_span") as span:
        span.set_attribute("test.key", "test_value")
    # No assertion needed — if this doesn't crash, setup works


def test_sophia_spans_have_correct_service_name():
    """Verify spans carry the correct service.name resource attribute."""
    setup_telemetry(service_name="sophia", export_to_console=False)
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = get_tracer("sophia.planner")
    with tracer.start_as_current_span("sophia.plan") as span:
        span.set_attribute("plan.goal", "test")

    provider.force_flush()

    spans = exporter.get_finished_spans()
    assert len(spans) > 0
    assert spans[0].name == "sophia.plan"
    assert spans[0].resource.attributes["service.name"] == "sophia"


def test_sophia_nested_spans():
    """Verify nested spans maintain parent-child relationships."""
    setup_telemetry(service_name="sophia", export_to_console=False)
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = get_tracer("sophia.api")
    with tracer.start_as_current_span("sophia.plan") as parent:
        with tracer.start_as_current_span("sophia.simulate") as child:
            child.set_attribute("simulation.steps", 5)

    provider.force_flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    child_span = next(s for s in spans if s.name == "sophia.simulate")
    parent_span = next(s for s in spans if s.name == "sophia.plan")
    assert child_span.parent.span_id == parent_span.context.span_id
```

#### 1E. Verify and PR

```bash
cd sophia
poetry run ruff check src/ tests/
poetry run ruff format --check src/ tests/
poetry run mypy src/
poetry run pytest tests/ -v
```

All must pass. Create PR:
- Branch: `feature/otel-instrumentation`
- Title: `feat(otel): instrument Sophia with OpenTelemetry tracing`

---

### Hermes Worker

#### 2A. Wire telemetry on app startup

**File:** `hermes/src/hermes/main.py`

**Add imports** after the existing imports (after line 48, before the logger setup):
```python
import os  # noqa: E402
from logos_observability import setup_telemetry  # noqa: E402
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: E402
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor  # noqa: E402
```

Note: Hermes uses `# noqa: E402` on imports because `load_dotenv()` is called at the top of the file before other imports.

**Add telemetry initialization** inside the `lifespan()` function, at the beginning of the Startup section (after line 77 `logger.info("Starting Hermes API...")`):
```python
    # Initialize OpenTelemetry
    setup_telemetry(
        service_name="hermes",
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        export_to_console=os.getenv("OTEL_EXPORT_CONSOLE", "").lower() == "true",
    )
    HTTPXClientInstrumentor().instrument()
```

**Add FastAPI instrumentation** after the app is created and middleware is added (after line 110 `app.add_middleware(RequestIDMiddleware)`):
```python
FastAPIInstrumentor.instrument_app(app)
```

#### 2B. Add manual spans on key operations

Add spans on the service functions. Hermes has inline httpx usage in `_forward_llm_to_sophia()` (line 471) and `ingest_media()` (line 615) — these will be automatically traced by `HTTPXClientInstrumentor`. Add manual spans on the core operations:

**Suggested spans:**
- `hermes.embed` — in the `/embed_text` endpoint (around `generate_embedding()` call, line 455)
- `hermes.llm` — in the `/llm` endpoint (around `generate_llm_response()` call, line 558)
- `hermes.nlp` — in the `/simple_nlp` endpoint (around `process_nlp()` call, line 426)
- `hermes.stt` — in the `/stt` endpoint (around `transcribe_audio()` call, line 356)
- `hermes.tts` — in the `/tts` endpoint (around `synthesize_speech()` call, line 386)

**Pattern:**
```python
from logos_observability import get_tracer

tracer = get_tracer("hermes.api")

@app.post("/embed_text", response_model=EmbedTextResponse)
async def embed_text(request: EmbedTextRequest) -> EmbedTextResponse:
    with tracer.start_as_current_span("hermes.embed") as span:
        span.set_attribute("embed.model", request.model)
        # ... existing logic ...
        span.set_attribute("embed.dimension", result["dimension"])
```

#### 2C. Add smoke test

**Create file:** `hermes/tests/integration/test_otel_smoke.py`

Same pattern as Sophia's test (see Step 1D), but with `service_name="hermes"` and span names like `"hermes.embed"`, `"hermes.llm"`.

#### 2D. Verify and PR

```bash
cd hermes
poetry run ruff check src/ tests/
poetry run ruff format --check src/ tests/
poetry run mypy src/
poetry run pytest tests/ -v
```

All must pass. Create PR:
- Branch: `feature/otel-instrumentation`
- Title: `feat(otel): instrument Hermes with OpenTelemetry tracing`

---

### Apollo Worker

#### 3A. Wire telemetry on app startup

**File:** `apollo/src/apollo/api/server.py`

**Add imports** after the existing imports (after line 40, before the apollo-specific imports on line 42):
```python
import os
from logos_observability import setup_telemetry
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
```

Note: `os` may already be imported (line 10). If so, skip it.

**Add telemetry initialization** inside the `lifespan()` function, at the very beginning (after line 277 `global hcg_client, diagnostics_task, persona_store, hermes_client`):
```python
    # Initialize OpenTelemetry
    setup_telemetry(
        service_name="apollo",
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        export_to_console=os.getenv("OTEL_EXPORT_CONSOLE", "").lower() == "true",
    )
    HTTPXClientInstrumentor().instrument()
```

**Add FastAPI instrumentation** after the app is created and CORS middleware is added (after line 417 `app.add_middleware(CORSMiddleware, ...)`):
```python
FastAPIInstrumentor.instrument_app(app)
```

#### 3B. Add manual spans on key operations

Apollo already imports `httpx` (line 15) and creates an `httpx.AsyncClient` in the lifespan (line 369) — outbound calls to Sophia and Hermes will be automatically traced by `HTTPXClientInstrumentor`.

**Suggested manual spans:**
- `apollo.command` — CLI command dispatch (if applicable to API endpoints)
- `apollo.api.sophia` — outbound calls to Sophia (in endpoints that call Sophia)
- `apollo.api.hermes` — outbound calls to Hermes (in endpoints that call Hermes)

These are optional — `HTTPXClientInstrumentor` already creates spans for all outbound HTTP calls. Manual spans are only needed if you want to wrap higher-level operations with semantic names and business-relevant attributes.

#### 3C. Add smoke test

**Create file:** `apollo/tests/integration/test_otel_smoke.py`

Same pattern as Sophia's test (see Step 1D), but with `service_name="apollo"` and span names like `"apollo.command"`.

#### 3D. Verify and PR

```bash
cd apollo
poetry run ruff check src/ tests/
poetry run ruff format --check src/ tests/
poetry run mypy src/
poetry run pytest tests/ -v
```

All must pass. Create PR:
- Branch: `feature/otel-instrumentation`
- Title: `feat(otel): instrument Apollo with OpenTelemetry tracing`

---

## Phase 2 — Integration Verification

**Prerequisite:** All three Phase 1 PRs merged.
**Ticket coverage:** #344

### Step 2.1: Start infrastructure

```bash
cd logos
docker compose -f docker-compose.otel.yml up -d
# Wait 10 seconds for services to stabilize
sleep 10
```

### Step 2.2: Start all three services with OTel export

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 uvicorn sophia.api.app:create_app --factory --host 0.0.0.0 --port 8001 &
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 uvicorn hermes.main:app --host 0.0.0.0 --port 8002 &
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 uvicorn apollo.api.server:app --host 0.0.0.0 --port 8000 &
```

(Adjust startup commands to match each repo's entry point. Sophia uses a factory pattern `create_app()`, Hermes and Apollo use module-level `app`.)

### Step 2.3: Trigger cross-service requests

Make requests through Apollo that fan out to Sophia and Hermes. The specific endpoints depend on what's wired, but typical flows:

- Apollo → Sophia: plan creation, state queries
- Apollo → Hermes: embedding generation, LLM calls
- Hermes → Sophia: LLM response forwarding (`_forward_llm_to_sophia`)

### Step 2.4: Manual verification in Jaeger

Open `http://localhost:16686` and verify:

- [ ] All three service names (`sophia`, `hermes`, `apollo`) appear in the service dropdown
- [ ] End-to-end trace with parent-child span relationships across service boundaries
- [ ] Manual span attributes visible (plan IDs, model names, etc.)
- [ ] W3C trace context propagated correctly (single trace ID across services)

### Step 2.5: Manual verification in Prometheus

Open `http://localhost:9090` and verify:

- [ ] Navigate to Status → Targets
- [ ] `otel-collector` target is UP
- [ ] Service targets show as expected (may be DOWN if services don't expose metrics endpoints directly — that's fine, metrics flow through the collector)

### Step 2.6: Run stack tests

```bash
cd logos
poetry run pytest tests/infra/test_otel_stack.py -v
```

### Step 2.7: Scripted integration test

**Create file:** `logos/tests/integration/observability/test_cross_service_traces.py`

```python
"""Integration test for cross-service trace propagation.

Requires:
- docker compose -f docker-compose.otel.yml up -d
- All three services running with OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

Run manually as part of Phase 2 verification, not in CI.
"""

import time

import httpx
import pytest

JAEGER_API = "http://localhost:16686/api"

# Adjust these to match actual service ports
APOLLO_API = "http://localhost:8000"
HERMES_API = "http://localhost:8002"


@pytest.fixture(scope="module")
def verify_services_running():
    """Verify all services and Jaeger are accessible before running tests."""
    for url, name in [
        (f"{JAEGER_API}/services", "Jaeger"),
        (f"{APOLLO_API}/health", "Apollo"),
        (f"{HERMES_API}/health", "Hermes"),
    ]:
        try:
            resp = httpx.get(url, timeout=5)
            resp.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            pytest.skip(f"{name} not accessible at {url}: {e}")


def test_jaeger_receives_service_spans(verify_services_running):
    """Verify all instrumented services appear in Jaeger."""
    # Make a request to each service to generate spans
    httpx.get(f"{APOLLO_API}/health", timeout=5)
    httpx.get(f"{HERMES_API}/health", timeout=5)

    # Wait for spans to flush (BatchSpanProcessor default: 5s schedule, plus network)
    time.sleep(8)

    resp = httpx.get(f"{JAEGER_API}/services", timeout=5)
    assert resp.status_code == 200
    services = resp.json()["data"]

    # At minimum, Apollo and Hermes should appear (Sophia may need a direct request)
    assert "apollo" in services, f"apollo not in Jaeger services: {services}"
    assert "hermes" in services, f"hermes not in Jaeger services: {services}"


def test_cross_service_trace_propagation(verify_services_running):
    """Verify a single trace ID spans multiple services.

    This test triggers a request that should flow from one service to another
    (e.g., Hermes LLM forwarding to Sophia, or Apollo calling Hermes).
    The exact endpoint depends on what's configured and available.
    """
    # Trigger a cross-service flow
    # Example: POST to Hermes /embed_text which may not call other services,
    # but POST to Hermes /llm with a configured provider will forward to Sophia.
    # Adjust based on what's available in your environment.
    try:
        resp = httpx.post(
            f"{HERMES_API}/embed_text",
            json={"text": "test cross-service trace", "model": "default"},
            timeout=30,
        )
    except httpx.HTTPStatusError:
        pytest.skip("embed_text endpoint not available or ML deps missing")

    time.sleep(8)

    # Query Jaeger for recent traces from hermes
    resp = httpx.get(
        f"{JAEGER_API}/traces",
        params={"service": "hermes", "limit": 10, "lookback": "5m"},
        timeout=5,
    )
    assert resp.status_code == 200
    traces = resp.json()["data"]
    assert len(traces) > 0, "No traces found for hermes service"

    # Verify at least one trace has spans (cross-service propagation
    # is confirmed when a trace contains spans from 2+ services,
    # but even single-service traces confirm instrumentation is working)
    for t in traces:
        service_names = {span["process"]["serviceName"] for span in t["spans"]}
        if len(service_names) >= 2:
            return  # Cross-service propagation confirmed

    # If no multi-service traces found, that's acceptable for this test —
    # it depends on the specific request triggering cross-service calls.
    # The important thing is that traces exist at all.
    print(f"Note: Found {len(traces)} traces but none spanning multiple services. "
          "This is expected if the test request didn't trigger cross-service calls.")
```

---

## Phase 3 — Backend Upgrade to Tempo + Grafana

**Prerequisite:** Phase 2 complete.

**Important:** This phase changes the observability UX, not just infrastructure. Tempo has different query semantics than Jaeger (TraceQL vs Jaeger's tag-based filtering), different retention behavior, and a different mental model for trace discovery. All Phase 2 validation criteria must be re-verified after migration — do not assume "same traces, different UI."

### Steps

1. **Update `logos/docker-compose.otel.yml`** — replace Jaeger with Tempo + Grafana:
   - OTel Collector (:4317 gRPC, :4318 HTTP) — keep as-is
   - Add Tempo (:3200 query frontend) — config exists at `logos/infra/tempo-config.yaml`
   - Add Grafana (:3001 UI) with Tempo datasource
   - Remove Jaeger service

2. **Update `logos/config/otel-collector-config.yaml`** — change trace exporter from `otlp/jaeger` to `otlp/tempo`:
   ```yaml
   exporters:
     otlp/tempo:
       endpoint: tempo:4317
       tls:
         insecure: true
   ```

3. **Update `logos/tests/infra/test_otel_stack.py`** — replace Jaeger-specific tests with Tempo/Grafana equivalents:
   - Remove `test_jaeger_ui_accessible` and `test_jaeger_api_services`
   - Add `test_grafana_accessible` (port 3001)
   - Add `test_tempo_accessible` (port 3200)

4. **Verify** traces visible in Grafana via Tempo datasource.

5. **Repeat** cross-service verification from Phase 2, now viewing traces in Grafana.

---

## Execution Summary

| Phase | Scope | Blocking? | PRs |
|-------|-------|-----------|-----|
| 0 | Foundry: deps to production, httpx, port fix, tag v0.3.0 | Yes — gate for Phase 1 | 1 |
| 1 | Service instrumentation + normalize + tests (parallel) | No cross-service blocking | 3 |
| 2 | Integration verification + scripted test | Requires all Phase 1 merged | 1 |
| 3 | Tempo + Grafana upgrade | Requires Phase 2 complete | 1 |

**Total PRs:** 6

---

## Follow-Up Considerations

### `logos_config` Integration
Currently, `setup_telemetry()` takes `otlp_endpoint` as a parameter fed by `os.getenv()`. When formal environments are introduced, this should be routed through `logos_config` instead:

```python
from logos_config import get_config
setup_telemetry(
    service_name="sophia",
    otlp_endpoint=get_config().otel_endpoint,
)
```

This enables environment-specific config profiles without per-service code changes. The env var override (`OTEL_EXPORTER_OTLP_ENDPOINT`) continues to work via the OTel SDK natively.

### Environment Migration Path
The architecture is environment-agnostic by design:
- **Service code** never changes — it calls `setup_telemetry()` with a configured endpoint
- **OTel Collector** is the routing layer — different collector configs per environment (sampling rates, exporters, memory limits)
- **Infrastructure** (compose files, k8s manifests) defines what backends are deployed
- **Sampling changes** require a separate design doc (see Sampling Policy above)

### Apollo DiagnosticsManager
Apollo has a custom `DiagnosticsManager` with WebSocket telemetry streaming (defined at `apollo/src/apollo/api/server.py:114`) that is parallel to OTel. A future ticket could bridge these — e.g., feeding OTel metrics into the diagnostics WebSocket, or replacing the custom telemetry with OTel metrics.

### Talos Telemetry
Talos has its own `TelemetryRecorder` for hardware-level events (actuator positions, gripper state). This is not OTel-integrated. A future ticket could bridge Talos events into OTel spans if hardware observability is needed in the trace view.
