# Logging & Observability

The harness emits structured JSON logs compatible with OpenTelemetry's log data model. This gives you two collection paths with zero config changes.

## Quick Start

```python
from harness.logging import get_logger

log = get_logger("my_experiment", attempt=1)
log.info("Starting training", approach="ridge_regression", lr=1e-4)
log.metric("loss", 0.42, step=100)
log.error("Gradient explosion", grad_norm=1e8, step=150)
```

This writes JSON lines to `experiments/my_experiment/logs/<timestamp>.jsonl`.

## Log Schema

Every line is a JSON object matching the [OTel Log Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/):

```json
{
  "timestamp": "2026-03-10T14:30:00.000000+00:00",
  "observed_timestamp": "2026-03-10T14:30:00.001000+00:00",
  "severity": "INFO",
  "severity_number": 9,
  "body": "Starting training",
  "attributes": {
    "approach": "ridge_regression",
    "lr": 0.0001,
    "logger.name": "harness.my_experiment",
    "code.function": "train",
    "code.filepath": "workspace/train.py",
    "code.lineno": 42
  },
  "resource": {
    "service.name": "logos-harness",
    "experiment.name": "my_experiment",
    "experiment.attempt": 1
  },
  "trace_id": "a1b2c3d4e5f6...",
  "span_id": null
}
```

### Fields

| Field | Source | Description |
|---|---|---|
| `timestamp` | Automatic | When the log was created (UTC, ISO 8601) |
| `severity` | Log level | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `severity_number` | Log level | OTel numeric severity (5, 9, 13, 17, 21) |
| `body` | Message | The log message text |
| `attributes` | kwargs + auto | Structured data: your kwargs + code location |
| `resource` | Logger init | Identifies the experiment and service |
| `trace_id` | Per logger | Groups all logs from one logger instance |
| `span_id` | Per span | Groups logs within a span (see Spans below) |

## Metrics

Use `log.metric()` for experiment metrics. These appear as normal log lines with special attributes:

```python
log.metric("cosine_similarity", 0.72, step=500, epoch=3)
```

Produces:
```json
{
  "body": "[METRIC] cosine_similarity=0.72",
  "attributes": {
    "metric.name": "cosine_similarity",
    "metric.value": 0.72,
    "step": 500,
    "epoch": 3
  }
}
```

The `[METRIC]` prefix in the body means these lines are also parseable by the training monitor (`harness-monitor`), which watches stdout for this format.

## Spans

Spans trace experiment phases. They create nested context with unique span IDs:

```python
with log.span("data_preparation") as s:
    log.info("Downloading dataset", source="kinetics")
    s.set_attribute("dataset.size", 50000)
    
    with log.span("preprocessing"):
        log.info("Normalizing")
```

This logs `span.start` and `span.end` entries with duration, and all logs inside a span carry the span's `span_id`. Nested spans get distinct IDs, and the parent's span_id resumes when the inner span ends.

When connected to an otel backend, spans show up as a trace timeline — you can see how long each phase of the experiment took.

## Collection Path 1: File-based (default)

Logs write to `experiments/<name>/logs/<timestamp>.jsonl`. No dependencies required.

Collect with [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)'s filelog receiver:

```yaml
# otel-collector-config.yaml
receivers:
  filelog:
    include:
      - /path/to/experiments/*/logs/*.jsonl
    operators:
      - type: json_parser
        timestamp:
          parse_from: attributes.timestamp
          layout: '%Y-%m-%dT%H:%M:%S'
        severity:
          parse_from: attributes.severity

exporters:
  otlp:
    endpoint: "localhost:4317"
  # or: loki, elasticsearch, etc.

service:
  pipelines:
    logs:
      receivers: [filelog]
      exporters: [otlp]
```

## Collection Path 2: SDK Export (optional)

For real-time export without file intermediary:

```bash
pip install "logos-harness[otel] @ git+https://github.com/yourusername/logos-harness.git"
```

```python
from harness.logging import setup_otel_export

# Connect to otel collector (additive — file logging continues)
setup_otel_export("http://localhost:4317")
```

This adds an OTLP log exporter alongside the file handler. All logs go to both destinations.

## Querying

Once logs are in your backend (Grafana/Loki, Elasticsearch, etc.), you can query by:

- `resource.experiment.name` — filter to one experiment
- `resource.experiment.attempt` — filter to one attempt
- `attributes.metric.name` — find all values of a specific metric
- `severity` — find errors and warnings
- `trace_id` — see all logs from one session
- `span_id` — see logs from one phase
- `attributes.experiment.phase` — filter by phase name

## Example: Watching an Experiment

```python
from harness.logging import get_logger

log = get_logger("vjepa_clip", attempt=3)

with log.span("research"):
    log.info("Searching for alignment approaches")
    log.info("Found Procrustes method", source="paper", url="...")

with log.span("implementation"):
    log.info("Computing SVD on training pairs", n_samples=5000)
    log.metric("svd_time_seconds", 2.3)

with log.span("evaluation"):
    log.metric("cosine_similarity", 0.68, split="val")
    log.metric("retrieval_at_5", 0.55, split="val")
    log.warning("Below threshold", target=0.70, actual=0.68)
```

In Grafana, this shows up as three phases with timing, metrics plotted over attempts, and the warning highlighted.
