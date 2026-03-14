"""
Structured logging for the experiment harness — otel-compatible.

Emits JSON logs with fields that map directly to OpenTelemetry's log data model:
- timestamp, severity, body, attributes, resource, trace_id, span_id

Two collection paths:
1. File-based (default): JSON lines to logs/*.jsonl → otel collector filelog receiver
2. SDK-based (optional): Direct export via opentelemetry-sdk when configured

Usage:
    from harness.logging import get_logger

    log = get_logger("vjepa_clip_alignment")
    log.info("Starting attempt", attempt=3, approach="procrustes")
    log.metric("cosine_similarity", 0.72, step=500)
    log.error("Training diverged", loss=float('nan'), step=42)

    # With spans (for tracing experiment phases)
    with log.span("data_preparation") as s:
        log.info("Downloading dataset", source="kinetics-400")
        s.set_attribute("dataset.size", 50000)

Otel collector config to ingest these logs:
    receivers:
      filelog:
        include: [/path/to/experiments/*/logs/*.jsonl]
        operators:
          - type: json_parser
            timestamp:
              parse_from: attributes.timestamp
              layout: '%Y-%m-%dT%H:%M:%S.%LZ'
            severity:
              parse_from: attributes.severity
    exporters:
      otlp:
        endpoint: "localhost:4317"
    service:
      pipelines:
        logs:
          receivers: [filelog]
          exporters: [otlp]
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ============================================================================
# Otel-compatible field names (semantic conventions)
# ============================================================================

# Resource attributes (identify the source)
RESOURCE_SERVICE_NAME = "service.name"
RESOURCE_SERVICE_VERSION = "service.version"
RESOURCE_SERVICE_INSTANCE = "service.instance.id"

# Experiment-specific resource attributes
RESOURCE_EXPERIMENT = "experiment.name"
RESOURCE_ATTEMPT = "experiment.attempt"

# Span attributes
ATTR_PHASE = "experiment.phase"
ATTR_APPROACH = "experiment.approach"
ATTR_METRIC_NAME = "metric.name"
ATTR_METRIC_VALUE = "metric.value"


# ============================================================================
# JSON Log Formatter
# ============================================================================

class OtelJsonFormatter(logging.Formatter):
    """
    Formats log records as JSON matching otel's log data model.

    Output schema:
    {
        "timestamp": "2026-03-10T14:30:00.000Z",
        "severity": "INFO",
        "body": "Starting attempt",
        "attributes": {"attempt": 3, "approach": "procrustes"},
        "resource": {"service.name": "logos-harness", "experiment.name": "vjepa_clip_alignment"},
        "trace_id": "abc123...",
        "span_id": "def456..."
    }
    """

    def __init__(self, resource: Optional[dict] = None):
        super().__init__()
        self.resource = resource or {}

    def format(self, record: logging.LogRecord) -> str:
        # Base structure matching otel log data model
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "observed_timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "severity_number": self._severity_number(record.levelno),
            "body": record.getMessage(),
            "attributes": {},
            "resource": self.resource.copy(),
        }

        # Pull structured attributes from the extra dict
        # (set via log.info("msg", extra={"key": "val"}) or our wrapper)
        if hasattr(record, "_structured"):
            log_entry["attributes"].update(record._structured)

        # Standard log record fields as attributes
        log_entry["attributes"]["logger.name"] = record.name
        if record.funcName:
            log_entry["attributes"]["code.function"] = record.funcName
        if record.pathname:
            log_entry["attributes"]["code.filepath"] = record.pathname
        if record.lineno:
            log_entry["attributes"]["code.lineno"] = record.lineno

        # Trace context (if set)
        trace_id = getattr(record, "_trace_id", None)
        span_id = getattr(record, "_span_id", None)
        if trace_id:
            log_entry["trace_id"] = trace_id
        if span_id:
            log_entry["span_id"] = span_id

        # Exception info
        if record.exc_info and record.exc_info[0]:
            log_entry["attributes"]["exception.type"] = record.exc_info[0].__name__
            log_entry["attributes"]["exception.message"] = str(record.exc_info[1])
            import traceback
            log_entry["attributes"]["exception.stacktrace"] = "".join(
                traceback.format_exception(*record.exc_info)
            )

        return json.dumps(log_entry, default=str)

    @staticmethod
    def _severity_number(levelno: int) -> int:
        """Map Python log levels to otel severity numbers."""
        # https://opentelemetry.io/docs/specs/otel/logs/data-model/#severity-fields
        mapping = {
            logging.DEBUG: 5,     # DEBUG
            logging.INFO: 9,      # INFO
            logging.WARNING: 13,  # WARN
            logging.ERROR: 17,    # ERROR
            logging.CRITICAL: 21, # FATAL
        }
        return mapping.get(levelno, 0)


# ============================================================================
# Experiment Logger
# ============================================================================

class ExperimentLogger:
    """
    Structured logger for experiments.

    Wraps Python's logging with:
    - Automatic JSON formatting for otel compatibility
    - Experiment context (name, attempt) as resource attributes
    - Structured key-value attributes on every log call
    - Span context for tracing experiment phases
    - Metric logging with dedicated method
    """

    def __init__(
        self,
        experiment: str,
        log_dir: Optional[Path] = None,
        attempt: Optional[int] = None,
        also_stderr: bool = True,
    ):
        self.experiment = experiment
        self.attempt = attempt
        self._trace_id = uuid.uuid4().hex
        self._current_span_id = None
        self._span_stack = []

        # Resource attributes (identify this experiment run)
        self.resource = {
            RESOURCE_SERVICE_NAME: "logos-harness",
            RESOURCE_EXPERIMENT: experiment,
        }
        if attempt is not None:
            self.resource[RESOURCE_ATTEMPT] = attempt

        # Python logger
        self._logger = logging.getLogger(f"harness.{experiment}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        # Clear existing handlers (avoid duplicates on re-init)
        self._logger.handlers.clear()

        # JSON file handler
        if log_dir is None:
            log_dir = Path("experiments") / experiment / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setFormatter(OtelJsonFormatter(resource=self.resource))
        self._logger.addHandler(file_handler)

        # Stderr handler (human-readable, for watching in terminal)
        if also_stderr:
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            ))
            stderr_handler.setLevel(logging.INFO)
            self._logger.addHandler(stderr_handler)

    def _make_record_extras(self, kwargs: dict) -> dict:
        """Build the extras dict for a log record."""
        return {
            "_structured": kwargs,
            "_trace_id": self._trace_id,
            "_span_id": self._current_span_id,
        }

    def debug(self, msg: str, **kwargs):
        self._logger.debug(msg, extra=self._make_record_extras(kwargs))

    def info(self, msg: str, **kwargs):
        self._logger.info(msg, extra=self._make_record_extras(kwargs))

    def warning(self, msg: str, **kwargs):
        self._logger.warning(msg, extra=self._make_record_extras(kwargs))

    def error(self, msg: str, **kwargs):
        self._logger.error(msg, extra=self._make_record_extras(kwargs))

    def critical(self, msg: str, **kwargs):
        self._logger.critical(msg, extra=self._make_record_extras(kwargs))

    def exception(self, msg: str, **kwargs):
        self._logger.exception(msg, extra=self._make_record_extras(kwargs))

    def metric(self, name: str, value: float, **kwargs):
        """Log a metric value. Shows up in otel as a log with metric attributes."""
        attrs = {
            ATTR_METRIC_NAME: name,
            ATTR_METRIC_VALUE: value,
            **kwargs,
        }
        self._logger.info(
            f"[METRIC] {name}={value}",
            extra=self._make_record_extras(attrs),
        )

    @contextmanager
    def span(self, name: str, **attributes):
        """
        Context manager for tracing experiment phases.

        Creates a span-like context with a unique span_id. Logs span start/end.
        When otel SDK is connected, these become real spans.
        """
        span_id = uuid.uuid4().hex[:16]
        parent_span_id = self._current_span_id

        self._span_stack.append(parent_span_id)
        self._current_span_id = span_id
        start_time = time.time()

        span_attrs = {ATTR_PHASE: name, **attributes}
        self.info(f"span.start: {name}", span_name=name,
                  parent_span_id=parent_span_id, **span_attrs)

        class SpanContext:
            def set_attribute(self_, key, value):
                span_attrs[key] = value

        ctx = SpanContext()
        try:
            yield ctx
        except Exception as e:
            duration = time.time() - start_time
            self.error(f"span.error: {name}", span_name=name,
                       duration_seconds=round(duration, 3),
                       error=str(e), **span_attrs)
            raise
        finally:
            duration = time.time() - start_time
            self.info(f"span.end: {name}", span_name=name,
                      duration_seconds=round(duration, 3), **span_attrs)
            self._current_span_id = self._span_stack.pop()


# ============================================================================
# Factory
# ============================================================================

_loggers: dict[str, ExperimentLogger] = {}


def get_logger(
    experiment: str,
    log_dir: Optional[Path] = None,
    attempt: Optional[int] = None,
    also_stderr: bool = True,
) -> ExperimentLogger:
    """Get or create a logger for an experiment."""
    key = f"{experiment}:{attempt}"
    if key not in _loggers:
        _loggers[key] = ExperimentLogger(
            experiment=experiment,
            log_dir=log_dir,
            attempt=attempt,
            also_stderr=also_stderr,
        )
    return _loggers[key]


def reset_loggers():
    """Clear cached loggers (for testing)."""
    _loggers.clear()


# ============================================================================
# Otel SDK integration (optional — only if packages installed)
# ============================================================================

def setup_otel_export(
    endpoint: str = "http://localhost:4317",
    service_name: str = "logos-harness",
    insecure: bool = True,
):
    """
    Optionally connect to an otel collector for direct log export.
    Requires: pip install opentelemetry-sdk opentelemetry-exporter-otlp

    This is additive — file logging continues alongside otel export.
    """
    try:
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry._logs import set_logger_provider

        resource = Resource.create({
            "service.name": service_name,
        })
        logger_provider = LoggerProvider(resource=resource)
        exporter = OTLPLogExporter(endpoint=endpoint, insecure=insecure)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
        set_logger_provider(logger_provider)

        # Attach otel handler to Python root logger
        from opentelemetry.sdk._logs import LoggingHandler
        handler = LoggingHandler(
            level=logging.DEBUG,
            logger_provider=logger_provider,
        )
        logging.getLogger("harness").addHandler(handler)

        return True

    except ImportError:
        return False
