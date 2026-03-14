"""
Tests for harness/logging.py — structured otel-compatible logging.

Validates:
- JSON output matches otel log data model
- Resource attributes propagate
- Structured attributes on log calls
- Metric logging format
- Span context (trace_id, span_id)
- Severity number mapping
- File output is parseable JSONL
- Otel collector filelog receiver compatibility
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from harness.logging import (
    ExperimentLogger,
    OtelJsonFormatter,
    get_logger,
    reset_loggers,
)


@pytest.fixture(autouse=True)
def clean_loggers():
    """Reset logger cache between tests."""
    reset_loggers()
    yield
    reset_loggers()


@pytest.fixture
def logger(tmp_path):
    """Create a logger writing to a temp dir."""
    return get_logger("test_exp", log_dir=tmp_path, also_stderr=False)


@pytest.fixture
def log_lines(logger, tmp_path):
    """Helper to emit logs and return parsed JSONL lines."""
    def _emit_and_read(fn):
        fn(logger)
        # Flush handlers
        for h in logger._logger.handlers:
            h.flush()
        # Read the log file
        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1, f"Expected 1 log file, got {len(files)}"
        lines = files[0].read_text().strip().splitlines()
        return [json.loads(line) for line in lines]
    return _emit_and_read


# ============================================================================
# Schema: Otel Log Data Model
# ============================================================================

class TestOtelSchema:
    def test_required_fields(self, log_lines):
        """Every log line must have otel's required top-level fields."""
        entries = log_lines(lambda l: l.info("test message"))
        entry = entries[0]

        assert "timestamp" in entry
        assert "severity" in entry
        assert "severity_number" in entry
        assert "body" in entry
        assert "attributes" in entry
        assert "resource" in entry

    def test_timestamp_is_iso8601(self, log_lines):
        entries = log_lines(lambda l: l.info("test"))
        ts = entries[0]["timestamp"]
        # Should parse without error
        datetime.fromisoformat(ts)

    def test_severity_text(self, log_lines):
        def emit(l):
            l.debug("d")
            l.info("i")
            l.warning("w")
            l.error("e")
            l.critical("c")

        entries = log_lines(emit)
        severities = [e["severity"] for e in entries]
        assert severities == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def test_severity_numbers(self, log_lines):
        """Severity numbers must match otel spec."""
        def emit(l):
            l.debug("d")
            l.info("i")
            l.warning("w")
            l.error("e")
            l.critical("c")

        entries = log_lines(emit)
        numbers = [e["severity_number"] for e in entries]
        # Otel: DEBUG=5, INFO=9, WARN=13, ERROR=17, FATAL=21
        assert numbers == [5, 9, 13, 17, 21]

    def test_body_is_message(self, log_lines):
        entries = log_lines(lambda l: l.info("Hello world"))
        assert entries[0]["body"] == "Hello world"

    def test_observed_timestamp_present(self, log_lines):
        entries = log_lines(lambda l: l.info("test"))
        assert "observed_timestamp" in entries[0]


# ============================================================================
# Resource Attributes
# ============================================================================

class TestResourceAttributes:
    def test_service_name(self, log_lines):
        entries = log_lines(lambda l: l.info("test"))
        assert entries[0]["resource"]["service.name"] == "logos-harness"

    def test_experiment_name(self, log_lines):
        entries = log_lines(lambda l: l.info("test"))
        assert entries[0]["resource"]["experiment.name"] == "test_exp"

    def test_attempt_number(self, tmp_path):
        log = get_logger("test_exp", log_dir=tmp_path, attempt=3, also_stderr=False)
        log.info("test")
        for h in log._logger.handlers:
            h.flush()

        lines = list(tmp_path.glob("*.jsonl"))[0].read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["resource"]["experiment.attempt"] == 3


# ============================================================================
# Structured Attributes
# ============================================================================

class TestStructuredAttributes:
    def test_kwargs_become_attributes(self, log_lines):
        entries = log_lines(lambda l: l.info("Starting", attempt=3, approach="mlp"))
        attrs = entries[0]["attributes"]
        assert attrs["attempt"] == 3
        assert attrs["approach"] == "mlp"

    def test_code_attributes(self, log_lines):
        """Logger name and code location should be in attributes."""
        entries = log_lines(lambda l: l.info("test"))
        attrs = entries[0]["attributes"]
        assert "logger.name" in attrs
        assert "code.lineno" in attrs

    def test_empty_kwargs(self, log_lines):
        entries = log_lines(lambda l: l.info("bare message"))
        # Should still have code attributes even with no custom kwargs
        assert "logger.name" in entries[0]["attributes"]


# ============================================================================
# Metrics
# ============================================================================

class TestMetricLogging:
    def test_metric_format(self, log_lines):
        entries = log_lines(lambda l: l.metric("cosine_similarity", 0.72, step=500))
        entry = entries[0]

        assert entry["severity"] == "INFO"
        assert "[METRIC]" in entry["body"]
        assert entry["attributes"]["metric.name"] == "cosine_similarity"
        assert entry["attributes"]["metric.value"] == 0.72
        assert entry["attributes"]["step"] == 500

    def test_multiple_metrics(self, log_lines):
        def emit(l):
            l.metric("loss", 0.5, step=1)
            l.metric("loss", 0.3, step=2)
            l.metric("accuracy", 0.9, step=2)

        entries = log_lines(emit)
        assert len(entries) == 3
        assert entries[0]["attributes"]["metric.value"] == 0.5
        assert entries[1]["attributes"]["metric.value"] == 0.3
        assert entries[2]["attributes"]["metric.name"] == "accuracy"


# ============================================================================
# Trace / Span Context
# ============================================================================

class TestSpanContext:
    def test_trace_id_consistent(self, log_lines):
        def emit(l):
            l.info("first")
            l.info("second")

        entries = log_lines(emit)
        assert entries[0]["trace_id"] == entries[1]["trace_id"]
        assert len(entries[0]["trace_id"]) == 32  # hex uuid

    def test_span_creates_span_id(self, log_lines):
        def emit(l):
            l.info("before span")
            with l.span("data_prep"):
                l.info("inside span")
            l.info("after span")

        entries = log_lines(emit)
        before = entries[0]
        span_start = entries[1]
        inside = entries[2]
        span_end = entries[3]
        after = entries[4]

        # Before/after span: no span_id
        assert before.get("span_id") is None
        assert after.get("span_id") is None

        # Inside span: has span_id
        assert inside.get("span_id") is not None
        assert span_start.get("span_id") is not None

        # Span start/end logged
        assert "span.start" in span_start["body"]
        assert "span.end" in span_end["body"]

    def test_span_nesting(self, log_lines):
        def emit(l):
            with l.span("outer"):
                l.info("in outer")
                with l.span("inner"):
                    l.info("in inner")
                l.info("back in outer")

        entries = log_lines(emit)
        # Find the "in inner" and "in outer" entries
        in_outer = [e for e in entries if e["body"] == "in outer"]
        in_inner = [e for e in entries if e["body"] == "in inner"]
        back_outer = [e for e in entries if e["body"] == "back in outer"]

        assert len(in_outer) == 1
        assert len(in_inner) == 1

        # Different span IDs for outer vs inner
        assert in_outer[0]["span_id"] != in_inner[0]["span_id"]
        # After inner span, back to outer span_id
        assert back_outer[0]["span_id"] == in_outer[0]["span_id"]

    def test_span_logs_duration(self, log_lines):
        import time

        def emit(l):
            with l.span("timed"):
                time.sleep(0.05)

        entries = log_lines(emit)
        end_entry = [e for e in entries if "span.end" in e["body"]][0]
        assert "duration_seconds" in end_entry["attributes"]
        assert end_entry["attributes"]["duration_seconds"] >= 0.04

    def test_span_logs_error(self, log_lines):
        def emit(l):
            try:
                with l.span("failing"):
                    raise ValueError("something broke")
            except ValueError:
                pass

        entries = log_lines(emit)
        error_entry = [e for e in entries if "span.error" in e["body"]]
        assert len(error_entry) == 1
        assert error_entry[0]["severity"] == "ERROR"
        assert "something broke" in error_entry[0]["attributes"]["error"]

    def test_span_with_attributes(self, log_lines):
        def emit(l):
            with l.span("training", model="mlp", epochs=10) as s:
                s.set_attribute("dataset.size", 5000)
                l.info("training")

        entries = log_lines(emit)
        start = [e for e in entries if "span.start" in e["body"]][0]
        assert start["attributes"]["model"] == "mlp"


# ============================================================================
# File Output
# ============================================================================

class TestFileOutput:
    def test_creates_jsonl_file(self, logger, tmp_path):
        logger.info("test")
        for h in logger._logger.handlers:
            h.flush()

        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        assert files[0].suffix == ".jsonl"

    def test_each_line_is_valid_json(self, logger, tmp_path):
        logger.info("first")
        logger.info("second")
        logger.error("third")
        for h in logger._logger.handlers:
            h.flush()

        content = list(tmp_path.glob("*.jsonl"))[0].read_text()
        lines = content.strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert "timestamp" in parsed

    def test_file_appendable(self, logger, tmp_path):
        """Multiple log calls append to same file."""
        for i in range(10):
            logger.info(f"line {i}")
        for h in logger._logger.handlers:
            h.flush()

        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().splitlines()
        assert len(lines) == 10


# ============================================================================
# Otel Collector Compatibility
# ============================================================================

class TestOtelCollectorCompat:
    """Verify logs can be parsed by otel collector's filelog receiver."""

    def test_timestamp_parseable_by_collector(self, log_lines):
        """Otel collector expects ISO 8601 timestamps."""
        entries = log_lines(lambda l: l.info("test"))
        ts = entries[0]["timestamp"]
        # Must be ISO 8601 with timezone
        assert "T" in ts
        assert ts.endswith("Z") or "+" in ts or "-" in ts[-6:]

    def test_severity_is_standard_text(self, log_lines):
        """Collector's severity parser expects standard level names."""
        entries = log_lines(lambda l: l.warning("test"))
        assert entries[0]["severity"] in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    def test_attributes_are_flat_or_simple(self, log_lines):
        """Attributes should be scalar values (strings, numbers, bools) for collector parsing."""
        entries = log_lines(lambda l: l.info("test", count=5, name="foo", flag=True))
        attrs = entries[0]["attributes"]
        assert attrs["count"] == 5
        assert attrs["name"] == "foo"
        assert attrs["flag"] is True

    def test_no_binary_data(self, log_lines):
        """JSON output should be text-safe (no binary blobs)."""
        entries = log_lines(lambda l: l.info("test"))
        raw = json.dumps(entries[0])
        # Should round-trip cleanly
        reparsed = json.loads(raw)
        assert reparsed["body"] == "test"


# ============================================================================
# Factory
# ============================================================================

class TestFactory:
    def test_get_logger_caches(self, tmp_path):
        l1 = get_logger("exp1", log_dir=tmp_path, also_stderr=False)
        l2 = get_logger("exp1", log_dir=tmp_path, also_stderr=False)
        assert l1 is l2

    def test_different_experiments_different_loggers(self, tmp_path):
        l1 = get_logger("exp1", log_dir=tmp_path / "a", also_stderr=False)
        l2 = get_logger("exp2", log_dir=tmp_path / "b", also_stderr=False)
        assert l1 is not l2

    def test_different_attempts_different_loggers(self, tmp_path):
        l1 = get_logger("exp1", log_dir=tmp_path / "a", attempt=1, also_stderr=False)
        l2 = get_logger("exp1", log_dir=tmp_path / "b", attempt=2, also_stderr=False)
        assert l1 is not l2

    def test_reset_clears_cache(self, tmp_path):
        l1 = get_logger("exp1", log_dir=tmp_path, also_stderr=False)
        reset_loggers()
        l2 = get_logger("exp1", log_dir=tmp_path / "new", also_stderr=False)
        assert l1 is not l2
