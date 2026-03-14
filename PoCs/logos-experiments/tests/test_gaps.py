"""
Tests for critical path gaps identified in coverage audit.

Covers:
- harness/metrics.py (was completely untested)
- Monitor NaN regex and metric capture patterns (were only integration-tested)
- Journal numbering with gaps
- pytest_eval output parsing edge cases
- Eval describe_npy function
"""

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent


# ============================================================================
# Metrics Module (was 0% covered)
# ============================================================================

class TestMetrics:
    def test_load_empty_dir(self, tmp_path):
        from harness.metrics import load_attempt_results
        # No logs dir at all
        results = load_attempt_results(tmp_path)
        assert results == {}

    def test_load_empty_logs(self, tmp_path):
        from harness.metrics import load_attempt_results
        (tmp_path / "logs").mkdir()
        results = load_attempt_results(tmp_path)
        assert results == {}

    def test_load_single_attempt(self, tmp_path):
        from harness.metrics import load_attempt_results
        attempt_dir = tmp_path / "logs" / "attempt_001"
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "result.json").write_text(json.dumps({
            "exit_code": 0,
            "succeeded": True,
            "metrics": {"loss": [0.5, 0.3, 0.1]},
        }))
        results = load_attempt_results(tmp_path)
        assert 1 in results
        assert results[1]["succeeded"] is True

    def test_load_multiple_attempts(self, tmp_path):
        from harness.metrics import load_attempt_results
        for i in [1, 2, 5]:
            d = tmp_path / "logs" / f"attempt_{i:03d}"
            d.mkdir(parents=True)
            (d / "result.json").write_text(json.dumps({
                "exit_code": 0 if i == 5 else 1,
                "succeeded": i == 5,
                "metrics": {"loss": [1.0 / i]},
            }))

        results = load_attempt_results(tmp_path)
        assert set(results.keys()) == {1, 2, 5}
        assert results[5]["succeeded"] is True
        assert results[1]["succeeded"] is False

    def test_skips_malformed_json(self, tmp_path):
        from harness.metrics import load_attempt_results
        d = tmp_path / "logs" / "attempt_001"
        d.mkdir(parents=True)
        (d / "result.json").write_text("not json{{{")

        results = load_attempt_results(tmp_path)
        assert results == {}

    def test_skips_non_directories(self, tmp_path):
        from harness.metrics import load_attempt_results
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "README.md").write_text("ignore me")

        results = load_attempt_results(tmp_path)
        assert results == {}

    def test_show_overview_no_results(self, tmp_path, capsys, monkeypatch):
        from harness.metrics import show_overview, EXPERIMENTS_DIR
        import harness.metrics as metrics_mod
        monkeypatch.setattr(metrics_mod, "EXPERIMENTS_DIR", tmp_path)
        (tmp_path / "empty_exp").mkdir()

        show_overview("empty_exp")
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_show_overview_with_results(self, tmp_path, capsys, monkeypatch):
        from harness.metrics import show_overview
        import harness.metrics as metrics_mod
        monkeypatch.setattr(metrics_mod, "EXPERIMENTS_DIR", tmp_path)

        exp = tmp_path / "my_exp"
        d = exp / "logs" / "attempt_001"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "succeeded": True,
            "duration_human": "0:05:00",
            "nan_detected": False,
            "metrics": {"loss": [0.5, 0.1]},
        }))

        show_overview("my_exp")
        captured = capsys.readouterr()
        assert "METRICS OVERVIEW" in captured.out
        assert "my_exp" in captured.out


# ============================================================================
# Monitor Internals (NaN regex, metric capture — were only integration-tested)
# ============================================================================

class TestMonitorInternals:
    def _get_monitor(self):
        from harness.monitor import TrainingMonitor
        return TrainingMonitor()

    def test_check_nan_loss_nan(self):
        m = self._get_monitor()
        assert m._check_nan("step=10 loss=nan") is True
        assert m._check_nan("step=10 loss=NaN") is True
        assert m._check_nan("step=10 loss=NAN") is True

    def test_check_nan_loss_inf(self):
        m = self._get_monitor()
        assert m._check_nan("step=10 loss=inf") is True
        assert m._check_nan("step=10 loss=Inf") is True

    def test_check_nan_normal_loss(self):
        m = self._get_monitor()
        assert m._check_nan("step=10 loss=0.5") is False
        assert m._check_nan("step=10 loss=1e-4") is False

    def test_check_nan_ignores_non_metric_lines(self):
        m = self._get_monitor()
        # "nan" in random text without metric keywords should not trigger
        assert m._check_nan("loading nantucket dataset") is False
        assert m._check_nan("information about the model") is False

    def test_check_nan_requires_metric_context(self):
        m = self._get_monitor()
        # Must contain keywords like loss, step, epoch, batch, metric
        assert m._check_nan("epoch 5 loss: nan") is True
        assert m._check_nan("batch 3: loss=inf") is True
        assert m._check_nan("random text nan") is False

    def test_capture_structured_metrics(self):
        m = self._get_monitor()
        metrics = {}
        m._capture_metrics("[METRIC] loss=0.5", metrics)
        m._capture_metrics("[METRIC] accuracy=0.9", metrics)
        m._capture_metrics("[METRIC] loss=0.3", metrics)

        assert metrics["loss"] == [0.5, 0.3]
        assert metrics["accuracy"] == [0.9]

    def test_capture_heuristic_metrics(self):
        m = self._get_monitor()
        metrics = {}
        m._capture_metrics("step 100: loss=0.5 lr=0.001", metrics)

        assert "loss" in metrics
        assert metrics["loss"] == [0.5]

    def test_capture_scientific_notation(self):
        m = self._get_monitor()
        metrics = {}
        m._capture_metrics("[METRIC] lr=1e-4", metrics)
        m._capture_metrics("[METRIC] loss=3.5e-2", metrics)

        assert metrics["lr"] == [1e-4]
        assert metrics["loss"] == [3.5e-2]

    def test_capture_ignores_non_metric_keys(self):
        m = self._get_monitor()
        metrics = {}
        # Heuristic capture only grabs known keys
        m._capture_metrics("filename=test123 version=2", metrics)
        assert "filename" not in metrics
        assert "version" not in metrics


# ============================================================================
# Journal Edge Cases
# ============================================================================

class TestJournalEdgeCases:
    def test_numbering_with_gaps(self, tmp_path):
        """If entries 001 and 003 exist (002 deleted), next should be 004."""
        from harness.journal import Journal
        j = Journal.__new__(Journal)
        j.experiment_name = "test"
        j.journal_dir = tmp_path

        (tmp_path / "001_first.md").write_text("# Attempt 001: First")
        (tmp_path / "003_third.md").write_text("# Attempt 003: Third")
        # 002 is missing (deleted)

        assert j.next_number() == 4

    def test_numbering_with_non_numeric_files(self, tmp_path):
        """Non-numeric .md files shouldn't break numbering."""
        from harness.journal import Journal
        j = Journal.__new__(Journal)
        j.experiment_name = "test"
        j.journal_dir = tmp_path

        (tmp_path / "001_first.md").write_text("# First")
        (tmp_path / "README.md").write_text("# Not an entry")
        (tmp_path / "notes.md").write_text("Random notes")

        # list_entries returns all .md files sorted
        entries = j.list_entries()
        assert len(entries) == 3  # All .md files

        # But numbering only looks at numeric prefixes
        assert j.next_number() == 2

    def test_slug_special_characters(self, tmp_path):
        """Title with special chars should produce a clean filename."""
        from harness.journal import Journal
        j = Journal.__new__(Journal)
        j.experiment_name = "test"
        j.journal_dir = tmp_path

        path = j.add_entry(
            title="NaN @ Batch #2 — Fixed!",
            hypothesis="h", changes="c", results="r",
            diagnosis="d", next_direction="n",
        )

        assert ".md" in path.name
        assert " " not in path.name
        assert "@" not in path.name
        assert "#" not in path.name

    def test_summary_with_missing_fields(self, tmp_path):
        """Journal entries missing expected fields shouldn't crash summary."""
        from harness.journal import Journal
        j = Journal.__new__(Journal)
        j.experiment_name = "test"
        j.journal_dir = tmp_path

        (tmp_path / "001_minimal.md").write_text("# Just a title\nSome text.")

        summary = j.summary()
        assert "Just a title" in summary

    def test_failures_with_no_failures(self, tmp_path):
        """When all entries are successes, failures summary says so."""
        from harness.journal import Journal
        j = Journal.__new__(Journal)
        j.experiment_name = "test"
        j.journal_dir = tmp_path

        j.add_entry("Success", "h", "c", "r", "Everything worked perfectly", "n")

        failures = j.failures_summary()
        assert "No clear failures" in failures


# ============================================================================
# Pytest Eval Parsing
# ============================================================================

class TestPytestEvalParsing:
    def test_parse_all_passed(self):
        from harness.pytest_eval import run_pytest
        # Create a trivial passing test
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test_a(): pass\ndef test_b(): pass\n")
            path = f.name

        try:
            results = run_pytest(path)
            assert results["tests_passed"] == 2
            assert results["tests_failed"] == 0
            assert results["pass_rate"] == 1.0
            assert results["exit_code"] == 0
        finally:
            import os
            os.unlink(path)

    def test_parse_mixed_results(self):
        from harness.pytest_eval import run_pytest
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test_pass(): pass\ndef test_fail(): assert False\n")
            path = f.name

        try:
            results = run_pytest(path)
            assert results["tests_passed"] == 1
            assert results["tests_failed"] == 1
            assert results["pass_rate"] == 0.5
        finally:
            import os
            os.unlink(path)

    def test_parse_all_failed(self):
        from harness.pytest_eval import run_pytest
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test_a(): assert False\ndef test_b(): assert 1==2\n")
            path = f.name

        try:
            results = run_pytest(path)
            assert results["tests_passed"] == 0
            assert results["tests_failed"] == 2
            assert results["pass_rate"] == 0.0
        finally:
            import os
            os.unlink(path)

    def test_parse_empty_test_file(self):
        from harness.pytest_eval import run_pytest
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# no tests here\n")
            path = f.name

        try:
            results = run_pytest(path)
            assert results["tests_total"] == 0
        finally:
            import os
            os.unlink(path)


# ============================================================================
# Eval: describe_npy
# ============================================================================

class TestDescribeNpy:
    def _import_eval(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "test_alignment",
            ROOT / "experiments" / "vjepa_clip_alignment" / "eval" / "test_alignment.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_describe_normal(self, tmp_path):
        mod = self._import_eval()
        arr = np.random.randn(50, 128).astype(np.float32)
        path = tmp_path / "test.npy"
        np.save(path, arr)

        info = mod.describe_npy(path)
        assert info["shape"] == (50, 128)
        assert info["dtype"] == "float32"
        assert info["has_nan"] is False
        assert info["has_inf"] is False
        assert isinstance(info["min"], float)
        assert isinstance(info["max"], float)
        assert isinstance(info["mean"], float)

    def test_describe_with_nan(self, tmp_path):
        mod = self._import_eval()
        arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        path = tmp_path / "nan.npy"
        np.save(path, arr)

        info = mod.describe_npy(path)
        assert info["has_nan"] is True

    def test_describe_with_inf(self, tmp_path):
        mod = self._import_eval()
        arr = np.array([[1.0, np.inf], [3.0, 4.0]], dtype=np.float32)
        path = tmp_path / "inf.npy"
        np.save(path, arr)

        info = mod.describe_npy(path)
        assert info["has_inf"] is True

    def test_describe_integer_array(self, tmp_path):
        mod = self._import_eval()
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        path = tmp_path / "int.npy"
        np.save(path, arr)

        info = mod.describe_npy(path)
        assert info["dtype"] == "int32"
        assert info["has_nan"] is False  # Integers can't be NaN


# ============================================================================
# Goal.yaml Loading Edge Cases
# ============================================================================

class TestGoalLoading:
    def test_minimal_ticket(self):
        """Absolute minimum ticket should be loadable."""
        ticket = yaml.safe_load(textwrap.dedent("""\
            objective: Do the thing.
            success_criteria:
              - metric: score
                threshold: 0.9
                primary: true
            eval: eval/evaluate.py
        """))
        assert ticket["objective"] == "Do the thing."
        assert len(ticket["success_criteria"]) == 1

    def test_ticket_with_all_optional_fields(self):
        """Ticket with everything filled in."""
        import yaml
        ticket = yaml.safe_load(textwrap.dedent("""\
            objective: Do the thing.
            success_criteria:
              - metric: score
                threshold: 0.9
                primary: true
            eval: eval/evaluate.py
            context: Some background.
            resources:
              - https://example.com
              - https://example.org
            prior_work: Tried X, it failed.
            notes: Be careful with Y.
        """))
        assert ticket["context"] == "Some background."
        assert len(ticket["resources"]) == 2
        assert "Tried X" in ticket["prior_work"]

    def test_ticket_multiple_criteria(self):
        import yaml
        ticket = yaml.safe_load(textwrap.dedent("""\
            objective: Multi-metric goal.
            success_criteria:
              - metric: precision
                threshold: 0.9
                primary: true
              - metric: recall
                threshold: 0.8
              - metric: latency_ms
                threshold: 100
            eval: eval/evaluate.py
        """))
        assert len(ticket["success_criteria"]) == 3
        primary = [c for c in ticket["success_criteria"] if c.get("primary")]
        assert len(primary) == 1
        assert primary[0]["metric"] == "precision"


# Need yaml import for TestGoalLoading
import yaml
