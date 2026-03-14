"""
Tests for the LOGOS Experiment Harness.

Run:
    pytest tests/ -v
    pytest tests/ -v -k journal
    pytest tests/ -v -k monitor
    pytest tests/ -v -k eval
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml


# ============================================================================
# Fixtures
# ============================================================================

ROOT = Path(__file__).parent.parent


@pytest.fixture
def tmp_experiments(tmp_path):
    """Create a temporary experiments directory and patch the harness to use it."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    return exp_dir


@pytest.fixture
def sample_experiment(tmp_experiments):
    """Create a minimal experiment structure."""
    exp = tmp_experiments / "test_exp"
    for d in ["journal", "eval", "workspace", "checkpoints"]:
        (exp / d).mkdir(parents=True)

    (exp / "goal.yaml").write_text(textwrap.dedent("""\
        objective: |
          Test objective.
        success_criteria:
          - metric: accuracy
            threshold: 0.90
            primary: true
        eval: eval/evaluate.py
    """))

    (exp / "status.yaml").write_text(textwrap.dedent("""\
        experiment: test_exp
        current_attempt: 0
        status: not_started
        total_gpu_hours: 0.0
        total_attempts: 0
        last_updated: null
    """))

    return exp




# ============================================================================
# Journal Tests
# ============================================================================

class TestJournal:
    def test_empty_journal(self, sample_experiment):
        from harness.journal import Journal
        j = Journal.__new__(Journal)
        j.experiment_name = "test_exp"
        j.journal_dir = sample_experiment / "journal"

        assert j.list_entries() == []
        assert j.next_number() == 1
        assert "No experiments recorded" in j.summary()

    def test_add_entry(self, sample_experiment):
        from harness.journal import Journal
        j = Journal.__new__(Journal)
        j.experiment_name = "test_exp"
        j.journal_dir = sample_experiment / "journal"

        path = j.add_entry(
            title="Baseline MLP",
            hypothesis="Simple MLP should get 0.3+",
            changes="Created workspace/train.py",
            results="cosine_sim=0.35",
            diagnosis="Works but low — need better loss",
            next_direction="Try contrastive loss",
            training_time="12 min",
        )

        assert path.exists()
        assert path.name == "001_baseline_mlp.md"
        content = path.read_text()
        assert "Baseline MLP" in content
        assert "cosine_sim=0.35" in content

    def test_sequential_numbering(self, sample_experiment):
        from harness.journal import Journal
        j = Journal.__new__(Journal)
        j.experiment_name = "test_exp"
        j.journal_dir = sample_experiment / "journal"

        j.add_entry("First", "h", "c", "r", "d", "n")
        j.add_entry("Second", "h", "c", "r", "d", "n")
        j.add_entry("Third", "h", "c", "r", "d", "n")

        entries = j.list_entries()
        assert len(entries) == 3
        assert entries[0].name.startswith("001_")
        assert entries[1].name.startswith("002_")
        assert entries[2].name.startswith("003_")
        assert j.next_number() == 4

    def test_add_raw(self, sample_experiment):
        from harness.journal import Journal
        j = Journal.__new__(Journal)
        j.experiment_name = "test_exp"
        j.journal_dir = sample_experiment / "journal"

        path = j.add_raw("Custom Entry", "# My custom content\nFreeform text here.")
        assert path.exists()
        assert "custom_entry" in path.name
        assert "Freeform text" in path.read_text()

    def test_summary_extracts_fields(self, sample_experiment):
        from harness.journal import Journal
        j = Journal.__new__(Journal)
        j.experiment_name = "test_exp"
        j.journal_dir = sample_experiment / "journal"

        j.add_entry(
            title="Test Run",
            hypothesis="Testing",
            changes="Changed stuff",
            results="accuracy=0.95",
            diagnosis="Worked great",
            next_direction="Ship it",
        )

        summary = j.summary()
        assert "Test Run" in summary
        assert "accuracy=0.95" in summary
        assert "Total attempts: 1" in summary

    def test_failures_summary_filters(self, sample_experiment):
        from harness.journal import Journal
        j = Journal.__new__(Journal)
        j.experiment_name = "test_exp"
        j.journal_dir = sample_experiment / "journal"

        j.add_entry("Good Run", "h", "c", "r", "It worked perfectly", "n")
        j.add_entry("Bad Run", "h", "c", "r", "NaN explosion at batch 2", "n")
        j.add_entry("Another Bad", "h", "c", "r", "Model failed to converge", "n")

        failures = j.failures_summary()
        assert "NaN explosion" in failures
        assert "failed to converge" in failures
        # The good run shouldn't appear
        assert "worked perfectly" not in failures

    def test_gitkeep_ignored(self, sample_experiment):
        from harness.journal import Journal
        j = Journal.__new__(Journal)
        j.experiment_name = "test_exp"
        j.journal_dir = sample_experiment / "journal"

        (j.journal_dir / ".gitkeep").touch()
        assert j.list_entries() == []


# ============================================================================
# Monitor Tests
# ============================================================================

class TestMonitor:
    def test_successful_run(self):
        from harness.monitor import TrainingMonitor
        monitor = TrainingMonitor(timeout_hours=0.01)
        result = monitor.run([sys.executable, "-c", "print('hello')"])

        assert result.exit_code == 0
        assert result.succeeded is True
        assert result.nan_detected is False
        assert result.killed_reason is None

    def test_captures_metrics(self):
        from harness.monitor import TrainingMonitor
        script = "print('[METRIC] loss=0.5'); print('[METRIC] accuracy=0.9')"
        monitor = TrainingMonitor(timeout_hours=0.01)
        result = monitor.run([sys.executable, "-c", script])

        assert result.succeeded is True
        assert "loss" in result.metrics_captured
        assert "accuracy" in result.metrics_captured
        assert result.metrics_captured["loss"] == [0.5]
        assert result.metrics_captured["accuracy"] == [0.9]

    def test_captures_multiple_metric_values(self):
        from harness.monitor import TrainingMonitor
        script = (
            "import time\n"
            "print('[METRIC] loss=1.0'); time.sleep(0.05)\n"
            "print('[METRIC] loss=0.5'); time.sleep(0.05)\n"
            "print('[METRIC] loss=0.1'); time.sleep(0.05)\n"
        )
        monitor = TrainingMonitor(timeout_hours=0.01)
        result = monitor.run([sys.executable, "-c", script])

        assert result.metrics_captured["loss"] == [1.0, 0.5, 0.1]

    def test_nan_detection_kills_run(self):
        from harness.monitor import TrainingMonitor
        script = textwrap.dedent("""\
            import time
            for i in range(20):
                if i < 3:
                    print(f"step={i} loss=0.5")
                else:
                    print(f"step={i} loss=nan")
                time.sleep(0.05)
        """)
        monitor = TrainingMonitor(timeout_hours=0.01, nan_patience=3)
        result = monitor.run([sys.executable, "-c", script])

        assert result.nan_detected is True
        assert result.succeeded is False
        assert result.killed_reason is not None
        assert "nan" in result.killed_reason.lower()

    def test_nan_resets_on_clean_line(self):
        from harness.monitor import TrainingMonitor
        # One NaN then clean lines — should NOT kill, but nan_detected
        # reflects that NaN was ever seen (nan_ever_seen semantics)
        script = (
            "import time\n"
            "print('step=0 loss=nan'); time.sleep(0.05)\n"
            "print('step=1 loss=0.5'); time.sleep(0.05)\n"
            "print('step=2 loss=0.4'); time.sleep(0.05)\n"
        )
        monitor = TrainingMonitor(timeout_hours=0.01, nan_patience=3)
        result = monitor.run([sys.executable, "-c", script])

        assert result.killed_reason is None  # Didn't hit patience
        assert result.nan_detected is True    # NaN was seen (ever)
        assert result.exit_code == 0          # Clean exit

    def test_nonzero_exit_code(self):
        from harness.monitor import TrainingMonitor
        monitor = TrainingMonitor(timeout_hours=0.01)
        result = monitor.run([sys.executable, "-c", "import sys; sys.exit(1)"])

        assert result.exit_code == 1
        assert result.succeeded is False

    def test_run_result_to_dict(self):
        from harness.monitor import RunResult
        r = RunResult(
            exit_code=0,
            duration_seconds=123.456,
            killed_reason=None,
            nan_detected=False,
            metrics_captured={"loss": [0.1]},
        )
        d = r.to_dict()
        assert d["exit_code"] == 0
        assert d["duration_seconds"] == 123.5
        assert d["succeeded"] is True
        assert d["metrics"]["loss"] == [0.1]

    def test_timeout(self):
        from harness.monitor import TrainingMonitor
        script = "import time; time.sleep(60)"
        # Very short timeout
        monitor = TrainingMonitor(timeout_hours=0.001, nan_patience=10)
        result = monitor.run([sys.executable, "-c", script])

        assert result.succeeded is False
        assert result.killed_reason is not None
        assert "timeout" in result.killed_reason.lower()



# ============================================================================
# New Experiment Tests
# ============================================================================

class TestNewExperiment:
    def test_creates_structure(self, tmp_path, monkeypatch):
        import harness.new as new_mod
        monkeypatch.setattr(new_mod, "EXPERIMENTS_DIR", tmp_path)

        new_mod.create_experiment("my_exp", "Solve world hunger")

        exp = tmp_path / "my_exp"
        assert exp.exists()
        assert (exp / "goal.yaml").exists()
        assert (exp / "status.yaml").exists()
        assert (exp / "eval" / "evaluate.py").exists()
        assert (exp / "journal").is_dir()
        assert (exp / "workspace").is_dir()
        assert (exp / "checkpoints").is_dir()

    def test_goal_yaml_has_objective(self, tmp_path, monkeypatch):
        import harness.new as new_mod
        monkeypatch.setattr(new_mod, "EXPERIMENTS_DIR", tmp_path)

        new_mod.create_experiment("my_exp", "Classify cats and dogs")

        goal = yaml.safe_load((tmp_path / "my_exp" / "goal.yaml").read_text())
        assert "Classify cats and dogs" in goal["objective"]

    def test_goal_yaml_ticket_format(self, tmp_path, monkeypatch):
        import harness.new as new_mod
        monkeypatch.setattr(new_mod, "EXPERIMENTS_DIR", tmp_path)

        new_mod.create_experiment("my_exp", "Test goal")

        goal = yaml.safe_load((tmp_path / "my_exp" / "goal.yaml").read_text())
        # Required fields
        assert "objective" in goal
        assert "success_criteria" in goal
        assert "eval" in goal

    def test_duplicate_experiment_fails(self, tmp_path, monkeypatch):
        import harness.new as new_mod
        monkeypatch.setattr(new_mod, "EXPERIMENTS_DIR", tmp_path)

        new_mod.create_experiment("my_exp", "First")
        with pytest.raises(SystemExit):
            new_mod.create_experiment("my_exp", "Duplicate")

    def test_no_constraints_by_default(self, tmp_path, monkeypatch):
        import harness.new as new_mod
        monkeypatch.setattr(new_mod, "EXPERIMENTS_DIR", tmp_path)

        new_mod.create_experiment("my_exp", "Test")

        # constraints.yaml should NOT be created by default
        assert not (tmp_path / "my_exp" / "constraints.yaml").exists()



# ============================================================================
# Integration: CLAUDE.md
# ============================================================================

class TestClaudeMd:
    def test_exists(self):
        assert (ROOT / "CLAUDE.md").exists()

    def test_references_ticket(self):
        content = (ROOT / "CLAUDE.md").read_text()
        assert "goal.yaml" in content
        assert "ticket" in content.lower()

    def test_references_journal(self):
        content = (ROOT / "CLAUDE.md").read_text()
        assert "journal" in content.lower()

    def test_references_eval(self):
        content = (ROOT / "CLAUDE.md").read_text()
        assert "eval" in content

    def test_references_agent_teams(self):
        content = (ROOT / "CLAUDE.md").read_text()
        assert "agent team" in content.lower()

    def test_no_approach_prescription(self):
        content = (ROOT / "CLAUDE.md").read_text().lower()
        # CLAUDE.md should not prescribe specific ML approaches
        for phrase in ["you should use", "start with procrustes",
                       "try mlp first", "use pytorch"]:
            assert phrase not in content, f"CLAUDE.md should not prescribe: '{phrase}'"
