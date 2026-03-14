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


@pytest.fixture
def projector_file(tmp_path):
    """Create a minimal projector module."""
    code = textwrap.dedent("""\
        import numpy as np
        
        def load(checkpoint_path=None):
            return {"d_out": 512}
        
        def project(state, vjepa_embeddings):
            d_out = state["d_out"]
            rng = np.random.RandomState(42)
            W = rng.randn(vjepa_embeddings.shape[1], d_out).astype(np.float32)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            return (vjepa_embeddings @ W).astype(np.float32)
    """)
    path = tmp_path / "projector.py"
    path.write_text(code)
    return path


@pytest.fixture
def perfect_projector_file(tmp_path):
    """Create a projector that achieves perfect alignment (for testing pass/fail)."""
    code = textwrap.dedent("""\
        import numpy as np
        _clip_embs = None
        
        def load(checkpoint_path=None):
            # Cheat: load the actual CLIP embeddings and return them directly
            global _clip_embs
            if checkpoint_path:
                _clip_embs = np.load(checkpoint_path)
            return {}
        
        def project(state, vjepa_embeddings):
            global _clip_embs
            if _clip_embs is not None:
                return _clip_embs
            # Fallback: return zeros with right shape (will fail)
            return np.zeros((vjepa_embeddings.shape[0], 512), dtype=np.float32)
    """)
    path = tmp_path / "perfect_projector.py"
    path.write_text(code)
    return path


@pytest.fixture
def nan_projector_file(tmp_path):
    """Create a projector that produces NaN."""
    code = textwrap.dedent("""\
        import numpy as np
        
        def load(checkpoint_path=None):
            return None
        
        def project(state, vjepa_embeddings):
            out = np.full((vjepa_embeddings.shape[0], 512), np.nan, dtype=np.float32)
            return out
    """)
    path = tmp_path / "nan_projector.py"
    path.write_text(code)
    return path


@pytest.fixture
def wrong_dim_projector_file(tmp_path):
    """Create a projector that returns wrong dimensions."""
    code = textwrap.dedent("""\
        import numpy as np
        
        def load(checkpoint_path=None):
            return None
        
        def project(state, vjepa_embeddings):
            return np.zeros((vjepa_embeddings.shape[0], 999), dtype=np.float32)
    """)
    path = tmp_path / "wrong_dim_projector.py"
    path.write_text(code)
    return path


@pytest.fixture
def paired_embeddings(tmp_path):
    """Create paired V-JEPA and CLIP embeddings with known alignment."""
    rng = np.random.RandomState(99)  # Different seed from projector fixture
    n, d_vjepa, d_clip = 100, 768, 512

    # Create embeddings with a known linear relationship
    W = rng.randn(d_vjepa, d_clip).astype(np.float32)
    W /= np.linalg.norm(W, axis=0, keepdims=True)

    vjepa = rng.randn(n, d_vjepa).astype(np.float32)
    clip = vjepa @ W  # Perfect linear alignment

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.save(data_dir / "vjepa_val.npy", vjepa)
    np.save(data_dir / "clip_val.npy", clip)

    return data_dir, vjepa, clip


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
        result = monitor.run(["python", "-c", "print('hello')"])

        assert result.exit_code == 0
        assert result.succeeded is True
        assert result.nan_detected is False
        assert result.killed_reason is None

    def test_captures_metrics(self):
        from harness.monitor import TrainingMonitor
        script = "print('[METRIC] loss=0.5'); print('[METRIC] accuracy=0.9')"
        monitor = TrainingMonitor(timeout_hours=0.01)
        result = monitor.run(["python", "-c", script])

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
        result = monitor.run(["python", "-c", script])

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
        result = monitor.run(["python", "-c", script])

        assert result.nan_detected is True
        assert result.succeeded is False
        assert result.killed_reason is not None
        assert "nan" in result.killed_reason.lower()

    def test_nan_resets_on_clean_line(self):
        from harness.monitor import TrainingMonitor
        # One NaN then clean lines — should NOT kill, and nan_detected
        # reflects final state (clean) not history
        script = (
            "import time\n"
            "print('step=0 loss=nan'); time.sleep(0.05)\n"
            "print('step=1 loss=0.5'); time.sleep(0.05)\n"
            "print('step=2 loss=0.4'); time.sleep(0.05)\n"
        )
        monitor = TrainingMonitor(timeout_hours=0.01, nan_patience=3)
        result = monitor.run(["python", "-c", script])

        assert result.killed_reason is None  # Didn't hit patience
        assert result.succeeded is True       # Clean exit after recovery

    def test_nonzero_exit_code(self):
        from harness.monitor import TrainingMonitor
        monitor = TrainingMonitor(timeout_hours=0.01)
        result = monitor.run(["python", "-c", "import sys; sys.exit(1)"])

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
        result = monitor.run(["python", "-c", script])

        assert result.succeeded is False
        assert result.killed_reason is not None
        assert "timeout" in result.killed_reason.lower()


# ============================================================================
# Eval Tests (V-JEPA → CLIP)
# ============================================================================

class TestEval:
    """Tests for the experiment eval script."""

    EVAL_SCRIPT = str(ROOT / "experiments" / "vjepa_clip_alignment" / "eval" / "test_alignment.py")

    def test_synthetic_mode(self, projector_file):
        result = subprocess.run(
            [sys.executable, self.EVAL_SCRIPT,
             "--projector", str(projector_file),
             "--synthetic"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "INTERFACE OK" in result.stdout
        assert "[METRIC] mean_cosine_similarity=" in result.stdout
        assert "[METRIC] training_stability=1.0" in result.stdout

    def test_synthetic_with_custom_dims(self, projector_file):
        result = subprocess.run(
            [sys.executable, self.EVAL_SCRIPT,
             "--projector", str(projector_file),
             "--synthetic",
             "--synthetic-d-vjepa", "1024",
             "--synthetic-d-clip", "512"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "d_vjepa=1024" in result.stdout
        assert "d_clip=512" in result.stdout

    def test_nan_projector_fails(self, nan_projector_file):
        result = subprocess.run(
            [sys.executable, self.EVAL_SCRIPT,
             "--projector", str(nan_projector_file),
             "--synthetic"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0  # Synthetic mode always exits 0
        assert "training_stability=0.0" in result.stdout

    def test_wrong_dim_contract_violation(self, wrong_dim_projector_file):
        result = subprocess.run(
            [sys.executable, self.EVAL_SCRIPT,
             "--projector", str(wrong_dim_projector_file),
             "--synthetic"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "Contract violation" in result.stdout

    def test_missing_projector(self):
        result = subprocess.run(
            [sys.executable, self.EVAL_SCRIPT,
             "--projector", "/nonexistent/projector.py",
             "--synthetic"],
            capture_output=True, text=True,
        )
        assert result.returncode == 2

    def test_data_discovery(self, projector_file, paired_embeddings):
        data_dir, _, _ = paired_embeddings
        result = subprocess.run(
            [sys.executable, self.EVAL_SCRIPT,
             "--projector", str(projector_file),
             "--data-dir", str(data_dir)],
            capture_output=True, text=True,
        )
        assert "Found V-JEPA embeddings" in result.stdout
        assert "Found CLIP embeddings" in result.stdout
        assert "[METRIC]" in result.stdout
        assert "[EVAL]" in result.stdout

    def test_discover_only(self, projector_file, paired_embeddings):
        data_dir, _, _ = paired_embeddings
        result = subprocess.run(
            [sys.executable, self.EVAL_SCRIPT,
             "--projector", str(projector_file),
             "--data-dir", str(data_dir),
             "--discover-only"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "V-JEPA:" in result.stdout
        assert "CLIP:" in result.stdout
        assert "shape=" in result.stdout

    def test_no_data_helpful_error(self, projector_file, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = subprocess.run(
            [sys.executable, self.EVAL_SCRIPT,
             "--projector", str(projector_file),
             "--data-dir", str(empty_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 2
        assert "Validation data not found" in result.stdout
        assert "--synthetic" in result.stdout  # Suggests synthetic mode

    def test_perfect_alignment_passes(self, perfect_projector_file, paired_embeddings):
        data_dir, _, clip = paired_embeddings
        # Save clip embeddings as the "checkpoint" the perfect projector loads
        ckpt = data_dir / "clip_ckpt.npy"
        np.save(ckpt, clip)

        result = subprocess.run(
            [sys.executable, self.EVAL_SCRIPT,
             "--projector", str(perfect_projector_file),
             "--checkpoint", str(ckpt),
             "--data-dir", str(data_dir)],
            capture_output=True, text=True,
        )
        assert "[EVAL] PASS" in result.stdout
        assert "mean_cosine_similarity=1.0" in result.stdout

    def test_output_json(self, projector_file, tmp_path):
        out = tmp_path / "result.json"
        subprocess.run(
            [sys.executable, self.EVAL_SCRIPT,
             "--projector", str(projector_file),
             "--synthetic",
             "--output-json", str(out)],
            capture_output=True, text=True,
        )
        assert out.exists()
        data = json.loads(out.read_text())
        assert "mean_cosine_similarity" in data
        assert "synthetic" in data
        assert data["synthetic"] is True

    def test_max_samples(self, projector_file, paired_embeddings):
        data_dir, _, _ = paired_embeddings
        result = subprocess.run(
            [sys.executable, self.EVAL_SCRIPT,
             "--projector", str(projector_file),
             "--data-dir", str(data_dir),
             "--max-samples", "10"],
            capture_output=True, text=True,
        )
        assert "n_samples=10" in result.stdout


# ============================================================================
# Eval Internal Tests (import and test functions directly)
# ============================================================================

class TestEvalInternal:
    def _import_eval(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "test_alignment",
            ROOT / "experiments" / "vjepa_clip_alignment" / "eval" / "test_alignment.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_make_synthetic_shapes(self):
        mod = self._import_eval()
        vjepa, clip = mod.make_synthetic(d_vjepa=256, d_clip=128, n=50)
        assert vjepa.shape == (50, 256)
        assert clip.shape == (50, 128)
        assert vjepa.dtype == np.float32
        assert clip.dtype == np.float32

    def test_make_synthetic_deterministic(self):
        mod = self._import_eval()
        v1, c1 = mod.make_synthetic()
        v2, c2 = mod.make_synthetic()
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(c1, c2)

    def test_check_pass_logic(self):
        mod = self._import_eval()
        assert mod.check_pass({"mean_cosine_similarity": 0.75, "training_stability": 1.0})
        assert not mod.check_pass({"mean_cosine_similarity": 0.60, "training_stability": 1.0})
        assert not mod.check_pass({"mean_cosine_similarity": 0.75, "training_stability": 0.0})
        assert not mod.check_pass({})

    def test_find_embedding_files(self, paired_embeddings):
        mod = self._import_eval()
        data_dir, _, _ = paired_embeddings
        vjepa_path, clip_path = mod.find_embedding_files([data_dir])
        assert vjepa_path is not None
        assert clip_path is not None
        assert "vjepa" in vjepa_path.name.lower()
        assert "clip" in clip_path.name.lower()

    def test_find_embedding_files_not_found(self, tmp_path):
        mod = self._import_eval()
        empty = tmp_path / "empty"
        empty.mkdir()
        vjepa_path, clip_path = mod.find_embedding_files([empty])
        assert vjepa_path is None
        assert clip_path is None

    def test_evaluate_identity(self):
        """Test that identity projection on same-space data scores 1.0."""
        mod = self._import_eval()
        n, d = 50, 128
        rng = np.random.RandomState(0)
        embs = rng.randn(n, d).astype(np.float32)

        def load(cp=None): return None
        def project(state, x): return x  # Identity

        metrics = mod.evaluate(load, project, embs, embs)
        assert metrics["mean_cosine_similarity"] == 1.0
        assert metrics["retrieval_accuracy_at_5"] == 1.0
        assert metrics["training_stability"] == 1.0

    def test_evaluate_detects_nan(self):
        mod = self._import_eval()
        n, d = 50, 128
        embs = np.random.randn(n, d).astype(np.float32)

        def load(cp=None): return None
        def project(state, x): return np.full_like(x, np.nan)

        metrics = mod.evaluate(load, project, embs, embs)
        assert metrics["training_stability"] == 0.0
        assert metrics["mean_cosine_similarity"] == 0.0

    def test_evaluate_wrong_output_dim_raises(self):
        mod = self._import_eval()
        vjepa = np.random.randn(10, 768).astype(np.float32)
        clip = np.random.randn(10, 512).astype(np.float32)

        def load(cp=None): return None
        def project(state, x): return np.zeros((10, 999))

        with pytest.raises(AssertionError, match="output dim"):
            mod.evaluate(load, project, vjepa, clip)

    def test_evaluate_sample_count_mismatch_raises(self):
        mod = self._import_eval()
        vjepa = np.random.randn(10, 768).astype(np.float32)
        clip = np.random.randn(20, 512).astype(np.float32)

        def load(cp=None): return None
        def project(state, x): return x

        with pytest.raises(AssertionError, match="mismatch"):
            mod.evaluate(load, project, vjepa, clip)


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
# Ticket Format Tests
# ============================================================================

class TestTicketFormat:
    """Test the V-JEPA experiment ticket conforms to spec."""

    def test_required_fields(self):
        goal = yaml.safe_load(
            (ROOT / "experiments" / "vjepa_clip_alignment" / "goal.yaml").read_text()
        )
        assert "objective" in goal
        assert "success_criteria" in goal
        assert "eval" in goal

    def test_success_criteria_structure(self):
        goal = yaml.safe_load(
            (ROOT / "experiments" / "vjepa_clip_alignment" / "goal.yaml").read_text()
        )
        criteria = goal["success_criteria"]
        assert isinstance(criteria, list)
        assert len(criteria) >= 1

        primary = [c for c in criteria if c.get("primary")]
        assert len(primary) == 1
        assert "metric" in primary[0]
        assert "threshold" in primary[0]

    def test_optional_fields_are_truly_optional(self):
        """Verify the harness works even if optional fields are absent."""
        minimal = yaml.safe_load(textwrap.dedent("""\
            objective: Test
            success_criteria:
              - metric: accuracy
                threshold: 0.9
                primary: true
            eval: eval/evaluate.py
        """))
        assert "objective" in minimal
        # These should NOT be required
        for field in ["context", "resources", "prior_work", "notes"]:
            assert field not in minimal  # Absent is fine

    def test_no_implementation_details_in_ticket(self):
        goal = yaml.safe_load(
            (ROOT / "experiments" / "vjepa_clip_alignment" / "goal.yaml").read_text()
        )
        # The ticket should NOT prescribe how to solve it
        for forbidden in ["models", "data", "input_format", "output_format",
                          "suggested_approaches", "architecture"]:
            assert forbidden not in goal, f"Ticket should not contain '{forbidden}'"


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
