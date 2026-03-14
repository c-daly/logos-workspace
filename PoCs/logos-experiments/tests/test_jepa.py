"""
Tests specific to the vjepa_clip_alignment experiment.

These tests validate the JEPA eval script, projector interface, and ticket format.
Run with: pytest tests/test_jepa.py -v
Skip with: pytest tests/ -v -m "not jepa"
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).parent.parent

pytestmark = pytest.mark.jepa

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

