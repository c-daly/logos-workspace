"""Tests for harness-run experiment runner."""

import subprocess
import tempfile
from pathlib import Path

import yaml
import pytest


def _write_goal(tmp_path: Path, goal: dict) -> Path:
    """Write a goal.yaml and return the experiment dir."""
    exp_dir = tmp_path / "experiments" / "test_exp"
    exp_dir.mkdir(parents=True)
    (exp_dir / "goal.yaml").write_text(yaml.dump(goal))
    return exp_dir


class TestGoalParsing:
    def test_integration_experiment_has_target(self, tmp_path):
        from harness.run import load_goal

        exp_dir = _write_goal(tmp_path, {
            "objective": "Add retry to EventBus",
            "target": "logos/logos_events/event_bus.py",
            "eval": "eval/",
            "success_criteria": [{"metric": "test_pass_rate", "threshold": 1.0, "primary": True}],
        })

        goal = load_goal(exp_dir)
        assert goal["target"] == "logos/logos_events/event_bus.py"
        assert goal.is_integration is True

    def test_standalone_experiment_no_target(self, tmp_path):
        from harness.run import load_goal

        exp_dir = _write_goal(tmp_path, {
            "objective": "Train a model",
            "eval": "eval/evaluate.py",
            "success_criteria": [{"metric": "accuracy", "threshold": 0.9, "primary": True}],
        })

        goal = load_goal(exp_dir)
        assert goal.get("target") is None
        assert goal.is_integration is False


class TestWorktreeSetup:
    def test_creates_worktree_for_integration(self, tmp_path):
        """Integration experiment gets a worktree on the target repo."""
        from harness.run import setup_worktree

        # Create a bare git repo to act as target
        repo_dir = tmp_path / "logos"
        repo_dir.mkdir()
        subprocess.run(["git", "init", str(repo_dir)], check=True, capture_output=True)
        (repo_dir / "event_bus.py").write_text("class EventBus: pass")
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "init"], check=True, capture_output=True)

        result = setup_worktree(repo_dir=repo_dir, experiment_name="test_retry")

        assert result.worktree_path.exists()
        assert result.branch_name == "exp/test_retry"
        assert (result.worktree_path / "event_bus.py").exists()

    def test_cleanup_removes_worktree(self, tmp_path):
        """cleanup_worktree removes the worktree but keeps the branch."""
        from harness.run import setup_worktree, cleanup_worktree

        repo_dir = tmp_path / "logos"
        repo_dir.mkdir()
        subprocess.run(["git", "init", str(repo_dir)], check=True, capture_output=True)
        (repo_dir / "dummy.py").write_text("x = 1")
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "init"], check=True, capture_output=True)

        result = setup_worktree(repo_dir=repo_dir, experiment_name="test_exp")
        wt_path = result.worktree_path
        assert wt_path.exists()

        cleanup_worktree(repo_dir=repo_dir, worktree_path=wt_path)
        assert not wt_path.exists()

        # Branch should still exist
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "branch", "--list", "exp/test_exp"],
            capture_output=True, text=True,
        )
        assert "exp/test_exp" in out.stdout

    def test_returns_none_for_standalone(self, tmp_path):
        """Standalone experiments don't get a worktree."""
        from harness.run import setup_worktree
        result = setup_worktree(repo_dir=None, experiment_name="standalone")
        assert result is None
