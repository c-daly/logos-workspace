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


class TestEvalExecution:
    def test_runs_pytest_eval(self, tmp_path):
        """Eval runs pytest on the eval directory and captures results."""
        from harness.run import run_eval

        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "test_trivial.py").write_text("def test_pass(): assert True\n")

        result = run_eval(eval_path="eval/", exp_dir=tmp_path, worktree_path=None)
        assert result["pass_rate"] == 1.0
        assert result["tests_passed"] == 1

    def test_eval_with_worktree_on_pythonpath(self, tmp_path):
        """Integration eval has the worktree importable."""
        from harness.run import run_eval

        worktree = tmp_path / "worktree"
        worktree.mkdir()
        (worktree / "my_module.py").write_text("VALUE = 42\n")

        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "test_import.py").write_text(
            "def test_import():\n"
            "    from my_module import VALUE\n"
            "    assert VALUE == 42\n"
        )

        result = run_eval(eval_path="eval/", exp_dir=tmp_path, worktree_path=worktree)
        assert result["pass_rate"] == 1.0


class TestPushFlag:
    def test_push_creates_pr(self, tmp_path):
        """--push pushes the branch and opens a PR via gh."""
        from unittest.mock import patch, MagicMock
        from harness.run import push_and_pr, WorktreeInfo

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        wt_info = WorktreeInfo(
            worktree_path=tmp_path / "worktree",
            branch_name="exp/test_retry",
            repo_dir=repo_dir,
        )

        eval_results = {"pass_rate": 1.0, "tests_passed": 5, "tests_total": 5}

        with patch("harness.run.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="https://github.com/pr/1\n")
            url = push_and_pr(
                wt_info=wt_info,
                experiment_name="test_retry",
                eval_results=eval_results,
            )

        # Should have called git push and gh pr create
        calls = [c.args[0] for c in mock_run.call_args_list]
        push_call = [c for c in calls if "push" in c]
        pr_call = [c for c in calls if "pr" in c[0:2] or "gh" in c]
        assert len(push_call) >= 1
        assert len(pr_call) >= 1
        assert url == "https://github.com/pr/1"

    def test_no_push_without_flag(self):
        """Default mode does not push."""
        from harness.run import push_and_pr
        # push_and_pr should not be called — this is a control flow test
        # verified in the main() integration, not here
        pass


class TestScaffoldIntegration:
    def test_new_integration_experiment_has_target(self, tmp_path, monkeypatch):
        """harness-new --target creates goal.yaml with target field."""
        from harness import new as new_module
        from harness.new import create_experiment

        monkeypatch.setattr(new_module, "EXPERIMENTS_DIR", tmp_path)

        create_experiment(
            name="retry_eventbus",
            goal="Add retry to EventBus.publish()",
            target="logos/logos_events/event_bus.py",
        )

        goal_text = (tmp_path / "retry_eventbus" / "goal.yaml").read_text()
        assert "target:" in goal_text
        assert "logos/logos_events/event_bus.py" in goal_text

    def test_new_standalone_no_target(self, tmp_path, monkeypatch):
        """harness-new without --target creates standard goal.yaml."""
        from harness import new as new_module
        from harness.new import create_experiment

        monkeypatch.setattr(new_module, "EXPERIMENTS_DIR", tmp_path)

        create_experiment(name="my_ml_exp", goal="Train a model")

        goal_text = (tmp_path / "my_ml_exp" / "goal.yaml").read_text()
        assert "target:" not in goal_text
