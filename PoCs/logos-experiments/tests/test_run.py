"""Tests for harness-run experiment runner."""

import subprocess
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

    def test_reuse_branch_cleans_orphaned_directory(self, tmp_path):
        """When branch exists and an orphaned directory is present, fallback removes it first."""
        from harness.run import setup_worktree

        repo_dir = tmp_path / "logos"
        repo_dir.mkdir()
        subprocess.run(["git", "init", str(repo_dir)], check=True, capture_output=True)
        (repo_dir / "dummy.py").write_text("x = 1")
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "init"], check=True, capture_output=True)

        # Create the branch manually so it exists
        subprocess.run(
            ["git", "-C", str(repo_dir), "branch", "exp/orphan_test"],
            check=True, capture_output=True,
        )
        # Create an orphaned directory at the expected worktree path
        orphan_path = repo_dir.parent / ".worktrees/orphan_test"
        orphan_path.mkdir(parents=True)
        (orphan_path / "leftover.py").write_text("# orphan")

        # setup_worktree should handle this without raising
        result = setup_worktree(repo_dir=repo_dir, experiment_name="orphan_test")
        assert result.worktree_path.exists()
        assert result.branch_name == "exp/orphan_test"

    def test_cleanup_skips_dirty_worktree(self, tmp_path):
        """cleanup_worktree warns and skips removal when worktree has uncommitted changes."""
        import logging
        from harness.run import setup_worktree, cleanup_worktree

        repo_dir = tmp_path / "logos"
        repo_dir.mkdir()
        subprocess.run(["git", "init", str(repo_dir)], check=True, capture_output=True)
        (repo_dir / "dummy.py").write_text("x = 1")
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "init"], check=True, capture_output=True)

        result = setup_worktree(repo_dir=repo_dir, experiment_name="dirty_exp")
        wt_path = result.worktree_path

        # Simulate uncommitted agent work in the worktree
        (wt_path / "agent_output.py").write_text("result = 42")

        # Should not raise even though worktree is dirty
        cleanup_worktree(repo_dir=repo_dir, worktree_path=wt_path)

        # Worktree should still exist — agent work preserved
        assert wt_path.exists()

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

    def test_timeout_raises_cleanly(self, tmp_path):
        """TimeoutExpired propagates from run_eval so main() can catch it cleanly."""
        from unittest.mock import patch
        from harness.run import run_eval

        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "test_trivial.py").write_text("def test_pass(): assert True\n")

        with patch("harness.run.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=1)):
            with pytest.raises(subprocess.TimeoutExpired):
                run_eval(eval_path="eval/", exp_dir=tmp_path, worktree_path=None, timeout=1)

    def test_goal_environment_applied(self, tmp_path):
        """goal.environment variables are injected into the eval subprocess env."""
        from harness.run import run_eval
        import os

        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "test_env.py").write_text(
            "import os\n"
            "def test_env_var():\n"
            "    assert os.environ.get('HARNESS_TEST_VAR') == 'hello'\n"
        )

        result = run_eval(
            eval_path="eval/",
            exp_dir=tmp_path,
            worktree_path=None,
            environment={"HARNESS_TEST_VAR": "hello"},
        )
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

    def test_push_ignored_for_standalone(self, tmp_path, capsys):
        """--push on a standalone experiment prints a clear ignored message."""
        import sys
        import yaml
        from unittest.mock import patch

        exp_dir = tmp_path / "experiments" / "test_exp"
        exp_dir.mkdir(parents=True)
        (exp_dir / "goal.yaml").write_text(yaml.dump({
            "objective": "Test",
            "eval": "eval/",
            "success_criteria": [{"metric": "test_pass_rate", "threshold": 1.0, "primary": True}],
        }))
        (exp_dir / "eval").mkdir()
        (exp_dir / "eval" / "test_pass.py").write_text("def test_ok(): assert True\n")

        with patch("harness.run.find_experiments_dir", return_value=tmp_path / "experiments"), \
             patch("sys.argv", ["harness-run", "test_exp", "--push"]):
            from harness import run as run_module
            try:
                run_module.main()
            except SystemExit:
                pass

        captured = capsys.readouterr()
        assert "[PUSH] Ignored" in captured.out

    def test_push_failure_prints_clean_message(self, tmp_path, capsys):
        """A git push failure prints [PUSH] FAILED instead of a raw traceback."""
        import sys
        import yaml
        from unittest.mock import patch
        from harness.run import WorktreeInfo

        exp_dir = tmp_path / "experiments" / "test_exp"
        exp_dir.mkdir(parents=True)
        # Integration experiment so wt_info is populated and push_and_pr is called
        (exp_dir / "goal.yaml").write_text(yaml.dump({
            "objective": "Test",
            "target": "logos/some_file.py",
            "eval": "eval/",
            "success_criteria": [{"metric": "test_pass_rate", "threshold": 1.0, "primary": True}],
        }))
        (exp_dir / "eval").mkdir()
        (exp_dir / "eval" / "test_pass.py").write_text("def test_ok(): assert True\n")

        fake_wt = WorktreeInfo(
            worktree_path=tmp_path / "worktree",
            branch_name="exp/test_exp",
            repo_dir=tmp_path / "repo",
        )
        push_error = subprocess.CalledProcessError(
            returncode=128, cmd=["git", "push"],
            stderr="error: failed to push some refs",
        )

        with patch("harness.run.find_experiments_dir", return_value=tmp_path / "experiments"), \
             patch("harness.run.resolve_target_repo", return_value=tmp_path / "repo"), \
             patch("harness.run.setup_worktree", return_value=fake_wt), \
             patch("harness.run.cleanup_worktree"), \
             patch("harness.run.push_and_pr", side_effect=push_error), \
             patch("sys.argv", ["harness-run", "test_exp", "--push"]):
            from harness import run as run_module
            try:
                run_module.main()
            except SystemExit:
                pass

        captured = capsys.readouterr()
        assert "[PUSH] FAILED" in captured.out
        assert "Traceback" not in captured.out

    def test_no_push_without_flag(self, tmp_path):
        """push_and_pr is never called when --push is absent."""
        import sys
        from unittest.mock import patch, MagicMock
        from harness.run import WorktreeInfo

        exp_dir = tmp_path / "experiments" / "test_exp"
        exp_dir.mkdir(parents=True)
        import yaml
        (exp_dir / "goal.yaml").write_text(yaml.dump({
            "objective": "Test",
            "eval": "eval/",
            "success_criteria": [{"metric": "test_pass_rate", "threshold": 1.0, "primary": True}],
        }))
        (exp_dir / "eval").mkdir()
        (exp_dir / "eval" / "test_pass.py").write_text("def test_ok(): assert True\n")

        with patch("harness.run.find_experiments_dir", return_value=tmp_path / "experiments"), \
             patch("harness.run.push_and_pr") as mock_push, \
             patch("sys.argv", ["harness-run", "test_exp"]):
            from harness import run as run_module
            try:
                run_module.main()
            except SystemExit:
                pass

        mock_push.assert_not_called()


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

        # Integration experiments must get a pytest test file, not the standalone harness
        exp_dir = tmp_path / "retry_eventbus"
        assert (exp_dir / "eval" / "test_integration.py").exists(), \
            "Integration experiment should scaffold test_integration.py for pytest"
        assert not (exp_dir / "eval" / "evaluate.py").exists(), \
            "Integration experiment must not scaffold standalone evaluate.py"
        test_text = (exp_dir / "eval" / "test_integration.py").read_text()
        assert "def test_" in test_text, "Scaffolded file must contain a pytest test function"

        # Integration experiments must NOT scaffold a local workspace/ dir
        assert not (exp_dir / "workspace").exists(), \
            "Integration experiment must not create workspace/ (agent uses the git worktree)"

    def test_new_standalone_no_target(self, tmp_path, monkeypatch):
        """harness-new without --target creates standard goal.yaml."""
        from harness import new as new_module
        from harness.new import create_experiment

        monkeypatch.setattr(new_module, "EXPERIMENTS_DIR", tmp_path)

        create_experiment(name="my_ml_exp", goal="Train a model")

        goal_text = (tmp_path / "my_ml_exp" / "goal.yaml").read_text()
        assert "target:" not in goal_text

        # Standalone experiments still get the old evaluate.py harness
        exp_dir = tmp_path / "my_ml_exp"
        assert (exp_dir / "eval" / "evaluate.py").exists(), \
            "Standalone experiment should scaffold evaluate.py"


class TestResolveTargetRepo:
    def test_resolves_first_path_component_as_repo(self, tmp_path):
        from harness.run import resolve_target_repo

        # Simulate workspace with a git repo
        repo = tmp_path / "logos"
        repo.mkdir()
        (repo / ".git").mkdir()  # Fake .git dir

        result = resolve_target_repo("logos/logos_events/event_bus.py", tmp_path)
        assert result == repo

    def test_raises_on_missing_repo(self, tmp_path):
        from harness.run import resolve_target_repo

        with pytest.raises(FileNotFoundError, match="Target repo not found"):
            resolve_target_repo("nonexistent/foo.py", tmp_path)

    def test_raises_on_path_traversal(self, tmp_path):
        from harness.run import resolve_target_repo

        with pytest.raises(ValueError, match="Invalid target repo name"):
            resolve_target_repo("../sibling_repo/some_file.py", tmp_path)

    def test_raises_on_absolute_path(self, tmp_path):
        from harness.run import resolve_target_repo

        with pytest.raises(ValueError, match="Invalid target repo name"):
            resolve_target_repo("/etc/passwd", tmp_path)

    @pytest.mark.parametrize("bad_name", [
        ";rm -rf /",
        "$HOME",
        "repo name",
        ".hidden",
    ])
    def test_raises_on_shell_special_chars(self, tmp_path, bad_name):
        from harness.run import resolve_target_repo

        with pytest.raises(ValueError, match="Invalid target repo name"):
            resolve_target_repo(f"{bad_name}/some_file.py", tmp_path)


class TestEvalPathBoundary:
    def test_eval_path_traversal_raises(self, tmp_path):
        """eval_path that escapes exp_dir raises ValueError."""
        from harness.run import run_eval

        exp_dir = tmp_path / "experiments" / "my_exp"
        exp_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="resolves outside experiment directory"):
            run_eval(eval_path="../../etc", exp_dir=exp_dir, worktree_path=None)

    def test_eval_path_within_exp_dir_is_allowed(self, tmp_path):
        """eval_path within the experiment dir does not raise."""
        from harness.run import run_eval

        exp_dir = tmp_path / "experiments" / "my_exp"
        eval_dir = exp_dir / "eval"
        eval_dir.mkdir(parents=True)
        (eval_dir / "test_ok.py").write_text("def test_pass(): assert True\n")

        result = run_eval(eval_path="eval/", exp_dir=exp_dir, worktree_path=None)
        assert result["pass_rate"] == 1.0


class TestGoalMissingFile:
    def test_missing_goal_raises(self, tmp_path):
        from harness.run import load_goal

        exp_dir = tmp_path / "empty_exp"
        exp_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No goal.yaml"):
            load_goal(exp_dir)
