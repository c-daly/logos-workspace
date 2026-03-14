# Experiment Execution Modes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `harness-run` command that creates git worktrees for integration experiments and optionally pushes results as PRs.

**Architecture:** `harness/run.py` orchestrates the experiment lifecycle — reads goal.yaml, creates a worktree for integration experiments (those with `target:`), runs eval with the worktree on PYTHONPATH, journals results, and optionally pushes + opens a PR. `harness/new.py` gains a `--target` flag to scaffold integration experiments with real-module imports.

**Tech Stack:** Python 3.12, subprocess (git worktree), PyYAML, pytest, gh CLI (for PRs)

---

### Task 1: Test and implement goal.yaml target parsing

**Files:**
- Create: `PoCs/logos-experiments/tests/test_run.py`
- Create: `PoCs/logos-experiments/harness/run.py`

**Step 1: Write the failing test**

```python
# tests/test_run.py
"""Tests for harness-run experiment runner."""

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
```

**Step 2: Run test to verify it fails**

Run: `cd PoCs/logos-experiments && python -m pytest tests/test_run.py::TestGoalParsing -v`
Expected: FAIL — `harness.run` does not exist

**Step 3: Write minimal implementation**

```python
# harness/run.py
"""
Experiment runner — orchestrates the experiment lifecycle.

Usage:
    harness-run <experiment> [--push]

For integration experiments (goal.yaml has target:), creates a git
worktree on the target repo so the agent works against real code.
For standalone experiments, uses workspace/ as-is.
"""

import sys
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from harness import find_experiments_dir


@dataclass
class Goal:
    """Parsed goal.yaml with convenience properties."""
    objective: str
    eval: str
    success_criteria: list
    target: Optional[str] = None
    context: Optional[str] = None
    environment: Optional[dict] = None
    _raw: dict = field(default_factory=dict, repr=False)

    @property
    def is_integration(self) -> bool:
        return self.target is not None

    def get(self, key, default=None):
        return self._raw.get(key, default)

    def __getitem__(self, key):
        return self._raw[key]


def load_goal(exp_dir: Path) -> Goal:
    """Load and parse goal.yaml from an experiment directory."""
    goal_path = exp_dir / "goal.yaml"
    if not goal_path.exists():
        raise FileNotFoundError(f"No goal.yaml in {exp_dir}")

    raw = yaml.safe_load(goal_path.read_text())

    return Goal(
        objective=raw.get("objective", ""),
        eval=raw.get("eval", "eval/"),
        success_criteria=raw.get("success_criteria", []),
        target=raw.get("target"),
        context=raw.get("context"),
        environment=raw.get("environment"),
        _raw=raw,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd PoCs/logos-experiments && python -m pytest tests/test_run.py::TestGoalParsing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add PoCs/logos-experiments/harness/run.py PoCs/logos-experiments/tests/test_run.py
git commit -m "feat: add goal.yaml parsing with target field for integration experiments"
```

---

### Task 2: Test and implement worktree creation

**Files:**
- Modify: `PoCs/logos-experiments/tests/test_run.py`
- Modify: `PoCs/logos-experiments/harness/run.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_run.py

import subprocess


class TestWorktreeSetup:
    def test_creates_worktree_for_integration(self, tmp_path):
        """Integration experiment gets a worktree on the target repo."""
        from harness.run import setup_worktree

        # Create a bare git repo to act as target
        repo_dir = tmp_path / "logos"
        repo_dir.mkdir()
        subprocess.run(["git", "init", str(repo_dir)], check=True,
                       capture_output=True)
        (repo_dir / "event_bus.py").write_text("class EventBus: pass")
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "init"],
                       check=True, capture_output=True)

        result = setup_worktree(
            repo_dir=repo_dir,
            experiment_name="test_retry",
        )

        assert result.worktree_path.exists()
        assert result.branch_name == "exp/test_retry"
        # The worktree should have the file from the repo
        assert (result.worktree_path / "event_bus.py").exists()

    def test_cleanup_removes_worktree(self, tmp_path):
        """cleanup_worktree removes the worktree but keeps the branch."""
        from harness.run import setup_worktree, cleanup_worktree

        repo_dir = tmp_path / "logos"
        repo_dir.mkdir()
        subprocess.run(["git", "init", str(repo_dir)], check=True,
                       capture_output=True)
        (repo_dir / "dummy.py").write_text("x = 1")
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(repo_dir), "commit", "-m", "init"],
                       check=True, capture_output=True)

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
```

**Step 2: Run test to verify it fails**

Run: `cd PoCs/logos-experiments && python -m pytest tests/test_run.py::TestWorktreeSetup -v`
Expected: FAIL — `setup_worktree` not defined

**Step 3: Write minimal implementation**

```python
# Add to harness/run.py

import subprocess
import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class WorktreeInfo(NamedTuple):
    worktree_path: Path
    branch_name: str
    repo_dir: Path


def setup_worktree(
    repo_dir: Optional[Path],
    experiment_name: str,
) -> Optional[WorktreeInfo]:
    """Create a git worktree for an integration experiment.

    Returns None for standalone experiments (repo_dir is None).
    """
    if repo_dir is None:
        return None

    branch_name = f"exp/{experiment_name}"
    worktree_path = repo_dir.parent / f".worktrees/{experiment_name}"

    # Create the worktree with a new branch
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "worktree", "add",
         "-b", branch_name, str(worktree_path)],
        check=True, capture_output=True, text=True,
    )

    logger.info("Created worktree at %s (branch: %s)", worktree_path, branch_name)
    return WorktreeInfo(
        worktree_path=worktree_path,
        branch_name=branch_name,
        repo_dir=repo_dir,
    )


def cleanup_worktree(repo_dir: Path, worktree_path: Path) -> None:
    """Remove a worktree but keep the branch for inspection."""
    subprocess.run(
        ["git", "-C", str(repo_dir), "worktree", "remove",
         str(worktree_path), "--force"],
        check=True, capture_output=True, text=True,
    )
    logger.info("Removed worktree at %s (branch kept)", worktree_path)
```

**Step 4: Run test to verify it passes**

Run: `cd PoCs/logos-experiments && python -m pytest tests/test_run.py::TestWorktreeSetup -v`
Expected: PASS

**Step 5: Commit**

```bash
git add PoCs/logos-experiments/harness/run.py PoCs/logos-experiments/tests/test_run.py
git commit -m "feat: add worktree creation for integration experiments"
```

---

### Task 3: Test and implement eval execution

**Files:**
- Modify: `PoCs/logos-experiments/tests/test_run.py`
- Modify: `PoCs/logos-experiments/harness/run.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_run.py

class TestEvalExecution:
    def test_runs_pytest_eval(self, tmp_path):
        """Eval runs pytest on the eval directory and captures results."""
        from harness.run import run_eval

        # Create a trivial passing test
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "test_trivial.py").write_text(
            "def test_pass(): assert True\n"
        )

        result = run_eval(eval_path="eval/", exp_dir=tmp_path, worktree_path=None)
        assert result["pass_rate"] == 1.0
        assert result["tests_passed"] == 1

    def test_eval_with_worktree_on_pythonpath(self, tmp_path):
        """Integration eval has the worktree importable."""
        from harness.run import run_eval

        # Create a "worktree" with a module
        worktree = tmp_path / "worktree"
        worktree.mkdir()
        (worktree / "my_module.py").write_text("VALUE = 42\n")

        # Create eval that imports from the worktree module
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "test_import.py").write_text(
            "def test_import():\n"
            "    from my_module import VALUE\n"
            "    assert VALUE == 42\n"
        )

        result = run_eval(
            eval_path="eval/",
            exp_dir=tmp_path,
            worktree_path=worktree,
        )
        assert result["pass_rate"] == 1.0
```

**Step 2: Run test to verify it fails**

Run: `cd PoCs/logos-experiments && python -m pytest tests/test_run.py::TestEvalExecution -v`
Expected: FAIL — `run_eval` not defined

**Step 3: Write minimal implementation**

```python
# Add to harness/run.py

import os
import re


def run_eval(
    eval_path: str,
    exp_dir: Path,
    worktree_path: Optional[Path] = None,
) -> dict:
    """Run the eval and return parsed results.

    For integration experiments, adds worktree_path to PYTHONPATH so
    eval can import real modules.
    """
    full_eval_path = exp_dir / eval_path

    env = os.environ.copy()
    if worktree_path:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{worktree_path}{os.pathsep}{existing}" if existing else str(worktree_path)

    cmd = [sys.executable, "-m", "pytest", str(full_eval_path),
           "--tb=short", "-q", "--no-header"]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Parse pytest output (same logic as pytest_eval.py)
    output = result.stdout + result.stderr
    passed = failed = errors = skipped = 0

    for line in output.splitlines():
        line = line.strip()
        if "passed" in line or "failed" in line or "error" in line or "skipped" in line:
            p = re.search(r'(\d+) passed', line)
            f = re.search(r'(\d+) failed', line)
            e = re.search(r'(\d+) error', line)
            s = re.search(r'(\d+) skipped', line)
            if p: passed = int(p.group(1))
            if f: failed = int(f.group(1))
            if e: errors = int(e.group(1))
            if s: skipped = int(s.group(1))

    total = passed + failed + errors
    pass_rate = passed / total if total > 0 else 0.0

    return {
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_errors": errors,
        "tests_skipped": skipped,
        "tests_total": total + skipped,
        "pass_rate": round(pass_rate, 4),
        "exit_code": result.returncode,
        "output": output,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd PoCs/logos-experiments && python -m pytest tests/test_run.py::TestEvalExecution -v`
Expected: PASS

**Step 5: Commit**

```bash
git add PoCs/logos-experiments/harness/run.py PoCs/logos-experiments/tests/test_run.py
git commit -m "feat: add eval execution with worktree PYTHONPATH support"
```

---

### Task 4: Test and implement the push flag

**Files:**
- Modify: `PoCs/logos-experiments/tests/test_run.py`
- Modify: `PoCs/logos-experiments/harness/run.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_run.py
from unittest.mock import patch, MagicMock


class TestPushFlag:
    def test_push_creates_pr(self, tmp_path):
        """--push pushes the branch and opens a PR via gh."""
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
```

**Step 2: Run test to verify it fails**

Run: `cd PoCs/logos-experiments && python -m pytest tests/test_run.py::TestPushFlag -v`
Expected: FAIL — `push_and_pr` not defined

**Step 3: Write minimal implementation**

```python
# Add to harness/run.py

def push_and_pr(
    wt_info: WorktreeInfo,
    experiment_name: str,
    eval_results: dict,
) -> str:
    """Push the experiment branch and open a PR. Returns the PR URL."""
    # Push the branch
    subprocess.run(
        ["git", "-C", str(wt_info.worktree_path), "push", "-u", "origin",
         wt_info.branch_name],
        check=True, capture_output=True, text=True,
    )

    # Build PR body
    pass_rate = eval_results.get("pass_rate", 0)
    status = "PASS" if pass_rate >= 1.0 else "FAIL"
    body = (
        f"## Experiment: {experiment_name}\n\n"
        f"**Eval result:** {status} (pass_rate={pass_rate})\n"
        f"**Tests:** {eval_results.get('tests_passed', 0)} passed, "
        f"{eval_results.get('tests_failed', 0)} failed, "
        f"{eval_results.get('tests_skipped', 0)} skipped\n\n"
        f"Generated by `harness-run {experiment_name} --push`."
    )

    result = subprocess.run(
        ["gh", "pr", "create",
         "--title", f"exp: {experiment_name}",
         "--body", body,
         "--head", wt_info.branch_name],
        check=True, capture_output=True, text=True,
        cwd=str(wt_info.worktree_path),
    )

    url = result.stdout.strip()
    logger.info("Created PR: %s", url)
    return url
```

**Step 4: Run test to verify it passes**

Run: `cd PoCs/logos-experiments && python -m pytest tests/test_run.py::TestPushFlag -v`
Expected: PASS

**Step 5: Commit**

```bash
git add PoCs/logos-experiments/harness/run.py PoCs/logos-experiments/tests/test_run.py
git commit -m "feat: add --push flag for pushing branch and opening PR"
```

---

### Task 5: Wire up the main() CLI entry point

**Files:**
- Modify: `PoCs/logos-experiments/harness/run.py`
- Modify: `PoCs/logos-experiments/pyproject.toml`
- Modify: `PoCs/logos-experiments/Justfile`

**Step 1: Write main() and resolve_target_repo()**

```python
# Add to harness/run.py

def resolve_target_repo(target: str, workspace_root: Path) -> Path:
    """Resolve a target path to the repo it lives in.

    target is like "logos/logos_events/event_bus.py".
    The first path component is the repo directory name.
    """
    parts = Path(target).parts
    if not parts:
        raise ValueError(f"Invalid target: {target}")
    repo_name = parts[0]
    repo_dir = workspace_root / repo_name
    if not (repo_dir / ".git").exists():
        raise FileNotFoundError(
            f"Target repo not found: {repo_dir} (from target '{target}'). "
            f"Expected a git repo at {repo_dir}."
        )
    return repo_dir


def main():
    """CLI entry point: harness-run <experiment> [--push]"""
    import argparse

    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--push", action="store_true",
                        help="Push branch and open PR when done")
    args = parser.parse_args()

    experiments_dir = find_experiments_dir()
    exp_dir = experiments_dir / args.experiment
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}")
        sys.exit(1)

    goal = load_goal(exp_dir)
    print(f"Experiment: {args.experiment}")
    print(f"Type: {'integration' if goal.is_integration else 'standalone'}")
    print(f"Objective: {goal.objective.strip()[:80]}")
    print()

    # Set up worktree for integration experiments
    wt_info = None
    if goal.is_integration:
        from harness import find_project_root
        workspace_root = find_project_root().parent  # logos-experiments -> LOGOS
        repo_dir = resolve_target_repo(goal.target, workspace_root)
        wt_info = setup_worktree(repo_dir=repo_dir, experiment_name=args.experiment)
        print(f"Worktree: {wt_info.worktree_path}")
        print(f"Branch: {wt_info.branch_name}")
        print(f"Target: {goal.target}")
    else:
        print(f"Workspace: {exp_dir / 'workspace'}")

    print()
    print("=" * 60)
    print("Agent: implement the objective, then signal ready for eval.")
    print("=" * 60)
    print()

    # TODO: Team dispatch goes here — for now, just run eval
    # The agent works in the worktree, then we run eval.

    print("Running eval...")
    worktree_path = wt_info.worktree_path if wt_info else None
    results = run_eval(goal.eval, exp_dir, worktree_path)

    for key in ["tests_passed", "tests_failed", "tests_skipped", "tests_total", "pass_rate"]:
        print(f"[METRIC] {key}={results[key]}")

    passed = results["pass_rate"] >= 1.0
    print(f"\n[EVAL] {'PASS' if passed else 'FAIL'}")

    if args.push and wt_info:
        if passed:
            url = push_and_pr(wt_info, args.experiment, results)
            print(f"\nPR created: {url}")
        else:
            print("\nEval failed — not pushing. Fix and re-run with --push.")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

**Step 2: Add entry point to pyproject.toml**

Add this line to the `[project.scripts]` section:

```
harness-run = "harness.run:main"
```

**Step 3: Add run command to Justfile**

```
run experiment *flags="":
    python -m harness.run {{experiment}} {{flags}}
```

**Step 4: Verify CLI works**

Run: `cd PoCs/logos-experiments && python -m harness.run --help`
Expected: Shows usage with `experiment` and `--push` args

**Step 5: Commit**

```bash
git add PoCs/logos-experiments/harness/run.py PoCs/logos-experiments/pyproject.toml PoCs/logos-experiments/Justfile
git commit -m "feat: wire up harness-run CLI with main(), pyproject entry, Justfile"
```

---

### Task 6: Update new.py scaffold for integration experiments

**Files:**
- Modify: `PoCs/logos-experiments/tests/test_run.py`
- Modify: `PoCs/logos-experiments/harness/new.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_run.py

class TestScaffoldIntegration:
    def test_new_integration_experiment_has_target(self, tmp_path, monkeypatch):
        """harness-new --target creates goal.yaml with target field."""
        from harness.new import create_experiment

        monkeypatch.setattr("harness.new.EXPERIMENTS_DIR", tmp_path)

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
        from harness.new import create_experiment

        monkeypatch.setattr("harness.new.EXPERIMENTS_DIR", tmp_path)

        create_experiment(name="my_ml_exp", goal="Train a model")

        goal_text = (tmp_path / "my_ml_exp" / "goal.yaml").read_text()
        assert "target:" not in goal_text
```

**Step 2: Run test to verify it fails**

Run: `cd PoCs/logos-experiments && python -m pytest tests/test_run.py::TestScaffoldIntegration -v`
Expected: FAIL — `create_experiment` doesn't accept `target` parameter

**Step 3: Update new.py**

Add `INTEGRATION_GOAL_TEMPLATE` after `GOAL_TEMPLATE`:

```python
INTEGRATION_GOAL_TEMPLATE = """# {name}

objective: |
  {goal}

target: {target}

success_criteria:
  - metric: test_pass_rate
    threshold: 1.0
    primary: true

eval: eval/

# context: |
#   Why this matters, background info.

# notes: |
#   Anything else the agent should know.
"""
```

Update `create_experiment` signature to accept optional `target`:

```python
def create_experiment(name: str, goal: str = "<describe the objective>", target: str = None):
```

Change the goal.yaml write to choose template based on `target`:

```python
    if target:
        (exp_dir / "goal.yaml").write_text(
            INTEGRATION_GOAL_TEMPLATE.format(name=name, goal=goal, target=target)
        )
    else:
        (exp_dir / "goal.yaml").write_text(
            GOAL_TEMPLATE.format(name=name, goal=goal)
        )
```

Update `main()` argparse to add `--target`:

```python
    parser.add_argument("--target", "-t", default=None,
                        help="Target file for integration experiments (e.g. logos/logos_events/event_bus.py)")
```

And pass it through:

```python
    create_experiment(args.name, args.goal, target=args.target)
```

**Step 4: Run test to verify it passes**

Run: `cd PoCs/logos-experiments && python -m pytest tests/test_run.py::TestScaffoldIntegration -v`
Expected: PASS

**Step 5: Commit**

```bash
git add PoCs/logos-experiments/harness/new.py PoCs/logos-experiments/tests/test_run.py
git commit -m "feat: add --target flag to harness-new for integration experiments"
```

---

### Task 7: Update CLAUDE.md protocol

**Files:**
- Modify: `PoCs/logos-experiments/CLAUDE.md`

**Step 1: Add integration experiment section**

After the "### 3. Work" section (line 25), update it to distinguish standalone vs integration:

Replace:
```
### 3. Work

Use `experiments/<name>/workspace/` for your code, models, artifacts. The ticket tells you *what* — you decide *how*.
```

With:
```
### 3. Work

**Standalone experiments** (no `target:` in goal.yaml): Use `experiments/<name>/workspace/` for your code, models, artifacts.

**Integration experiments** (`target:` in goal.yaml): The harness creates a git worktree on the target repo. You work in the worktree — modifying the real codebase, not a sandbox copy. The eval imports and tests the real module.

The ticket tells you *what* — you decide *how*.
```

Add `harness-run` to the CLI commands table:

```
| `harness-run <experiment>` | Run experiment (creates worktree for integration) |
| `harness-run <experiment> --push` | Run + push branch + open PR |
```

**Step 2: Commit**

```bash
git add PoCs/logos-experiments/CLAUDE.md
git commit -m "docs: update CLAUDE.md protocol for integration experiments"
```

---

### Task 8: Integration test — run against example_software_task

**Files:**
- Modify: `PoCs/logos-experiments/tests/test_run.py`

**Step 1: Write the test**

```python
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
```

**Step 2: Run full test suite**

Run: `cd PoCs/logos-experiments && python -m pytest tests/test_run.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add PoCs/logos-experiments/tests/test_run.py
git commit -m "test: add resolve_target_repo tests and verify full test_run suite"
```
