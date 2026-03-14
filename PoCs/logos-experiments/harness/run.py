"""
Experiment runner — orchestrates the experiment lifecycle.

Usage:
    harness-run <experiment> [--push]
"""

import logging
import os
import re
import subprocess
import sys
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

from harness import find_experiments_dir

logger = logging.getLogger(__name__)


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


class WorktreeInfo(NamedTuple):
    worktree_path: Path
    branch_name: str
    repo_dir: Path


def setup_worktree(
    repo_dir: Optional[Path],
    experiment_name: str,
) -> Optional[WorktreeInfo]:
    """Create a git worktree for an integration experiment."""
    if repo_dir is None:
        return None

    branch_name = f"exp/{experiment_name}"
    worktree_path = repo_dir.parent / f".worktrees/{experiment_name}"

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
