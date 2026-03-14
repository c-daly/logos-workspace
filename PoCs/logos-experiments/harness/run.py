"""
Experiment runner — orchestrates the experiment lifecycle.

Usage:
    harness-run <experiment> [--push]
"""

import logging
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
