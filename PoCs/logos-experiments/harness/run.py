"""
Experiment runner — orchestrates the experiment lifecycle.

Usage:
    harness-run <experiment> [--push]
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
