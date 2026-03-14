"""LOGOS Experiment Harness — Structured autonomous ML experimentation."""

import os
from pathlib import Path


def find_experiments_dir() -> Path:
    """
    Resolve the experiments directory.

    Search order:
    1. HARNESS_ROOT env var → $HARNESS_ROOT/experiments/
    2. Walk up from cwd looking for experiments/ or CLAUDE.md
    3. Fall back to cwd/experiments/
    """
    root = os.environ.get("HARNESS_ROOT")
    if root:
        return Path(root) / "experiments"
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if (p / "experiments").is_dir():
            return p / "experiments"
        if (p / "CLAUDE.md").exists():
            return p / "experiments"
    return cwd / "experiments"


def find_project_root() -> Path:
    """Find the project root (directory containing CLAUDE.md or experiments/)."""
    root = os.environ.get("HARNESS_ROOT")
    if root:
        return Path(root)
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if (p / "CLAUDE.md").exists() or (p / "experiments").is_dir():
            return p
    return cwd

