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

from harness import find_experiments_dir, find_project_root

logger = logging.getLogger(__name__)

_EVAL_TIMEOUT = int(os.environ.get("HARNESS_EVAL_TIMEOUT", "300"))


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
    """Create a git worktree for an integration experiment.

    If the branch already exists, checks it out into the worktree
    rather than creating a new branch.
    """
    if repo_dir is None:
        return None

    branch_name = f"exp/{experiment_name}"
    worktree_path = repo_dir.parent / f".worktrees/{experiment_name}"

    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["git", "-C", str(repo_dir), "worktree", "add",
             "-b", branch_name, str(worktree_path)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        if "already exists" not in stderr or branch_name not in stderr:
            raise
        # Branch exists from a previous run — reuse it
        subprocess.run(
            ["git", "-C", str(repo_dir), "worktree", "add",
             str(worktree_path), branch_name],
            check=True, capture_output=True, text=True,
        )
        logger.info("Reusing existing branch %s", branch_name)

    logger.info("Created worktree at %s (branch: %s)", worktree_path, branch_name)
    return WorktreeInfo(
        worktree_path=worktree_path,
        branch_name=branch_name,
        repo_dir=repo_dir,
    )


def cleanup_worktree(repo_dir: Path, worktree_path: Path) -> None:
    """Remove a worktree but keep the branch for inspection.

    Does NOT force-remove. If the worktree has uncommitted changes (e.g. agent
    work in progress), logs a warning and leaves it intact rather than silently
    discarding work. Caller can force-remove manually.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "worktree", "remove", str(worktree_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.warning(
            "Worktree %s not removed (may have uncommitted changes). "
            "Remove manually: git worktree remove --force %s",
            worktree_path, worktree_path,
        )
        return
    logger.info("Removed worktree at %s (branch kept)", worktree_path)


def run_eval(
    eval_path: str,
    exp_dir: Path,
    worktree_path: Optional[Path] = None,
    timeout: int = _EVAL_TIMEOUT,
    environment: Optional[dict] = None,
) -> dict:
    """Run the eval and return parsed results.

    For integration experiments, adds worktree_path to PYTHONPATH so
    eval can import real modules. Applies goal.environment overrides on top
    of the inherited environment. Raises subprocess.TimeoutExpired if the
    eval hangs beyond `timeout` seconds.
    """
    full_eval_path = exp_dir / eval_path
    try:
        full_eval_path.resolve().relative_to(exp_dir.resolve())
    except ValueError:
        raise ValueError(
            f"eval_path '{eval_path}' resolves outside experiment directory"
        )

    env = os.environ.copy()
    if environment:
        env.update({str(k): str(v) for k, v in environment.items()})
    if worktree_path:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{worktree_path}{os.pathsep}{existing}" if existing else str(worktree_path)

    cmd = [sys.executable, "-m", "pytest", str(full_eval_path),
           "--tb=short", "-q", "--no-header"]

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=timeout,
    )

    output = result.stdout + result.stderr
    passed = failed = errors = skipped = 0

    summary_line = None
    for line in reversed(output.splitlines()):
        if re.search(r'\d+ (passed|failed|error|skipped)', line.strip()):
            summary_line = line.strip()
            break
    if summary_line:
        p = re.search(r'(\d+) passed', summary_line)
        f = re.search(r'(\d+) failed', summary_line)
        e = re.search(r'(\d+) error', summary_line)
        s = re.search(r'(\d+) skipped', summary_line)
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

    try:
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
    except subprocess.CalledProcessError as e:
        if "already exists" not in (e.stderr or "") and "already exists" not in (e.stdout or ""):
            raise
        # PR already exists for this branch — fetch the URL
        result = subprocess.run(
            ["gh", "pr", "view", "--json", "url", "-q", ".url"],
            check=True, capture_output=True, text=True,
            cwd=str(wt_info.worktree_path),
        )
        url = result.stdout.strip()
        logger.info("PR already exists: %s", url)
    return url


def resolve_target_repo(target: str, workspace_root: Path) -> Path:
    """Resolve a target path to the repo it lives in.

    target is like "logos/logos_events/event_bus.py".
    The first path component is the repo directory name.
    """
    parts = Path(target).parts
    if not parts:
        raise ValueError(f"Invalid target: {target}")
    repo_name = parts[0]
    if repo_name in ("..", ".") or Path(repo_name).is_absolute():
        raise ValueError(f"Invalid target repo name '{repo_name}' in target '{target}'")
    repo_dir = workspace_root / repo_name
    try:
        repo_dir.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        raise ValueError(f"Target '{target}' resolves outside workspace boundary")
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
        _workspace_env = os.environ.get("WORKSPACE_ROOT")
        workspace_root = Path(_workspace_env) if _workspace_env else find_project_root().parent
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

    passed = False
    try:
        print("Running eval...")
        worktree_path = wt_info.worktree_path if wt_info else None
        results = run_eval(goal.eval, exp_dir, worktree_path, environment=goal.environment)

        for key in ["tests_passed", "tests_failed", "tests_skipped", "tests_total", "pass_rate"]:
            print(f"[METRIC] {key}={results[key]}")

        passed = results["pass_rate"] >= 1.0
        print(f"\n[EVAL] {'PASS' if passed else 'FAIL'}")

        if args.push and wt_info:
            if passed:
                try:
                    url = push_and_pr(wt_info, args.experiment, results)
                    print(f"\nPR created: {url}")
                except subprocess.CalledProcessError as e:
                    stderr = (e.stderr or "").strip()
                    print(f"\n[PUSH] FAILED — {stderr or str(e)}")
            else:
                print("\nEval failed — not pushing. Fix and re-run with --push.")
    except subprocess.TimeoutExpired:
        print(f"\n[EVAL] TIMEOUT — eval exceeded {_EVAL_TIMEOUT}s limit")
    finally:
        if wt_info:
            cleanup_worktree(wt_info.repo_dir, wt_info.worktree_path)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
