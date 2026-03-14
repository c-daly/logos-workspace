"""
Metrics Manager — View and compare metrics across experiment attempts.

Usage:
    python -m harness.metrics vjepa_clip_alignment
    python -m harness.metrics vjepa_clip_alignment --compare 1 3 5
"""

import sys
import json
from pathlib import Path

from harness import find_experiments_dir

EXPERIMENTS_DIR = find_experiments_dir()


def load_attempt_results(experiment_dir: Path) -> dict[int, dict]:
    """Load all attempt results."""
    logs_dir = experiment_dir / "logs"
    if not logs_dir.exists():
        return {}

    results = {}
    for attempt_dir in sorted(logs_dir.iterdir()):
        if not attempt_dir.is_dir():
            continue
        result_file = attempt_dir / "result.json"
        if result_file.exists():
            try:
                num = int(attempt_dir.name.split("_")[-1])
                with open(result_file) as f:
                    results[num] = json.load(f)
            except (ValueError, json.JSONDecodeError):
                pass
    return results


def show_overview(experiment_name: str):
    """Show overview of all attempts."""
    experiment_dir = EXPERIMENTS_DIR / experiment_name
    results = load_attempt_results(experiment_dir)

    if not results:
        print(f"No results found for {experiment_name}")
        return

    print(f"\n{'='*70}")
    print(f"METRICS OVERVIEW: {experiment_name}")
    print(f"{'='*70}")
    print(f"{'Attempt':>8} {'Status':>10} {'Duration':>12} {'Killed':>20} {'NaN':>5}")
    print(f"{'-'*8:>8} {'-'*10:>10} {'-'*12:>12} {'-'*20:>20} {'-'*5:>5}")

    for num, result in sorted(results.items()):
        status = "✅" if result.get("succeeded") else "❌"
        duration = result.get("duration_human", "?")
        killed = result.get("killed_reason", "-") or "-"
        nan = "yes" if result.get("nan_detected") else "no"
        print(f"#{num:>6d} {status:>10} {duration:>12} {killed:>20} {nan:>5}")

    # Show metrics across attempts
    all_metric_keys = set()
    for result in results.values():
        metrics = result.get("metrics", {})
        all_metric_keys.update(metrics.keys())

    if all_metric_keys:
        print(f"\n{'FINAL METRIC VALUES':}")
        print(f"{'Attempt':>8}", end="")
        for key in sorted(all_metric_keys):
            print(f" {key:>15}", end="")
        print()

        for num, result in sorted(results.items()):
            metrics = result.get("metrics", {})
            print(f"#{num:>6d}", end="")
            for key in sorted(all_metric_keys):
                values = metrics.get(key, [])
                if values and isinstance(values, list):
                    print(f" {values[-1]:>15.4f}", end="")
                else:
                    print(f" {'—':>15}", end="")
            print()

    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m harness.metrics <experiment_name> [--compare N N N]")
        sys.exit(1)

    experiment_name = sys.argv[1]
    show_overview(experiment_name)


if __name__ == "__main__":
    main()
