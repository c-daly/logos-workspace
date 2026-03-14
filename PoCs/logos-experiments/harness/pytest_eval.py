"""
Generic Pytest Eval — Use any test suite as an experiment's success criteria.

Point this at a pytest test file or directory. It runs the tests, counts
pass/fail, and reports structured metrics the harness can parse.

This lets you write tickets like "make these tests pass" and use the
experiment harness to track progress.

Usage:
    # Run a specific test file
    python -m harness.pytest_eval --tests tests/test_search.py

    # Run a test directory
    python -m harness.pytest_eval --tests tests/

    # Run with a marker
    python -m harness.pytest_eval --tests tests/ --marker "not slow"

    # Set pass threshold (default: 1.0 = all tests must pass)
    python -m harness.pytest_eval --tests tests/ --threshold 0.95

Output:
    [METRIC] tests_passed=18
    [METRIC] tests_failed=2
    [METRIC] tests_total=20
    [METRIC] pass_rate=0.9
    [EVAL] FAIL ❌
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path


def run_pytest(test_path: str, marker: str = None, verbose: bool = False) -> dict:
    """Run pytest and parse results."""
    cmd = [sys.executable, "-m", "pytest", test_path, "--tb=short", "-q"]
    if marker:
        cmd.extend(["-m", marker])
    if verbose:
        cmd.append("-v")

    # Use JSON report if available, otherwise parse output
    cmd.extend(["--no-header"])

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse the summary line like "18 passed, 2 failed in 1.23s"
    output = result.stdout + result.stderr
    passed = 0
    failed = 0
    errors = 0
    skipped = 0

    for line in output.splitlines():
        line = line.strip()
        if "passed" in line or "failed" in line or "error" in line or "skipped" in line:
            import re
            p = re.search(r'(\d+) passed', line)
            f = re.search(r'(\d+) failed', line)
            e = re.search(r'(\d+) error', line)
            s = re.search(r'(\d+) skipped', line)
            if p:
                passed = int(p.group(1))
            if f:
                failed = int(f.group(1))
            if e:
                errors = int(e.group(1))
            if s:
                skipped = int(s.group(1))

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


def main():
    parser = argparse.ArgumentParser(description="Pytest Eval Adapter")
    parser.add_argument("--tests", required=True, help="Test file or directory")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Pass rate threshold (0-1, default 1.0 = all must pass)")
    parser.add_argument("--marker", default=None, help="Pytest marker expression")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    print(f"Running tests: {args.tests}")
    print(f"Threshold: {args.threshold}")
    print()

    results = run_pytest(args.tests, marker=args.marker, verbose=args.verbose)

    if args.verbose:
        print(results["output"])
        print()

    # Structured output
    for key in ["tests_passed", "tests_failed", "tests_skipped", "tests_total", "pass_rate"]:
        print(f"[METRIC] {key}={results[key]}")

    passed = results["pass_rate"] >= args.threshold
    print(f"\n[EVAL] {'PASS ✅' if passed else 'FAIL ❌'}")

    if not passed and results["tests_failed"] > 0:
        print(f"\n{results['tests_failed']} test(s) failing. Run with --verbose for details.")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
