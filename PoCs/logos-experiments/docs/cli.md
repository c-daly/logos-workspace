# CLI Reference

All commands are available after installing the package (see README for install instructions).

## harness-init

Scaffold a new harness project in any directory.

```bash
harness-init                    # Current directory
harness-init /path/to/project   # Specific directory
```

Creates `CLAUDE.md` and `experiments/` directory. Safe to run in an existing project — won't overwrite an existing CLAUDE.md.

## harness-new

Create a new experiment.

```bash
harness-new my_experiment
harness-new my_experiment --goal "Classify images with >95% accuracy"
```

Creates the experiment directory structure under `experiments/`:
```
experiments/my_experiment/
├── goal.yaml       # Ticket (edit this)
├── eval/
│   └── evaluate.py # Eval skeleton (edit this)
├── workspace/      # Agent's working space
├── journal/        # Experiment memory
├── checkpoints/    # Saved intermediate states
└── status.yaml     # Machine-readable state
```

## harness-journal

Manage experiment journals.

```bash
harness-journal my_experiment summary    # Overview of all attempts
harness-journal my_experiment failures   # Extract failure modes
harness-journal my_experiment add        # Add entry interactively
```

**summary** — Shows each attempt's title, results, and diagnosis. Use this to quickly see where an experiment stands.

**failures** — Extracts just the entries with failure-related diagnoses (NaN, crash, diverge, etc). Useful for building a constraints.yaml.

**add** — Interactive prompts for title, hypothesis, changes, results, diagnosis, next direction, and training time.

## harness-metrics

View metrics across experiment attempts.

```bash
harness-metrics my_experiment
```

Reads from `experiments/my_experiment/logs/attempt_NNN/result.json` files. Shows a table of attempt status, duration, and final metric values.

Note: This requires results to be written in the expected directory structure. The training monitor writes these automatically when used via `harness-monitor`.

## harness-monitor

Run a command with NaN detection, metric capture, and timeout enforcement.

```bash
harness-monitor -- python train.py
harness-monitor --timeout 2 -- python train.py --lr 1e-4
harness-monitor --timeout 4 --nan-patience 5 -- ./run_experiment.sh
```

**Options:**
- `--timeout HOURS` — Kill the process after this many hours (default: 4)
- `--nan-patience N` — Kill after N consecutive NaN/Inf lines (default: 10)
- `--log-dir DIR` — Save stdout/stderr to files in this directory

**What it monitors:**
- NaN/Inf in output lines containing metric keywords (loss, step, epoch, batch)
- `[METRIC] key=value` structured metric lines
- Heuristic metric capture (`loss=0.5`, `lr=1e-4`)

**Output:** Prints a JSON summary on completion:
```json
{
  "exit_code": 0,
  "duration_seconds": 123.4,
  "killed_reason": null,
  "nan_detected": false,
  "metrics": {"loss": [1.0, 0.5, 0.1]},
  "succeeded": true
}
```

## harness-eval

Run a pytest suite as an experiment eval.

```bash
harness-eval --tests eval/test_search.py
harness-eval --tests tests/ --threshold 0.95
harness-eval --tests tests/ --marker "not slow" --verbose
harness-eval --tests tests/ --output-json results.json
```

**Options:**
- `--tests PATH` — Test file or directory (required)
- `--threshold FLOAT` — Pass rate required (0-1, default: 1.0 = all must pass)
- `--marker EXPR` — Pytest marker expression for filtering
- `--verbose` — Show full pytest output
- `--output-json PATH` — Save results as JSON

**Output:**
```
[METRIC] tests_passed=18
[METRIC] tests_failed=2
[METRIC] tests_total=20
[METRIC] pass_rate=0.9
[EVAL] FAIL ❌
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `HARNESS_ROOT` | Override project root detection. All commands resolve `experiments/` relative to this. |

## Path Resolution

CLI commands find the `experiments/` directory by:
1. Checking `HARNESS_ROOT` environment variable
2. Walking up from the current directory looking for `experiments/` or `CLAUDE.md`
3. Falling back to `./experiments/`

This means you can run commands from any subdirectory of your project.
