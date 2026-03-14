# Experiment Execution Modes

**Date:** 2026-03-14
**Status:** Draft
**Ticket:** TBD

## Problem

The experiment harness lets agents build throwaway code in a sandbox
`workspace/` directory that passes eval tests without touching the real
codebase. The example_software_task demonstrated this failure mode: the
agent created a toy EventBus implementation that passed all tests while
the real `logos_events.EventBus` was never modified.

Software experiments must produce real changes to real code.

## Design

### Experiment Types

Two types, same harness:

- **Standalone** — agent creates new code (ML PoCs). Workspace is the
  working directory. Eval tests what the agent built. No target repo.
  No change to current behavior.
- **Integration** — agent modifies existing code in a target repo.
  Eval imports and tests the real module. `goal.yaml` has a `target:`
  field pointing at the file to change.

The distinction is whether `target:` is present in goal.yaml.

### Execution Modes

```
harness-run <experiment> [--push]
```

- **Default (dry-run):** Agent works on a feature branch in a git
  worktree. Eval runs. Results captured. Branch and worktree persist
  locally for inspection. Nothing pushed.
- **`--push`:** Same workflow, but at the end pushes the branch and
  opens a PR referencing the experiment and results.

### Integration Experiment Flow

1. Read goal.yaml — find `target:`, resolve which repo it points to.
2. Create worktree — `git worktree add` on the target repo, branch
   named `exp/<experiment-name>`.
3. Install deps — experiment's own pyproject.toml defines what it
   needs (deferred: full container support).
4. Spawn team — harness agent dispatches worker with goal + worktree
   path.
5. Worker implements — works in the worktree against real code.
   Signals "ready for eval" when done.
6. Run eval — eval imports from the worktree's module paths.
7. Capture results — metrics, journal entry, status.yaml updated.
8. Iterate or done — harness agent decides whether to re-dispatch
   worker or declare success/failure.
9. If `--push` — push branch, open PR.

For standalone experiments, steps 2-4 are skipped. Agent works in
workspace/ as today.

### Team Structure

```
harness-run <experiment> [--push]
    │
    ├── Harness Agent (PM)
    │   - Reads goal.yaml, journal, constraints
    │   - Creates worktree + branch on target repo
    │   - Dispatches worker with goal + worktree path
    │   - Runs eval when worker signals ready
    │   - Writes journal entry with results
    │   - Updates status.yaml
    │   - If --push: pushes branch, opens PR
    │   - Decides: iterate or done
    │
    └── Worker Agent (Implementer)
        - Receives goal + worktree path
        - Works in the worktree against real code
        - Signals "ready for eval" when done
        - Receives eval results from harness agent
        - Can iterate (fix and signal again)
        - For complex work, can spawn sub-agents
```

The iteration loop stays with the harness agent. The worker implements
and signals.

### goal.yaml Schema Changes

Two new optional fields:

```yaml
# Existing fields (unchanged)
objective: |
  Add retry with exponential backoff to EventBus.publish()...
success_criteria:
  - metric: test_pass_rate
    threshold: 1.0
    primary: true
eval: eval/
context: |
  ...

# New fields
target: logos/logos_events/event_bus.py    # triggers worktree mode
environment:                               # per-experiment deps
  pyproject: ./pyproject.toml
```

- `target:` absent → standalone (workspace/ mode)
- `target:` present → integration (worktree mode)
- `environment:` is the hook for per-experiment dependencies.

### Eval for Integration Experiments

Eval tests import the real module, not a workspace copy:

```python
from logos_events.event_bus import EventBus
from logos_config.settings import RedisConfig
```

The `new.py` scaffold generates this import pattern when creating an
integration experiment. The eval runs with the worktree on `sys.path`
(or installed into a venv), so it always tests the agent's actual
changes to the real code.

## Codebase Changes

### New

- `harness/run.py` — `harness-run` entry point. Reads goal.yaml,
  decides standalone vs integration, creates worktree if needed,
  spawns team, runs eval loop, journals results.
- `pyproject.toml` entry: `harness-run = "harness.run:main"`

### Modified

- `harness/new.py` — ask standalone vs integration when scaffolding.
  If integration, prompt for `target:` and generate eval skeleton
  with real module imports.
- `CLAUDE.md` — update protocol: integration experiments work in
  worktrees against real code, not workspace/.
- Docs — document `target:` and `environment:` fields.

### Unchanged

- monitor.py, pytest_eval.py, journal.py, metrics.py — all work as-is.
- Standalone experiment flow — untouched.

## Deferred

These are real requirements but out of scope for the initial
implementation:

- **Container-based eval execution.** Build from Dockerfile.foundry +
  experiment's pyproject.toml. Ensures reproducible eval environment.
- **Full infra composition.** Some experiments need running services
  (Redis, Neo4j, Milvus). A docker-compose per experiment or a shared
  infra stack.
- **Per-experiment pyproject.toml.** Each experiment declares its own
  dependencies vs what the target modules usually run with. The
  `environment:` field in goal.yaml is the hook for this.
- **Multi-repo targets.** An experiment that modifies files across
  multiple LOGOS repos (e.g., logos + sophia). Would need multiple
  worktrees coordinated by the harness agent.
