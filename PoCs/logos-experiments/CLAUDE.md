# logos-harness

Ticket-driven autonomous experiment framework. You read a ticket, work toward it, eval your progress, and journal what you learn.

## Protocol

### 1. Read the Ticket

Every experiment lives in `experiments/<name>/`. Read `goal.yaml` — it defines:

- **objective**: What to achieve (plain language)
- **success_criteria**: Measurable thresholds with one `primary: true`
- **eval**: Path to evaluation script or test suite

Optional: `context`, `resources`, `prior_work`, `notes`.

If `constraints.yaml` exists, read it too — time limits, known failures, things not to do.

### 2. Read the Journal

Check `experiments/<name>/journal/` for entries from prior attempts. Each entry records what was tried, what happened, and what to try next. **Never repeat a failed approach** without a new hypothesis.

### 3. Work

Use `experiments/<name>/workspace/` for your code, models, artifacts. The ticket tells you *what* — you decide *how*.

### 4. Eval

Run the eval specified in `goal.yaml`:

- **Pytest eval**: `harness-eval experiments/<name>` — runs a test suite, reports pass/fail
- **Custom eval**: `python experiments/<name>/eval/evaluate.py` — reports metrics against thresholds

### 5. Journal Your Results

After each attempt, add a journal entry:

```bash
harness-journal experiments/<name> add --attempt N --result "what happened" --next "what to try"
```

Or write directly to `experiments/<name>/journal/NNN_<summary>.md`.

### 6. Iterate or Done

If eval passes → done. If not → read your journal, form a new hypothesis, go to step 3.

## Agent Team Integration

For complex experiments, use agent teams to parallelize independent sub-tasks. The ticket defines the goal; teams coordinate how to get there.

## Key Rules

- **Ticket is the contract.** Don't modify `goal.yaml` — it's the spec.
- **Journal is memory.** Always journal before ending a session.
- **Eval is the judge.** Don't declare success without running eval.
- **No approach prescription.** The ticket says what, not how.

## CLI Commands

| Command | Purpose |
|---------|---------|
| `harness-new <name>` | Scaffold a new experiment |
| `harness-eval <experiment>` | Run pytest-based eval |
| `harness-journal <experiment> add` | Add a journal entry |
| `harness-journal <experiment> summary` | View attempt history |
| `harness-metrics <experiment>` | Show metric trends |
| `harness-monitor -- <cmd>` | Run a subprocess with NaN/timeout/OOM detection |

## Directory Structure

```
experiments/<name>/
├── goal.yaml           # The ticket (required)
├── constraints.yaml    # Guardrails (optional)
├── status.yaml         # Current state
├── eval/               # Evaluation scripts or test suite
├── journal/            # Append-only attempt log
├── workspace/          # Your working area
└── checkpoints/        # Saved models/artifacts
```
