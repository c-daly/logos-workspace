# logos-harness

Ticket-driven autonomous experiment framework. An agent reads a goal ticket, works toward it, evaluates progress, and journals learnings — iterating until done.

## Core Concepts

- **Ticket** (`goal.yaml`): Defines objective, success criteria, and eval path
- **Constraints** (`constraints.yaml`): Time limits, known failures, do-not-do lists
- **Journal**: Append-only experiment memory for cross-session knowledge
- **Eval**: Pluggable evaluation — custom metrics or pytest suites

## Quick Start

```bash
pip install -e ".[dev]"

# Create a new experiment
harness-new my-experiment

# Run evaluation
harness-eval experiments/my-experiment

# View journal
harness-journal experiments/my-experiment
```

## Eval Patterns

1. **Custom eval** — metric thresholds (e.g. ML research)
2. **Pytest eval** — test suite pass/fail (e.g. software engineering)

See `docs/concepts.md` for full documentation.
