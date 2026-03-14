"""
Initialize a new harness project in any directory.

Usage:
    harness-init                  # Current directory
    harness-init /path/to/project # Specific directory
    python -m harness.init .

Creates:
    CLAUDE.md
    experiments/
    Justfile (optional)
"""

import shutil
import sys
from pathlib import Path
from importlib import resources


CLAUDE_MD = '''\
# Experiment Harness

You are an autonomous research agent. Your job is to achieve the experiment's goal.
How you get there is up to you.

## The Ticket

Each experiment starts from a **goal.yaml**. This is the ticket.

**Required:**
- `objective` — what to achieve
- `success_criteria` — measurable metrics and thresholds
- `eval` — the script that measures success

**Optional:**
- `context`, `resources`, `prior_work`, `notes`

## Experiment Structure

- **goal.yaml** — the ticket. Read it first.
- **eval/** — measures success. Do not modify.
- **constraints.yaml** — hard limits, known failures. May not exist.
- **journal/** — experiment memory. Write after meaningful work, read before planning.
- **status.yaml** — machine-readable state. Keep current.
- **workspace/** — yours. Create, destroy, restructure freely.
- **checkpoints/** — save promising intermediate states.

## What's Free

You decide what to try, how to orchestrate, when to research vs build,
how to structure code, and how to get data.

## Discipline

Before work: read goal.yaml, constraints.yaml (if exists), and journal.
After work: write a journal entry, update status.yaml.

## Orchestration

Use agent teams when work has independent parallel tracks.
Keep teams small (2-3). Plan first, then execute.
Journal is the shared memory across sessions and teams.

## Principles

1. Never modify eval scripts.
2. Never skip the journal.
3. Understand before you build.
4. The solution space is wide.
5. Fail fast.
6. Budget awareness.
7. Save intermediate states.
8. Coordinate, don't collide.
9. Escalate when stuck.
'''


def init_project(target: str = "."):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    # CLAUDE.md
    claude_md = target / "CLAUDE.md"
    if claude_md.exists():
        print(f"  CLAUDE.md already exists, skipping")
    else:
        claude_md.write_text(CLAUDE_MD)
        print(f"  Created CLAUDE.md")

    # experiments/
    exp_dir = target / "experiments"
    exp_dir.mkdir(exist_ok=True)
    print(f"  Created experiments/")

    print(f"\n✅ Harness initialized at {target}")
    print(f"   Create an experiment: harness-new my_experiment --goal 'Do the thing'")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    init_project(target)


if __name__ == "__main__":
    main()
