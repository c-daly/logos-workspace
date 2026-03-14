"""
Experiment scaffolding — create new experiments interactively or from args.

Usage:
    python -m harness.new <name>
    python -m harness.new <name> --goal "Achieve X as measured by Y"
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime


from harness import find_experiments_dir

EXPERIMENTS_DIR = find_experiments_dir()


GOAL_TEMPLATE = """# {name}

objective: |
  {goal}

success_criteria:
  - metric: <metric_name>
    threshold: <number>
    primary: true

eval: eval/evaluate.py

# Everything below is optional. The agent works from what's above.
# Add context, resources, prior_work, or notes if useful.

# context: |
#   Why this matters, background info.

# resources:
#   - https://relevant-repo-or-paper

# prior_work: |
#   What's been tried before.

# notes: |
#   Anything else the agent should know but doesn't need to know.
"""


CONSTRAINTS_TEMPLATE = """\
# Constraints for: {name}
# Hard limits and known failure modes.
# The agent reads this before every attempt.

time_limits:
  max_hours_per_run: 4
  max_total_gpu_hours: 40

resource_limits: {{}}
  # gpu_memory_target_gb: 20

# Known failure modes from prior work (if any).
# Add entries as you discover them — these persist across agent sessions.
known_failures: []
  # - id: F001
  #   name: "Brief name"
  #   description: |
  #     What happened and why.

# Things the agent must not do.
do_not_do: []
  # - "Do NOT do X because Y"
"""


STATUS_TEMPLATE = """\
experiment: {name}
current_attempt: 0
status: not_started
best_result:
  attempt: null
  metric: null
  value: null
total_gpu_hours: 0.0
total_attempts: 0
last_updated: null
"""


EVAL_TEMPLATE = """\
\"\"\"
Evaluation harness for: {name}

DO NOT MODIFY THIS FILE DURING EXPERIMENTATION.

Interface:
    The --solution argument points to a Python module that must define:
    
    def load(checkpoint_path: str = None) -> Any:
        '''Load/initialize the solution. Return any state object.'''

    def evaluate(state: Any) -> dict:
        '''
        Run the solution and return metrics as a dict.
        Keys should match the metric names in goal.yaml.
        '''

Usage:
    python eval/evaluate.py --solution workspace/solution.py
    python eval/evaluate.py --solution workspace/solution.py --checkpoint path/to/state
    python eval/evaluate.py --solution workspace/solution.py --verbose
\"\"\"

import sys
import argparse
import importlib.util
import json
import traceback
from pathlib import Path
from typing import Any, Callable, Optional


def load_solution_module(path: str) -> tuple[Callable, Callable]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Solution not found: {{path}}")

    spec = importlib.util.spec_from_file_location("solution", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "load"):
        raise AttributeError(f"{{path}} must define load(checkpoint_path=None)")
    if not hasattr(module, "evaluate"):
        raise AttributeError(f"{{path}} must define evaluate(state) -> dict")

    return module.load, module.evaluate


def check_pass(metrics: dict) -> bool:
    \"\"\"
    TODO: Implement pass/fail logic based on goal.yaml thresholds.
    Update this when you fill in goal.yaml.
    \"\"\"
    raise NotImplementedError(
        "Fill in check_pass() with your success criteria from goal.yaml"
    )


def main():
    parser = argparse.ArgumentParser(description="Eval: {name}")
    parser.add_argument("--solution", required=True,
                        help="Path to solution module (.py with load/evaluate)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path for solution's load()")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    print(f"Solution: {{args.solution}}")
    if args.checkpoint:
        print(f"Checkpoint: {{args.checkpoint}}")
    print()

    try:
        load_fn, evaluate_fn = load_solution_module(args.solution)
    except (FileNotFoundError, AttributeError) as e:
        print(f"❌ {{e}}")
        sys.exit(2)

    try:
        state = load_fn(args.checkpoint)
        metrics = evaluate_fn(state)
    except NotImplementedError as e:
        print(f"⚠️  {{e}}")
        sys.exit(2)
    except Exception as e:
        print(f"❌ Eval crashed: {{e}}")
        traceback.print_exc()
        sys.exit(1)

    # Structured output
    for key, value in metrics.items():
        print(f"[METRIC] {{key}}={{value}}")

    try:
        passed = check_pass(metrics)
    except NotImplementedError:
        print("\\n[EVAL] UNKNOWN (check_pass not implemented yet)")
        sys.exit(2)

    print(f"\\n[EVAL] {{'PASS ✅' if passed else 'FAIL ❌'}}")

    if args.output_json:
        output = {{**metrics, "passed": passed}}
        Path(args.output_json).write_text(json.dumps(output, indent=2))

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
"""


def create_experiment(name: str, goal: str = "<describe the objective>"):
    exp_dir = EXPERIMENTS_DIR / name
    if exp_dir.exists():
        print(f"❌ Experiment '{name}' already exists at {exp_dir}")
        sys.exit(1)

    date = datetime.now().strftime("%Y-%m-%d")

    # Create directories
    for subdir in ["journal", "eval", "workspace", "checkpoints"]:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Write the ticket
    (exp_dir / "goal.yaml").write_text(
        GOAL_TEMPLATE.format(name=name, date=date, goal=goal)
    )

    # Status tracking
    (exp_dir / "status.yaml").write_text(
        STATUS_TEMPLATE.format(name=name)
    )

    # Eval skeleton
    (exp_dir / "eval" / "evaluate.py").write_text(
        EVAL_TEMPLATE.format(name=name)
    )

    # Journal placeholder
    (exp_dir / "journal" / ".gitkeep").touch()

    # Constraints only if needed — don't create an empty one
    # The agent and CLAUDE.md treat it as optional

    print(f"✅ Created: {exp_dir}")
    print(f"   Edit goal.yaml to define the ticket.")
    print(f"   Edit eval/evaluate.py to define how success is measured.")


def main():
    parser = argparse.ArgumentParser(description="Create a new experiment")
    parser.add_argument("name", help="Experiment name")
    parser.add_argument("--goal", "-g", default="<describe the objective>",
                        help="Objective (can also edit goal.yaml directly)")
    args = parser.parse_args()

    create_experiment(args.name, args.goal)


if __name__ == "__main__":
    main()
