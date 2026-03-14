# Writing Evals

The eval is the most important part of an experiment. A well-written eval makes the agent's job clear; a poorly written eval makes the experiment fail regardless of the agent's ability.

## Principles

1. **Evals define the contract, not the implementation.** Specify what the solution must do, not how it should work internally.
2. **Evals should be self-sufficient.** Discover data, discover dimensions, give helpful errors when something is missing.
3. **Evals should be deterministic.** Same solution, same eval result. Use fixed random seeds for sampling.
4. **Evals should fail fast.** Check preconditions before expensive computation.
5. **Evals should report actionable failures.** "Assertion failed" is useless. "Output dimension is 999, expected 512" is useful.

## Pattern 1: Pytest Eval

Best for: well-defined interfaces, functional correctness, edge cases.

Write a test suite that imports and tests the agent's solution module:

```python
# eval/test_solution.py
import importlib.util
import pytest
from pathlib import Path

SOLUTION_PATH = Path(__file__).parent.parent / "workspace" / "solution.py"

@pytest.fixture
def solution():
    if not SOLUTION_PATH.exists():
        pytest.skip(f"No solution at {SOLUTION_PATH}")
    spec = importlib.util.spec_from_file_location("solution", SOLUTION_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class TestInterface:
    def test_has_required_functions(self, solution):
        assert hasattr(solution, "load")
        assert hasattr(solution, "process")

class TestBehavior:
    def test_handles_empty_input(self, solution):
        result = solution.process([])
        assert result == []

    def test_output_format(self, solution):
        result = solution.process(["hello"])
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)
```

Run with the harness adapter:
```bash
harness-eval --tests eval/test_solution.py
```

### Tips for Pytest Evals

- Use `pytest.skip()` when the solution doesn't exist yet, not `pytest.fail()`. Skipped tests show up as "no solution" rather than "broken solution."
- Test edge cases explicitly: empty input, None, special characters, very large input.
- Test the interface contract (return types, required fields) separately from the behavior.
- Use descriptive test names and docstrings — the agent reads these to understand what's expected.

## Pattern 2: Metric Eval

Best for: ML experiments where success is a continuous metric, not pass/fail tests.

```python
# eval/evaluate.py
import numpy as np
import importlib.util
from pathlib import Path

def load_solution(path):
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load, mod.project

def evaluate(load_fn, project_fn, val_data, val_targets):
    state = load_fn()
    predictions = project_fn(state, val_data)
    
    # Compute metrics
    cosine_sim = compute_cosine_similarity(predictions, val_targets)
    
    return {
        "cosine_similarity": cosine_sim,
        "stability": 0.0 if np.any(np.isnan(predictions)) else 1.0,
    }

def check_pass(metrics):
    return metrics["cosine_similarity"] >= 0.70

# Report structured output
for key, val in metrics.items():
    print(f"[METRIC] {key}={val}")
print(f"[EVAL] {'PASS' if check_pass(metrics) else 'FAIL'}")
```

### Making Metric Evals Self-Sufficient

Agents shouldn't need to know file paths, dimensions, or data formats in advance. The eval should discover these:

```python
# Bad — hardcoded
val_data = np.load("data/val/vjepa_768d.npy")  

# Good — discovered
val_path = find_file_matching("*vjepa*val*.npy", search_dirs)
val_data = np.load(val_path)
d_input = val_data.shape[1]  # Discover dimensions from data
```

### Synthetic Mode

For ML evals, support a `--synthetic` flag that tests the solution interface without real data:

```python
if args.synthetic:
    val_data = np.random.randn(100, 768).astype(np.float32)
    val_targets = np.random.randn(100, 512).astype(np.float32)
    # Metrics will be meaningless but interface is validated
```

This lets agents verify their code works before committing to expensive data generation.

## Solution Interface

Both eval patterns load the agent's solution as a Python module. The standard interface is:

```python
# workspace/solution.py

def load(checkpoint_path=None):
    """Initialize or load the solution. Return any state object."""
    ...

def do_the_thing(state, input_data):
    """The main function. Name and signature defined by the eval."""
    ...
```

The eval imports this module dynamically with `importlib`. The agent reads the eval to learn what functions it needs to implement and what the expected signatures are.

## Error Messages

Good eval error messages save the agent hours of debugging:

```python
# Bad
assert output.shape == expected_shape

# Good
assert output.shape[1] == d_clip, (
    f"Output dimension is {output.shape[1]}, expected {d_clip} (CLIP dim). "
    f"The projector must map into CLIP's embedding space."
)
```

```python
# Bad
raise FileNotFoundError(f"File not found: {path}")

# Good
raise FileNotFoundError(
    f"Validation data not found.\n"
    f"Expected: {path}\n"
    f"To generate: run V-JEPA encoder on videos, save as *vjepa*val*.npy\n"
    f"Or use --synthetic to test the interface."
)
```

## Checklist for New Evals

- [ ] Solution module is loaded dynamically (no hardcoded imports)
- [ ] All dimensions/shapes discovered from data, not hardcoded
- [ ] Helpful error when solution module is missing
- [ ] Helpful error when data is missing (with instructions to generate or use --synthetic)
- [ ] Contract violations (wrong output shape, wrong return type) give actionable messages
- [ ] NaN/Inf in outputs is explicitly checked
- [ ] `[METRIC] key=value` structured output for every metric
- [ ] `[EVAL] PASS` or `[EVAL] FAIL` final verdict
- [ ] Deterministic results (fixed random seeds where applicable)
- [ ] `--help` documents usage, interface, and expected solution format
