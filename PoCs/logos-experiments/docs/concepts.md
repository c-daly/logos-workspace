# Core Concepts

logos-harness is a ticket system for autonomous research agents. You write tickets that describe what success looks like, agents figure out how to get there.

## The Ticket

Every experiment starts from a `goal.yaml`. This is the ticket — the contract between you and the agent.

A ticket has three required fields:

```yaml
objective: |
  What to achieve. Written in plain language.
  
success_criteria:
  - metric: accuracy
    threshold: 0.95
    primary: true

eval: eval/evaluate.py
```

And four optional fields the agent can use if they're there:

```yaml
context: |          # Why this matters, background information.
resources:          # Links to repos, papers, datasets.
  - https://...
prior_work: |       # What's been tried before.
notes: |            # Anything else that might help.
```

The agent should be able to solve the objective from the required fields alone. Optional fields are hints, not instructions. If they're absent, the agent figures it out or escalates.

### Writing Good Tickets

A good ticket constrains the *what* without prescribing the *how*:

**Good:** "Map V-JEPA embeddings into CLIP space with >0.7 cosine similarity"
**Bad:** "Train a 2-layer MLP with AdamW at lr=1e-4 to project V-JEPA into CLIP"

The first lets the agent try analytical approaches, neural approaches, or something you haven't thought of. The second is a recipe, not a ticket.

**Good:** Success criteria with measurable thresholds
**Bad:** "Make it work well" or "Improve performance"

### Multiple Success Criteria

Tickets can have multiple criteria. Mark one as `primary: true` — that's the main pass/fail gate. Others are secondary goals:

```yaml
success_criteria:
  - metric: mean_cosine_similarity
    threshold: 0.70
    primary: true
  - metric: retrieval_accuracy_at_5
    threshold: 0.60
  - metric: training_stability
    threshold: 1.0
    description: No NaN/Inf in outputs.
```

## The Eval

The eval is the ground truth. It measures whether the agent's solution meets the success criteria. It's a script that:

1. Loads the agent's solution
2. Runs it against test data (or generates its own)
3. Reports metrics in structured format
4. Returns pass or fail

Two patterns:

### Pytest Eval (simplest)

Write a test suite. The agent's job is to make the tests pass.

```python
# eval/test_search.py
def test_returns_results(engine):
    results = engine.search("python")
    assert isinstance(results, list)
    assert len(results) > 0
```

Run via: `harness-eval --tests eval/test_search.py`

### Custom Eval

For ML experiments where success is a metric threshold, not pass/fail tests:

```python
# eval/evaluate.py
def load_projector_module(path): ...
def evaluate(load_fn, project_fn, data): ...
def check_pass(metrics): ...

# Reports:
# [METRIC] cosine_similarity=0.72
# [EVAL] PASS ✅
```

### Eval Rules

1. **Never modify eval scripts during experimentation.** They are the ground truth.
2. Evals should be self-sufficient — discover data, discover dimensions, give helpful errors when something is missing.
3. If possible, support a `--synthetic` mode for testing the solution interface without real data.

## The Journal

The journal is the experiment's memory. It's an append-only log of what was tried, what happened, and what it means. Journal entries are markdown files in `journal/`:

```
journal/
├── 001_baseline_mlp.md
├── 002_contrastive_loss.md
└── 003_procrustes_alignment.md
```

Each entry documents an attempt:

```markdown
# Attempt 003: Procrustes Alignment

**Hypothesis:** Linear alignment via SVD might work since the spaces are structurally similar.
**Approach:** Computed Procrustes rotation matrix on training pairs.
**Results:** cosine_sim=0.68, retrieval@5=0.55
**Diagnosis:** Close to threshold. Linear alignment captures most of the structure.
**What I learned:** The spaces ARE mostly linearly related — nonlinear projection probably unnecessary.
**Next direction:** Try whitening before Procrustes, or ridge regression.
```

### Why the Journal Matters

- **Memory across sessions.** Agent teams don't persist context. The journal is how knowledge survives.
- **Prevents repeating failures.** Reading "NaN at batch 2 when lr > 1e-3" saves hours.
- **Coordination across agents.** When teammates work in parallel, the journal is where results converge.
- **Audit trail.** You can review what the agent tried and whether its reasoning was sound.

The harness provides utilities:
```bash
harness-journal my_experiment summary    # Overview of all attempts
harness-journal my_experiment failures   # Just the failure modes
```

## Constraints

`constraints.yaml` is optional. Use it when you have hard limits or known failure modes from prior work:

```yaml
time_limits:
  max_hours_per_run: 4
  max_total_gpu_hours: 40

known_failures:
  - id: F001
    name: "Device placement mismatch"
    description: |
      LoRA layers initialize on CPU while backbone is on GPU.

do_not_do:
  - "Do NOT fine-tune the full encoder"
```

This file persists hard-won knowledge. When an agent reads it, it avoids repeating past mistakes.

## Experiment Lifecycle

```
1. You write a ticket (goal.yaml) and eval
2. Agent reads CLAUDE.md, goal.yaml, constraints (if any), journal (if any)
3. Agent works — researching, building, testing
4. Agent runs the eval
5. Agent writes a journal entry
6. If eval passes → done
7. If eval fails → agent reads journal, plans next attempt, goto 3
```

The agent decides everything about step 3: what approach to try, whether to work sequentially or use agent teams for parallel work, whether to research first or build first, where to run (local, remote, RunPod).
