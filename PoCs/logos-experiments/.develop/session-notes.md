# Session Notes — teams-develop on vjepa_clip_alignment
Date: 2026-03-10

## Bugs & Issues Encountered

### 1. Eval script PASS/FAIL logic is incomplete
`eval/test_alignment.py` `check_pass()` only checks `cosine >= 0.70` and `stability == 1.0`.
It does NOT check `retrieval_accuracy_at_5` or `retrieval_accuracy_at_1` despite reporting them.
The eval said `[EVAL] PASS ✅` when R@5=0.24 and R@1=0.08 — both far below target.
**Impact:** Silent false positive. Cannot trust the eval verdict.
**Status:** User will fix the eval script separately.

### 2. HDF5 key names differed from spec
Spec said keys were `jepa` and `clip_text`.
Actual keys in the HDF5: `jepa_embeddings` and `clip_text_embeddings`.
**Impact:** data_prep.py had to auto-detect both naming conventions.
**Resolution:** Implementer handled it; should be corrected in spec/goal.yaml.

### 3. Broken test imports (test_model.py)
`test_model.py` used `from workspace.model import ...` which failed when pytest ran
from within `workspace/`. Inconsistent with `test_data_prep.py` which used `from data_prep import`.
**Resolution:** Fixed by implementer — added sys.path manipulation.

### 4. Synthetic smoke test default dims mismatch
`--synthetic` defaults to 768→512 dims. Our model is 1024→768.
Running `python eval/test_alignment.py --projector workspace/projector.py --synthetic`
gives misleading output unless `--synthetic-d-vjepa 1024 --synthetic-d-clip 768` is specified.
**Impact:** Minor confusion. Interface test passes with explicit flags.

### 5. InfoNCE collapse at epoch 31
Hard switch from MSE-only to InfoNCE-only caused cosine to drop from 0.86 to 0.29.
Retrieval never recovered. Best checkpoint from MSE phase has cosine=0.86 but R@5=0.24.
Root cause: MSE trains for closeness but not discrimination; abrupt InfoNCE transition
causes gradient shock before model has learned discriminative structure.
**Status:** Fix in progress (new impl agent being spawned).

### 6. GPU smaller than expected
Handoff said ~12GB local GPU. Actual: RTX 3070 at 8.6GB.
**Impact:** None — model is a small MLP on pre-computed embeddings, well within budget.
Training configs designed for 12GB were fine at 8.6GB with batch_size=128.

### 7. No git repository
Working directory is not a git repo. teams-develop skill assumes git throughout:
feature branches, PRs, `git diff` for reviewer, worktree isolation for implementers.
**Impact:** Skipped branching, PR creation, worktree isolation. Reviewer read files directly.
Commits not tracked — no audit trail of changes across iterations.

---

## Orchestrator Recommendations

### On agent management
- **Kill and respawn confused agents** rather than sending corrective messages.
  Patching a confused agent with 3-4 messages makes things worse, not better.
  If an agent has received contradictory instructions, shut it down and start fresh.
- **Brief quality is the orchestrator's responsibility.** If an agent goes off the rails,
  the first question is: was the brief complete? Fix the brief before respawning.

### On brief completeness
Implementers need:
1. The experiment goal and success metrics (not just the code task)
2. Current state — what ran, what the metrics were, what failed
3. Scope of their work — what they own and what they don't
4. The outcome they're responsible for, not just the implementation

Briefs that only describe the code task leave agents unable to reason about
whether their solution is on the right track.

### On PM scope
- **PM should not prescribe implementation approach** (e.g. specific loss schedules,
  hyperparameters). That's the implementer's call. PM states the problem and the target.
- PM prescribing implementation is over-reach and removes the implementer's agency
  to make good technical decisions.

### On the review loop for ML tasks
- **Review before integration is premature for ML tasks.** Code that passes unit tests
  may still fail to achieve metric targets. Review adds overhead without value if the
  training run will require another code change anyway.
- Better loop for ML: IMPLEMENT (code + unit tests) → INTEGRATE (train + eval)
  → if fail: IMPLEMENT again → ... → once metrics pass: REVIEW → COMPLETE.

### On state management across iterations
- **Write a state document after each training run** to `.develop/current_state.md`:
  what was tried, metrics achieved, what failed, next hypothesis.
- Any new implementer spawned for a fix should read this document first.
  This gives them situational awareness without PM having to explain everything.

### On skill fit
- **teams-develop is designed for software features, not ML research.**
  The skill works well for: interface design, TDD, code review, PR workflow.
  It needs adaptation for: iterative training, metric-driven success criteria,
  hyperparameter search, experiment tracking.
- Consider a hybrid: use teams-develop for code structure and review,
  but assign a dedicated "training agent" that owns the run→eval→diagnose loop.

### On eval script trust
- **Always read [METRIC] lines directly.** Never rely on [EVAL] PASS/FAIL alone.
  Verify eval PASS/FAIL logic matches goal.yaml before trusting it.

---

## What's Working Well

- Parallel T1+T2 implementation was smooth and efficient
- Researcher findings were thorough and accurate (caught HDF5 key mismatch)
- Architect design was clean and directly implementable
- Test coverage is solid (40 tests across 4 modules)
- Code review caught real issues (NaN guard, weights_only, untested NaN path)
- Agent shutdown/spawn lifecycle worked correctly
