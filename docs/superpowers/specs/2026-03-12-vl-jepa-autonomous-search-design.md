# VL-JEPA Autonomous Iterative Search — Design Spec

**Date:** 2026-03-12
**Branch:** `full-jepa-token-grid`
**Location:** `LOGOS/PoCs/vl_jepa/`

---

## Context

Two separate systems exist today:

1. **`PoCs/jepa_clip_translator/`** — LLM-guided iterative hyperparameter search (`search.py` + `coordinator.py`). 75 experiments on CPU, LLM proposes configs each round. No GPU/DataParallel. Best result: ResidualTranslator, R@1=0.445, R@5=0.704.

2. **`claude_autonomous/logos-experiments/experiments/vjepa_clip_alignment/workspace/`** — GPU training scripts (`train_vljepa_v5.py`, `train_vljepa_v6.py`). Has `nn.DataParallel`, seconds-per-epoch with precomputed embeddings in RAM, false-negative masked InfoNCE, mixed clip_text+clip_image target. Single training runs only — no iteration/search.

**Goal:** Merge both into `LOGOS/PoCs/vl_jepa/` — a new PoC that runs the LLM-guided iterative search autonomously on RunPod with GPU support.

---

## Architecture

### Directory Layout

```
LOGOS/PoCs/vl_jepa/
├── search.py           # autonomous LLM-guided outer loop
├── coordinator.py      # LLM calls + config generation
├── translator.py       # all model architectures
├── losses.py           # MSE + cosine + false-negative masked InfoNCE
├── config.py           # ExperimentConfig schema
├── evaluate.py         # standalone retrieval evaluation
├── launch_search.sh    # LOCAL: full pod lifecycle (start → upload → run)
├── podrun_search.sh    # ON-POD: deps, tmux, run search.py, auto-stop
└── requirements.txt
```

No new abstractions beyond what already exists. Each file has one clear purpose.

---

## Components

### `search.py` — Autonomous Search Loop

Evolved from `jepa_clip_translator/search.py`. Changes:

**DataParallel** — added to `run_experiment()`:
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
# unwrap before checkpoint save:
state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
```

**Per-experiment log save** — `experiment_log.json` written after every experiment, not just at the end. Prevents progress loss on crash.

**Resume on startup** — if `--output` file exists and `--resume` is set, existing results load into `all_results`. Search continues from where it left off.

**New stopping criteria:**

| Flag | Behaviour |
|---|---|
| `--max-experiments N` | Stop after N total experiments |
| `--target-metric [R@1|R@5|val_cosine_sim]` | Stop when metric hits threshold |
| `--target-value F` | Threshold for `--target-metric` |
| `--convergence-patience N` | Stop after N rounds no improvement (existing, kept) |
| `--resume` | Load existing log on startup |

Stopping checks run after each round. On clean exit (code 0), `podrun_search.sh` auto-stops the pod.

**Unchanged:** LR schedule, optimizer logic, `_prepare_batch`, `_augment`, `compute_retrieval_metrics`, `evaluate_best`, round summaries, final leaderboard.

---

### `coordinator.py` — LLM Coordinator

Evolved from `jepa_clip_translator/coordinator.py`. One change:

**Full history context** — `generate_next_configs()` now passes to the LLM:
- Top-10 results (id, arch, loss, val_cosine_sim, val_loss, epochs)
- Last 5 results (trajectory — improving or flat?)
- Counts of architecture types tried (what's over/under-explored)
- Plateau note: rounds without improvement, best metric so far

System prompt updated with v6 findings:
- False-negative masked InfoNCE works; vanilla contrastive collapses
- Mixed target (0.7×clip_text + 0.3×clip_image) improves text retrieval

Round 1 configs (hand-crafted) unchanged: linear, MLP, residual, transformer, pipeline.

---

### `losses.py` — Loss Functions

Carried over from `jepa_clip_translator/losses.py` with one addition:

**False-negative masked InfoNCE** (`"infonce"` function value):
- Computes similarity matrix between predictions and targets
- Masks same-video pairs before softmax (prevents false negatives with multiple captions)
- Weight, temperature, label_smoothing configurable same as existing terms

`"mse"`, `"cosine"`, `"contrastive"` unchanged. Coordinator prompt warns against `"contrastive"`.

---

### `translator.py` — Model Definitions

Carried over from `jepa_clip_translator/translator.py` with one update:

**ResidualBlock init** — updated to match v6:
- Linear layers: `kaiming_normal_` (was `trunc_normal_` std=0.02)
- `"silu"` added as valid activation alongside `"gelu"` / `"relu"`

All architectures preserved: LinearTranslator, MLPTranslator, ResidualTranslator, TransformerTranslator, PipelineTranslator.

---

### `config.py` — ExperimentConfig

Carried over unchanged. `LossConfig.terms` gains `"infonce"` as a valid `function` value.

---

### `evaluate.py` — Standalone Evaluation

Carried over from `jepa_clip_translator/evaluate.py`. Checkpoint key fix: loads from `model_state_dict` (not `best_state`).

---

### `podrun_search.sh` — On-Pod Script

1. Print GPU info (`nvidia-smi`)
2. Install deps (`pip install -r requirements.txt`)
3. Launch `search.py` in tmux session `search`, args forwarded from env vars
4. Append auto-stop call to pod exit: `curl` RunPod stop API on clean exit
5. Print monitoring instructions + tail log

Required env vars: `POD_ID`, `RUNPOD_API_KEY`, `ANTHROPIC_API_KEY`.
Optional env vars with defaults: `ROUNDS=10`, `CPR=3`, `MAX_EXPERIMENTS=150`, `TARGET_R5=0.80`, `CONV_PATIENCE=4`.

---

### `launch_search.sh` — Local Launch Script

Single local command. Full lifecycle:

```
1. Validate env vars: POD_ID, RUNPOD_API_KEY, ANTHROPIC_API_KEY, SSH_KEY
2. POST https://api.runpod.io/v2/pod/$POD_ID/start
3. Poll GET .../status until RUNNING + SSH port available (10s backoff, 5 min timeout)
4. rsync LOGOS/PoCs/vl_jepa/ → pod:/workspace/vl_jepa/
5. rsync msrvtt_embeddings.h5 → pod:/workspace/ (--ignore-existing for speed)
6. SSH: bash /workspace/vl_jepa/podrun_search.sh
7. Print: ssh $POD_SSH "tail -f /workspace/vl_jepa/search.log"
          ssh $POD_SSH "tmux attach -t search"
8. Exit (pod is autonomous from here)
```

SSH: `ssh -i $SSH_KEY -o StrictHostKeyChecking=no $POD_USER@ssh.runpod.io`

---

## Data Flow

```
Local: launch_search.sh
  → RunPod API: start pod
  → rsync: vl_jepa/ + msrvtt_embeddings.h5 → /workspace/
  → SSH: podrun_search.sh

Pod: podrun_search.sh
  → tmux session "search": search.py
      → load_embeddings() → all tensors to GPU
      → Round 1: 5 hand-crafted configs
      → run_experiment(cfg):
          model.to(device)
          nn.DataParallel if device_count > 1
          train → save experiment_log.json
      → Round 2+: coordinator LLM call → new configs
      → repeat until stopping criterion
      → evaluate_best() → save best_translator_<id>.pt
      → exit 0
  → auto-stop: RunPod API

Local (any time):
  → ssh pod "tail -f /workspace/vl_jepa/search.log"
  → ssh pod "tmux attach -t search"
```

---

## Stopping Criteria (checked after each round, all optional)

1. `--max-experiments N` — total experiments >= N
2. `--target-metric` + `--target-value` — best metric >= threshold
3. `--convergence-patience N` — rounds without improvement >= N
4. `--rounds N` — max rounds reached

First criterion triggered wins.

---

## Error Handling

| Error | Handling |
|---|---|
| LLM unavailable | Coordinator falls back to random mutation |
| NaN loss | Experiment marked failed; coordinator receives failure signal |
| SSH not ready | 10s backoff, 5 min timeout, then abort |
| Interrupted run | `--resume` reloads log; continues from next round |
| Auto-stop API fails | Pod stays running; training already complete |

---

## Out of Scope

- DDP — DataParallel sufficient for 2-GPU pod; DDP would complicate process management
- Wandb/MLflow — stdout + JSON is sufficient for this PoC stage
- Optuna/Ray Tune — LLM-guided search is the stated approach
- Automated artifact pull-back — manual rsync after training

---

## Migration from `jepa_clip_translator`

Pass existing `experiment_log.json` (75 experiments) via `--output path --resume`. Coordinator uses prior results to inform new configs — effectively continuing the search from experiment 76.
