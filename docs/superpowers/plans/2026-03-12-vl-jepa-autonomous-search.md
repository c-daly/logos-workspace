# VL-JEPA Autonomous Iterative Search -- Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `LOGOS/PoCs/vl_jepa/` -- an LLM-guided iterative training search that runs autonomously on RunPod with GPU/DataParallel support, resumable runs, and configurable stopping criteria.

**Architecture:** Evolve the existing `jepa_clip_translator` search system (LLM coordinator + config schema + training loop) with targeted additions: `nn.DataParallel`, false-negative masked InfoNCE loss, full history context for the LLM, per-experiment log saves, resume-on-restart, new stopping criteria, and a RunPod launch script pair.

**Tech Stack:** Python 3.10+, PyTorch 2.x, h5py, anthropic/openai SDK, bash

**Spec:** `docs/superpowers/specs/2026-03-12-vl-jepa-autonomous-search-design.md`

**Source for carry-overs:** `PoCs/jepa_clip_translator/`

---

## Chunk 1: Foundation -- config, translator, losses

### Task 1: Project scaffold + config.py

**Files:**
- Create: `PoCs/vl_jepa/__init__.py`
- Create: `PoCs/vl_jepa/tests/__init__.py`
- Create: `PoCs/vl_jepa/config.py`
- Create: `PoCs/vl_jepa/tests/test_config.py`

- [ ] **Step 1.1: Create directory structure**

```bash
mkdir -p /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/tests
touch /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/__init__.py
touch /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/tests/__init__.py
```

- [ ] **Step 1.2: Write failing test for infonce validation**

Create `PoCs/vl_jepa/tests/test_config.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from config import LossTerm, ExperimentConfig


def test_infonce_is_valid_loss_function():
    term = LossTerm(function="infonce", target="clip_image", weight=1.0)
    assert term.function == "infonce"


def test_invalid_function_raises():
    with pytest.raises(ValueError, match="function"):
        LossTerm(function="unknown_loss", target="clip_image")


def test_existing_functions_still_valid():
    for fn in ("mse", "cosine", "contrastive"):
        term = LossTerm(function=fn, target="clip_image")
        assert term.function == fn


def test_experiment_config_roundtrip_with_infonce():
    cfg = ExperimentConfig(experiment_id="test_infonce")
    cfg.loss.terms = [{"function": "infonce", "target": "clip_image", "weight": 1.0,
                       "temperature": 0.07, "label_smoothing": 0.0}]
    d = cfg.to_dict()
    cfg2 = ExperimentConfig.from_dict(d)
    assert cfg2.loss.terms[0]["function"] == "infonce"
```

- [ ] **Step 1.3: Run test to confirm it fails**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -m pytest tests/test_config.py -v 2>&1 | head -20
```

Expected: ModuleNotFoundError or ImportError (config.py not created yet).

- [ ] **Step 1.4: Copy config.py and add infonce validation**

```bash
cp /home/fearsidhe/projects/LOGOS/PoCs/jepa_clip_translator/config.py \
   /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/config.py
```

**Replace** the entire `LossTerm` class definition (the source uses `Literal["mse", "cosine", "contrastive"]` -- change it to `str` and add `__post_init__` runtime validation):

```python
VALID_FUNCTIONS = {"mse", "cosine", "contrastive", "infonce"}
VALID_TARGETS = {"clip_image", "clip_text_mean", "clip_text_first"}

@dataclass
class LossTerm:
    function: str = "mse"  # replaces Literal["mse","cosine","contrastive"]
    target: str = "clip_image"
    weight: float = 1.0
    temperature: float = 0.07
    label_smoothing: float = 0.0

    def __post_init__(self):
        if self.function not in VALID_FUNCTIONS:
            raise ValueError(f"Unknown loss function {self.function!r}. Valid: {VALID_FUNCTIONS}")
        if self.target not in VALID_TARGETS:
            raise ValueError(f"Unknown target {self.target!r}. Valid: {VALID_TARGETS}")
```

Note: resume loads plain dicts -- it never re-instantiates `LossTerm` from old logs, so the new target validation is safe.

- [ ] **Step 1.5: Run tests to confirm they pass**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -m pytest tests/test_config.py -v
```

Expected: 4 PASSED.

- [ ] **Step 1.6: Commit**

```bash
cd /home/fearsidhe/projects/LOGOS
git add PoCs/vl_jepa/
git commit -m "feat(vl_jepa): scaffold + config.py with infonce validation"
```

---

### Task 2: translator.py

**Files:**
- Create: `PoCs/vl_jepa/translator.py`
- Create: `PoCs/vl_jepa/tests/test_translator.py`

- [ ] **Step 2.1: Write failing tests**

Create `PoCs/vl_jepa/tests/test_translator.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, pytest
from config import ArchitectureConfig
from translator import build_translator, ResidualTranslator


def test_linear_output_shape():
    m = build_translator(ArchitectureConfig(type="linear"), 1024, 768)
    assert m(torch.randn(4, 1024)).shape == (4, 768)


def test_residual_output_l2_normalized():
    m = build_translator(ArchitectureConfig(type="residual", hidden_dim=512, num_blocks=2), 1024, 768)
    out = m(torch.randn(4, 1024))
    assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_mlp_output_shape():
    m = build_translator(ArchitectureConfig(type="mlp", hidden_dim=256, num_layers=2), 1024, 768)
    assert m(torch.randn(4, 1024)).shape == (4, 768)


def test_transformer_token_input():
    m = build_translator(ArchitectureConfig(type="transformer", hidden_dim=256, num_blocks=2, num_heads=4), 1024, 768)
    assert m(torch.randn(2, 8, 1024)).shape == (2, 768)


def test_pipeline_output_shape():
    cfg = ArchitectureConfig(stages=[
        {"type": "residual", "hidden_dim": 512, "num_blocks": 2},
        {"type": "mlp", "hidden_dim": 256, "num_layers": 2},
    ])
    assert build_translator(cfg, 1024, 768)(torch.randn(4, 1024)).shape == (4, 768)


def test_residual_kaiming_init():
    m = ResidualTranslator(1024, 768, hidden_dim=256, num_blocks=2)
    linear_weights = [p for n, p in m.named_parameters() if "weight" in n and p.dim() == 2]
    assert all(w.std().item() > 0.01 for w in linear_weights), "kaiming init should produce weights > 0.01 std"


def test_silu_activation():
    m = build_translator(ArchitectureConfig(type="residual", hidden_dim=256, num_blocks=2, activation="silu"), 1024, 768)
    assert m(torch.randn(4, 1024)).shape == (4, 768)
```

- [ ] **Step 2.2: Run to confirm failure**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -m pytest tests/test_translator.py -v 2>&1 | head -10
```

- [ ] **Step 2.3: Copy and update translator.py**

```bash
cp /home/fearsidhe/projects/LOGOS/PoCs/jepa_clip_translator/translator.py \
   /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/translator.py
```

In `ResidualTranslator._init_weights`, replace `trunc_normal_` with `kaiming_normal_`:

```python
def _init_weights(self, m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
```

Confirm `_activation` already includes `"silu": nn.SiLU` -- add it if missing.

- [ ] **Step 2.4: Run tests to confirm they pass**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -m pytest tests/test_translator.py -v
```

Expected: 7 PASSED.

- [ ] **Step 2.5: Commit**

```bash
cd /home/fearsidhe/projects/LOGOS
git add PoCs/vl_jepa/translator.py PoCs/vl_jepa/tests/test_translator.py
git commit -m "feat(vl_jepa): translator.py -- kaiming init, silu activation"
```

---

### Task 3: losses.py -- false-negative masked InfoNCE

**Files:**
- Create: `PoCs/vl_jepa/losses.py`
- Create: `PoCs/vl_jepa/tests/test_losses.py`

- [ ] **Step 3.1: Write failing tests**

Create `PoCs/vl_jepa/tests/test_losses.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from losses import compute_loss, _infonce


def test_mse_loss_nonneg():
    pred, tgt = torch.randn(8, 768), torch.randn(8, 768)
    r = compute_loss(pred, tgt, tgt, [{"function": "mse", "target": "clip_image",
                                        "weight": 1.0, "temperature": 0.07, "label_smoothing": 0.0}])
    assert r["loss"].item() >= 0


def test_cosine_loss_range():
    pred, tgt = torch.randn(8, 768), torch.randn(8, 768)
    r = compute_loss(pred, tgt, tgt, [{"function": "cosine", "target": "clip_image",
                                        "weight": 1.0, "temperature": 0.07, "label_smoothing": 0.0}])
    assert 0.0 <= r["loss"].item() <= 2.0


def test_infonce_loss_nonneg():
    pred, tgt = torch.randn(8, 768), torch.randn(8, 768)
    r = compute_loss(pred, tgt, tgt, [{"function": "infonce", "target": "clip_image",
                                        "weight": 1.0, "temperature": 0.07, "label_smoothing": 0.0}])
    assert r["loss"].item() >= 0


def test_infonce_fn_masking():
    torch.manual_seed(0)
    pred, tgt = torch.randn(8, 64), torch.randn(8, 64)
    # items 0,4 are duplicates (same video); masked = less false negatives = easier task
    loss_masked = _infonce(pred, tgt, batch_indices=[0,1,2,3,0,5,6,7]).item()
    loss_plain  = _infonce(pred, tgt, batch_indices=list(range(8))).item()
    assert loss_masked <= loss_plain + 0.5


def test_infonce_no_batch_indices():
    pred, tgt = torch.randn(8, 768), torch.randn(8, 768)
    loss = _infonce(pred, tgt, batch_indices=None)
    assert loss.shape == ()


def test_compute_loss_accepts_batch_indices():
    pred, tgt = torch.randn(8, 768), torch.randn(8, 768)
    terms = [{"function": "infonce", "target": "clip_image", "weight": 1.0,
              "temperature": 0.07, "label_smoothing": 0.0}]
    r = compute_loss(pred, tgt, tgt, terms, batch_indices=list(range(8)))
    assert r["loss"].item() >= 0


def test_token_level_pred_pooled():
    pred = torch.randn(4, 8, 768)
    tgt  = torch.randn(4, 768)
    terms = [{"function": "mse", "target": "clip_image", "weight": 1.0,
              "temperature": 0.07, "label_smoothing": 0.0}]
    r = compute_loss(pred, tgt, tgt, terms)
    assert r["loss"].shape == ()
```

- [ ] **Step 3.2: Run to confirm failure**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -m pytest tests/test_losses.py -v 2>&1 | head -10
```

- [ ] **Step 3.3: Copy losses.py and add _infonce**

```bash
cp /home/fearsidhe/projects/LOGOS/PoCs/jepa_clip_translator/losses.py \
   /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/losses.py
```

Add `_infonce` after `_contrastive`:

```python
def _infonce(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
    label_smoothing: float = 0.0,
    batch_indices: list | None = None,
    **_,
) -> torch.Tensor:
    """InfoNCE with optional false-negative masking."""
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    B = pred.shape[0]
    sim = pred_n @ target_n.T / temperature
    if batch_indices is not None:
        vi = torch.tensor(batch_indices, device=pred.device)
        fn_mask = (vi.unsqueeze(1) == vi.unsqueeze(0)) & ~torch.eye(B, dtype=torch.bool, device=pred.device)
        sim = sim.masked_fill(fn_mask, -1e9)
    labels = torch.arange(B, device=pred.device)
    return 0.5 * (
        F.cross_entropy(sim, labels, label_smoothing=label_smoothing)
        + F.cross_entropy(sim.T, labels, label_smoothing=label_smoothing)
    )
```

Add to `_PRIMITIVES`: `"infonce": _infonce`

Update `compute_loss` signature and loop to accept and forward `batch_indices`:

```python
def compute_loss(
    pred: torch.Tensor,
    clip_image: torch.Tensor,
    clip_text: torch.Tensor,
    terms: list,
    batch_indices: list | None = None,
) -> dict:
    total = torch.tensor(0.0, device=pred.device)
    total_weight = 0.0
    acc = None
    for term in terms:
        if isinstance(term, dict):
            t = LossTerm(**{k: v for k, v in term.items() if k in LossTerm.__dataclass_fields__})
        else:
            t = term
        target = _resolve_target(t.target, clip_image, clip_text)
        effective_pred = pred.mean(dim=1) if pred.dim() > target.dim() else pred
        raw = _PRIMITIVES[t.function](
            effective_pred, target,
            temperature=t.temperature,
            label_smoothing=t.label_smoothing,
            batch_indices=batch_indices if t.function == "infonce" else None,
        )
        if isinstance(raw, dict):
            loss_val = raw["loss"]
            if "accuracy" in raw and acc is None:
                acc = raw["accuracy"]
        else:
            loss_val = raw
        total = total + t.weight * loss_val
        total_weight += t.weight
    out: dict = {"loss": total / max(total_weight, 1e-8)}
    if acc is not None:
        out["accuracy"] = acc
    return out
```

- [ ] **Step 3.4: Run tests to confirm they pass**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -m pytest tests/test_losses.py -v
```

Expected: 7 PASSED.

- [ ] **Step 3.5: Commit**

```bash
cd /home/fearsidhe/projects/LOGOS
git add PoCs/vl_jepa/losses.py PoCs/vl_jepa/tests/test_losses.py
git commit -m "feat(vl_jepa): losses.py -- false-negative masked InfoNCE"
```

---

## Chunk 2: Coordinator

### Task 4: coordinator.py -- full history context

**Files:**
- Create: `PoCs/vl_jepa/coordinator.py`
- Create: `PoCs/vl_jepa/tests/test_coordinator.py`

- [ ] **Step 4.1: Write failing tests**

Create `PoCs/vl_jepa/tests/test_coordinator.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from coordinator import generate_round1_configs, generate_next_configs_random, _build_history_context, LLMConfig
from config import ExperimentConfig


def _fake(n=10, best_cos=0.78):
    base_cfg = {
        "architecture": {"type": "residual"}, "vjepa_dim": 1024, "clip_dim": 768,
        "loss": {"terms": [{"function": "mse", "target": "clip_image", "weight": 1.0,
                             "temperature": 0.07, "label_smoothing": 0.0}],
                 "warmup_terms": [], "warmup_epochs": 0},
        "training": {"optimizer": "adamw", "lr": 3e-4, "lr_min": 1e-6, "lr_schedule": "cosine",
                     "warmup_epochs": 5, "cooldown_epochs": 10, "cooldown_lr": 1e-6,
                     "weight_decay": 0.01, "batch_size": 256, "max_epochs": 200,
                     "early_stop_patience": 10, "grad_clip": 1.0},
        "data": {"noise_std": 0.0, "embedding_dropout": 0.0, "val_fraction": 0.15, "num_tokens": None},
    }
    return [{"experiment_id": f"exp_{i:03d}", "val_loss": 0.001 + i*0.0001,
             "val_cosine_sim": best_cos - i*0.002, "epochs_trained": 100,
             "best_epoch": 80, "config": {**base_cfg, "experiment_id": f"exp_{i:03d}"},
             "history": [{"epoch": 1, "train_loss": 0.1, "val_loss": 0.001,
                          "val_cosine_sim": best_cos, "lr": 3e-4}]}
            for i in range(n)]


def test_history_context_top10():
    ctx = _build_history_context(_fake(15), 0)
    assert "top10" in ctx and len(ctx["top10"]) == 10


def test_history_context_last5():
    ctx = _build_history_context(_fake(12), 2)
    assert "last5_trajectory" in ctx and len(ctx["last5_trajectory"]) == 5


def test_history_context_arch_counts():
    ctx = _build_history_context(_fake(8), 0)
    assert "architecture_counts" in ctx
    assert sum(ctx["architecture_counts"].values()) == 8


def test_history_context_plateau_info():
    ctx = _build_history_context(_fake(10), 3)
    assert ctx["rounds_without_improvement"] == 3
    assert "best_val_cosine_sim" in ctx


def test_round1_returns_5_configs():
    cfgs = generate_round1_configs()
    assert len(cfgs) == 5
    assert all(isinstance(c, ExperimentConfig) for c in cfgs)


def test_random_mutator_no_contrastive():
    results = _fake(5)
    for seed in range(20):
        cfgs = generate_next_configs_random(results, num_configs=3, seed=seed)
        for cfg in cfgs:
            for term in cfg.loss.terms:
                fn = term.get("function") if isinstance(term, dict) else term.function
                assert fn != "contrastive", f"Seed {seed}: random mutator should not use contrastive"


def test_llm_config_defaults():
    cfg = LLMConfig()
    assert len(cfg.model) > 0
```

- [ ] **Step 4.2: Run to confirm failure**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -m pytest tests/test_coordinator.py -v 2>&1 | head -10
```

- [ ] **Step 4.3: Copy coordinator.py and apply changes**

```bash
cp /home/fearsidhe/projects/LOGOS/PoCs/jepa_clip_translator/coordinator.py \
   /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/coordinator.py
```

**Change 1: Add `_build_history_context`** (insert before `_call_llm`):

```python
def _build_history_context(all_results: list[dict], rounds_without_improvement: int) -> dict:
    compact = []
    for r in all_results:
        c = {k: v for k, v in r.items() if k not in ("best_state", "history")}
        h = r.get("history", [])
        if h:
            c["history_summary"] = {"first": h[0], "last": h[-1],
                                    "best": h[r.get("best_epoch", 1) - 1] if r.get("best_epoch") else h[0]}
        compact.append(c)

    sorted_all = sorted(compact, key=lambda r: -r.get("val_cosine_sim", 0))
    arch_counts: dict[str, int] = {}
    for r in compact:
        arch = r.get("config", {}).get("architecture", {})
        t = "pipeline" if arch.get("stages") else arch.get("type", "unknown")
        arch_counts[t] = arch_counts.get(t, 0) + 1

    return {
        "total_experiments": len(all_results),
        "rounds_without_improvement": rounds_without_improvement,
        "best_val_cosine_sim": sorted_all[0].get("val_cosine_sim", 0.0) if sorted_all else 0.0,
        "top10": sorted_all[:10],
        "last5_trajectory": compact[-5:],
        "architecture_counts": arch_counts,
    }
```

**Change 2: Update `_call_llm` prompt block** (replace existing prompt assembly):

```python
def _call_llm(results, num_configs, llm_config, rounds_without_improvement=0):
    ctx = _build_history_context(results, rounds_without_improvement)
    plateau = ctx["rounds_without_improvement"] >= 2 and ctx["total_experiments"] >= 5
    prompt = (
        f"Total experiments: {ctx['total_experiments']}  |  "
        f"Best val_cosine_sim: {ctx['best_val_cosine_sim']:.4f}  |  "
        f"Rounds without improvement: {ctx['rounds_without_improvement']}\n"
        f"Architecture attempts: {ctx['architecture_counts']}\n"
        + ("Status: PLATEAUING -- try something genuinely different.\n" if plateau else "")
        + f"\nTop 10 results:\n{json.dumps(ctx['top10'], indent=2)}"
        + f"\n\nLast 5 experiments (trajectory):\n{json.dumps(ctx['last5_trajectory'], indent=2)}"
        + f"\n\nPropose {num_configs} new ExperimentConfig dicts."
    )
    print(f"  Calling {llm_config} ...")
    # ... rest unchanged (provider dispatch to _call_openai / _call_anthropic) ...
```

**Change 3: Update `generate_next_configs`** to accept and forward `rounds_without_improvement`:

```python
def generate_next_configs(all_results, num_configs=2, llm_config=None, rounds_without_improvement=0):
    if llm_config is None:
        llm_config = LLMConfig()
    configs = _call_llm(all_results, num_configs, llm_config, rounds_without_improvement)
    if configs:
        print(f"  LLM ({llm_config}) proposed {len(configs)} configs.")
        return configs
    print("  LLM unavailable, falling back to random mutation.")
    return generate_next_configs_random(all_results, num_configs)
```

**Change 4: Update `SYSTEM_PROMPT`** -- replace the contrastive warning block with:

```
IMPORTANT -- loss function guidance:
- Vanilla "contrastive" causes training collapse. Do NOT use it.
- Use "infonce" (false-negative masked InfoNCE) for contrastive objectives -- it is safe and effective.
- Mixed targets work well: 0.7xmse->clip_text_mean + 0.3xcosine->clip_image improved text retrieval significantly.
- Good combination: 0.5xmse->clip_image + 0.3xcosine->clip_text_mean + 0.2xinfonce->clip_image
```

Also update the schema line to: `function (mse|cosine|infonce)` and remove `contrastive`.

**Change 5: Update random mutator** -- replace `rng.choice(["mse", "cosine", "contrastive"])` with `rng.choice(["mse", "cosine", "infonce"])`.

- [ ] **Step 4.4: Run tests to confirm they pass**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -m pytest tests/test_coordinator.py -v
```

Expected: 7 PASSED.

- [ ] **Step 4.5: Commit**

```bash
cd /home/fearsidhe/projects/LOGOS
git add PoCs/vl_jepa/coordinator.py PoCs/vl_jepa/tests/test_coordinator.py
git commit -m "feat(vl_jepa): coordinator.py -- full history context, infonce guidance"
```

---

## Chunk 3: Search loop

### Task 5: search.py -- DataParallel, resume, per-experiment save, stopping criteria

**Files:**
- Create: `PoCs/vl_jepa/search.py`
- Create: `PoCs/vl_jepa/tests/test_search.py`

- [ ] **Step 5.1: Copy search.py as baseline**

```bash
cp /home/fearsidhe/projects/LOGOS/PoCs/jepa_clip_translator/search.py \
   /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/search.py
```

- [ ] **Step 5.2: Write failing tests**

Create `PoCs/vl_jepa/tests/test_search.py`:

```python
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, pytest
from search import _check_stopping, _save_log, _load_existing_log, run_experiment
from config import ExperimentConfig, ArchitectureConfig, TrainingConfig, LossConfig, DataConfig


def _fake(n=5, cos=0.78):
    return [{"experiment_id": f"exp_{i}", "val_loss": 0.001, "val_cosine_sim": cos,
             "epochs_trained": 10, "best_epoch": 8, "config": {}, "history": [],
             "best_state": None} for i in range(n)]


# --- stopping criteria ---

def test_stop_max_experiments():
    stop, reason = _check_stopping(_fake(10), 2, 10, max_experiments=10,
                                   target_metric=None, target_value=None,
                                   convergence_patience=None, rounds_without_improvement=0)
    assert stop and "max_experiments" in reason


def test_no_stop_below_max():
    stop, _ = _check_stopping(_fake(9), 2, 10, max_experiments=10,
                               target_metric=None, target_value=None,
                               convergence_patience=None, rounds_without_improvement=0)
    assert not stop


def test_stop_target_metric_hit():
    stop, reason = _check_stopping(_fake(5, cos=0.85), 1, 10, max_experiments=None,
                                   target_metric="val_cosine_sim", target_value=0.80,
                                   convergence_patience=None, rounds_without_improvement=0)
    assert stop and "target" in reason


def test_no_stop_target_not_hit():
    stop, _ = _check_stopping(_fake(5, cos=0.75), 1, 10, max_experiments=None,
                               target_metric="val_cosine_sim", target_value=0.80,
                               convergence_patience=None, rounds_without_improvement=0)
    assert not stop


def test_stop_convergence_patience():
    stop, reason = _check_stopping(_fake(5), 3, 10, max_experiments=None,
                                   target_metric=None, target_value=None,
                                   convergence_patience=3, rounds_without_improvement=3)
    assert stop


def test_stop_max_rounds():
    stop, _ = _check_stopping(_fake(5), 10, 10, max_experiments=None,
                               target_metric=None, target_value=None,
                               convergence_patience=None, rounds_without_improvement=0)
    assert stop


# --- save / resume ---

def test_save_log_writes_json():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    _save_log(_fake(3), path)
    with open(path) as f:
        data = json.load(f)
    assert len(data["experiments"]) == 3
    os.unlink(path)


def test_save_log_excludes_best_state():
    results = _fake(2)
    results[0]["best_state"] = {"w": [1.0]}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    _save_log(results, path)
    with open(path) as f:
        data = json.load(f)
    assert all("best_state" not in e for e in data["experiments"])
    os.unlink(path)


def test_load_existing_log_roundtrip():
    results = _fake(4)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({"experiments": [{k: v for k, v in r.items() if k != "best_state"}
                                    for r in results]}, f)
        path = f.name
    loaded = _load_existing_log(path)
    assert len(loaded) == 4
    os.unlink(path)


def test_load_missing_file_returns_empty():
    assert _load_existing_log("/tmp/vl_jepa_nonexistent_xyz.json") == []


# --- run_experiment smoke test ---

def _minimal_cfg():
    return ExperimentConfig(
        experiment_id="test_run",
        architecture=ArchitectureConfig(type="linear"),
        training=TrainingConfig(batch_size=16, max_epochs=2, early_stop_patience=2,
                                lr=1e-3, warmup_epochs=0, cooldown_epochs=0,
                                weight_decay=0.0, grad_clip=0.0, lr_schedule="none",
                                lr_min=1e-6, cooldown_lr=1e-7, optimizer="adamw"),
        loss=LossConfig(terms=[{"function": "mse", "target": "clip_image",
                                "weight": 1.0, "temperature": 0.07, "label_smoothing": 0.0}]),
        data=DataConfig(),
    )


def test_run_experiment_keys():
    torch.manual_seed(42)
    train_data = {"jepa": torch.randn(32, 1024), "clip_image": torch.randn(32, 768),
                  "clip_text": torch.randn(32, 5, 768)}
    val_data   = {"jepa": torch.randn(8, 1024),  "clip_image": torch.randn(8, 768),
                  "clip_text": torch.randn(8, 5, 768)}
    result = run_experiment(_minimal_cfg(), train_data, val_data, torch.device("cpu"))
    for key in ("experiment_id", "val_loss", "val_cosine_sim", "epochs_trained", "best_state"):
        assert key in result
```

- [ ] **Step 5.3: Run to confirm failure**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -m pytest tests/test_search.py -v 2>&1 | head -20
```

- [ ] **Step 5.4: Add `_save_log` and `_load_existing_log` to search.py**

After imports, add:

```python
def _save_log(all_results: list[dict], output_path: str) -> None:
    """Write experiment log to JSON, excluding non-serializable best_state."""
    log_data = {"experiments": [{k: v for k, v in r.items() if k != "best_state"}
                                 for r in all_results]}
    with open(output_path, "w") as f:
        json.dump(log_data, f, indent=2)


def _load_existing_log(path: str) -> list[dict]:
    """Load prior results from JSON. Returns [] if file missing or malformed."""
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f).get("experiments", [])
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Could not load existing log %s: %s", path, e)
        return []
```

- [ ] **Step 5.5: Add `_check_stopping` to search.py**

```python
def _check_stopping(
    all_results: list[dict],
    round_num: int,
    max_rounds: int,
    max_experiments: int | None,
    target_metric: str | None,
    target_value: float | None,
    convergence_patience: int | None,
    rounds_without_improvement: int,
) -> tuple[bool, str]:
    if max_experiments is not None and len(all_results) >= max_experiments:
        return True, f"max_experiments={max_experiments} reached ({len(all_results)} run)"
    if target_metric and target_value is not None and all_results:
        best = max(r.get(target_metric, 0.0) for r in all_results)
        if best >= target_value:
            return True, f"target {target_metric}>={target_value:.4f} reached (best={best:.4f})"
    if convergence_patience is not None and rounds_without_improvement >= convergence_patience:
        return True, f"convergence patience={convergence_patience} ({rounds_without_improvement} rounds no improvement)"
    if round_num >= max_rounds:
        return True, f"max_rounds={max_rounds} reached"
    return False, ""
```

- [ ] **Step 5.6: Add DataParallel to `run_experiment`**

After `model = build_translator(...).to(device)`:

```python
if torch.cuda.device_count() > 1:
    logger.info("  Using DataParallel across %d GPUs", torch.cuda.device_count())
    model = nn.DataParallel(model)
```

When saving `best_state`, unwrap:

```python
_m = model.module if isinstance(model, nn.DataParallel) else model
best_state = {k: v.cpu().clone() for k, v in _m.state_dict().items()}
```

- [ ] **Step 5.7: Update `run_search` signature and loop**

Add parameters to `run_search`:

```python
def run_search(
    embeddings_path, max_rounds=5, configs_per_round=5,
    convergence_patience=None, max_experiments=None,
    target_metric=None, target_value=None, resume=False,
    device=torch.device("cpu"), output_path="experiment_log.json", llm_config=None,
):
```

After loading embeddings, initialize results:

```python
all_results: list[dict] = _load_existing_log(output_path) if resume else []
```

Inside experiment loop after `all_results.append(result)`:

```python
_save_log(all_results, output_path)
```

Replace old convergence check at end of each round with:

```python
should_stop, stop_reason = _check_stopping(
    all_results, round_num, max_rounds,
    max_experiments, target_metric, target_value,
    convergence_patience, rounds_without_improvement,
)
if should_stop:
    logger.info("  Stopping: %s", stop_reason)
    break
```

Update `generate_next_configs` call:

```python
configs = generate_next_configs(
    all_results, num_configs=configs_per_round,
    llm_config=llm_config,
    rounds_without_improvement=rounds_without_improvement,
)
```

- [ ] **Step 5.8: Update `main()` CLI**

```python
parser.add_argument("--max-experiments", type=int, default=None)
parser.add_argument("--target-metric", default=None, choices=["val_cosine_sim", "R@1", "R@5"])
parser.add_argument("--target-value", type=float, default=None)
parser.add_argument("--resume", action="store_true")
```

Pass to `run_search`: `max_experiments=args.max_experiments, target_metric=args.target_metric, target_value=args.target_value, resume=args.resume`

- [ ] **Step 5.9: Run all tests**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -m pytest tests/ -v
```

Expected: all PASSED.

- [ ] **Step 5.10: Commit**

```bash
cd /home/fearsidhe/projects/LOGOS
git add PoCs/vl_jepa/search.py PoCs/vl_jepa/tests/test_search.py
git commit -m "feat(vl_jepa): search.py -- DataParallel, resume, per-exp save, stopping criteria"
```

---

## Chunk 4: Evaluate + Deployment

### Task 6: evaluate.py

**Files:**
- Create: `PoCs/vl_jepa/evaluate.py`

- [ ] **Step 6.1: Copy and fix checkpoint key**

```bash
cp /home/fearsidhe/projects/LOGOS/PoCs/jepa_clip_translator/evaluate.py \
   /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/evaluate.py
```

In the checkpoint loading path, support both key formats:

```python
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
state = ckpt.get("model_state_dict", ckpt.get("best_state", ckpt))
model.load_state_dict({k: v.to(device) for k, v in state.items()})
```

- [ ] **Step 6.2: Smoke test**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -c "import evaluate; print('OK')"
```

- [ ] **Step 6.3: Commit**

```bash
cd /home/fearsidhe/projects/LOGOS
git add PoCs/vl_jepa/evaluate.py
git commit -m "feat(vl_jepa): evaluate.py -- checkpoint key compatibility fix"
```

---

### Task 7: requirements.txt + shell scripts

**Files:**
- Create: `PoCs/vl_jepa/requirements.txt`
- Create: `PoCs/vl_jepa/podrun_search.sh`
- Create: `PoCs/vl_jepa/launch_search.sh`

- [ ] **Step 7.1: Create requirements.txt**

```
torch>=2.1.0
h5py>=3.9.0
numpy>=1.24.0
anthropic>=0.25.0
openai>=1.0.0
```

- [ ] **Step 7.2: Create podrun_search.sh**

```bash
#!/usr/bin/env bash
# On-pod search launcher. Required: ANTHROPIC_API_KEY, POD_ID, RUNPOD_API_KEY
# Optional: ROUNDS=10 CPR=3 MAX_EXPERIMENTS=150 TARGET_R5=0.82 CONV_PATIENCE=4
set -e

WORKSPACE=/workspace/vl_jepa
H5=/workspace/msrvtt_embeddings.h5
LOG=$WORKSPACE/search.log

ROUNDS=${ROUNDS:-10}
CPR=${CPR:-3}
MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-150}
TARGET_R5=${TARGET_R5:-0.82}
CONV_PATIENCE=${CONV_PATIENCE:-4}

echo "=== VL-JEPA Autonomous Search ==="
python3 -c "
import torch; n=torch.cuda.device_count(); print(f'GPUs: {n}')
[print(f'  GPU {i}: {torch.cuda.get_device_properties(i).name}') for i in range(n)]
"

pip install -q -r $WORKSPACE/requirements.txt

if [ -n "$POD_ID" ] && [ -n "$RUNPOD_API_KEY" ]; then
    STOP_CMD="curl -s -X POST https://api.runpod.io/v2/pod/$POD_ID/stop -H 'Authorization: Bearer $RUNPOD_API_KEY' && echo 'Pod stop requested.'"
else
    STOP_CMD="echo 'No POD_ID/RUNPOD_API_KEY set -- pod will not auto-stop.'"
fi

TRAIN_CMD="cd $WORKSPACE && python search.py \
  --embeddings $H5 \
  --rounds $ROUNDS --configs-per-round $CPR \
  --max-experiments $MAX_EXPERIMENTS \
  --target-metric val_cosine_sim --target-value $TARGET_R5 \
  --convergence-patience $CONV_PATIENCE \
  --resume --output $WORKSPACE/experiment_log.json \
  --llm-provider anthropic 2>&1 | tee $LOG"

tmux new-session -d -s search "bash -c '$TRAIN_CMD; $STOP_CMD'"

echo ""
echo "Training launched in tmux session 'search'."
echo "  tmux attach -t search"
echo "  tail -f $LOG"
sleep 2 && tail -f $LOG
```

- [ ] **Step 7.3: Create launch_search.sh**

```bash
#!/usr/bin/env bash
# Local launch: start pod, upload, run podrun_search.sh.
# Required: POD_ID, RUNPOD_API_KEY, ANTHROPIC_API_KEY
# Optional: SSH_KEY, ROUNDS, CPR, MAX_EXPERIMENTS, TARGET_R5, CONV_PATIENCE
set -e

: "${POD_ID:?Set POD_ID}"
: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_runpod_key}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
H5="$HOME/projects/LOGOS/PoCs/jepa_clip_translator/msrvtt_embeddings.h5"
SSH_USER="${POD_ID}-root"
SSH_HOST="ssh.runpod.io"
SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no"

echo "=== VL-JEPA Launch === Pod: $POD_ID"

echo "--- Starting pod ---"
curl -s -X POST "https://api.runpod.io/v2/pod/$POD_ID/start" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print('Response:', d.get('status', d))"

echo "--- Waiting for SSH (max 5 min) ---"
MAX_WAIT=300; WAITED=0
while true; do
    PORTS=$( curl -s "https://api.runpod.io/v2/pod/$POD_ID" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" | python3 -c \
      "import sys,json; d=json.load(sys.stdin)
pods=d.get('data',{}).get('myself',{}).get('pods',[{}])
p=pods[0] if pods else {}
print(p.get('desiredStatus','?'), str(p.get('runtime',{}).get('ports','')))" 2>/dev/null || echo "?")
    echo "  $PORTS (${WAITED}s elapsed)"
    if echo "$PORTS" | grep -q RUNNING && echo "$PORTS" | grep -q "22"; then break; fi
    sleep 10; WAITED=$((WAITED+10))
    [ $WAITED -ge $MAX_WAIT ] && { echo "Timeout waiting for pod."; exit 1; }
done
sleep 5

echo "--- Uploading vl_jepa/ ---"
$SSH "$SSH_USER@$SSH_HOST" "mkdir -p /workspace/vl_jepa"
rsync -avz --progress -e "$SSH" "$SCRIPT_DIR/" "$SSH_USER@$SSH_HOST:/workspace/vl_jepa/"

echo "--- Uploading msrvtt_embeddings.h5 (skips if unchanged) ---"
rsync -avz --progress --ignore-existing -e "$SSH" "$H5" "$SSH_USER@$SSH_HOST:/workspace/"

echo "--- Launching training ---"
$SSH "$SSH_USER@$SSH_HOST" \
  "POD_ID=$POD_ID RUNPOD_API_KEY=$RUNPOD_API_KEY ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
   ROUNDS=${ROUNDS:-10} CPR=${CPR:-3} MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-150} \
   TARGET_R5=${TARGET_R5:-0.82} CONV_PATIENCE=${CONV_PATIENCE:-4} \
   bash /workspace/vl_jepa/podrun_search.sh &"

echo ""
echo "============================================"
echo "Training launched."
echo "  $SSH $SSH_USER@$SSH_HOST 'tail -f /workspace/vl_jepa/search.log'"
echo "  $SSH $SSH_USER@$SSH_HOST 'tmux attach -t search'"
echo "Artifacts when done:"
echo "  rsync -avz -e '$SSH' $SSH_USER@$SSH_HOST:/workspace/vl_jepa/ ./results/"
echo "============================================"
```

- [ ] **Step 7.4: Make scripts executable**

```bash
chmod +x /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/podrun_search.sh
chmod +x /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/launch_search.sh
```

- [ ] **Step 7.5: Validate shell syntax**

```bash
bash -n /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/podrun_search.sh && echo "podrun OK"
bash -n /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/launch_search.sh && echo "launch OK"
```

- [ ] **Step 7.6: Full import smoke test**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa
python3 -c "import config, translator, losses, coordinator, search, evaluate; print('All imports OK')"
```

- [ ] **Step 7.7: Full test suite**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -m pytest tests/ -v
```

Expected: all PASSED.

- [ ] **Step 7.8: Final commit**

```bash
cd /home/fearsidhe/projects/LOGOS
git add PoCs/vl_jepa/
git commit -m "feat(vl_jepa): complete -- requirements, podrun_search.sh, launch_search.sh"
```

---

## End-to-End Verification

- [ ] **Dry-run: load existing experiment log and build history context**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa
python3 -c "
from search import _load_existing_log
results = _load_existing_log('../jepa_clip_translator/experiment_log.json')
print(f'Loaded {len(results)} prior experiments')
from coordinator import _build_history_context
ctx = _build_history_context(results, rounds_without_improvement=0)
print(f'Best cos sim: {ctx[\"best_val_cosine_sim\"]:.4f}')
print(f'Arch counts: {ctx[\"architecture_counts\"]}')
"
```

Expected: `Loaded 75 prior experiments`, best cos ~0.785, residual dominant in arch counts.

- [ ] **Shell syntax check**

```bash
bash -n /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/launch_search.sh && echo "launch OK"
bash -n /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa/podrun_search.sh && echo "podrun OK"
```

- [ ] **Full test suite**

```bash
cd /home/fearsidhe/projects/LOGOS/PoCs/vl_jepa && python -m pytest tests/ -v --tb=short
```

All green = ready to launch.
