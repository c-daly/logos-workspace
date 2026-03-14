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
