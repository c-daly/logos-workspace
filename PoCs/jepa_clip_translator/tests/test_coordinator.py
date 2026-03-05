import json
import pytest
from coordinator import ExperimentLog, generate_round1_configs, generate_next_configs


def test_experiment_log_empty():
    log = ExperimentLog()
    assert log.num_experiments() == 0
    assert log.best_result() is None


def test_experiment_log_tracks_best():
    log = ExperimentLog()
    log.add_result({"experiment_id": "exp_001", "val_loss": 0.5, "val_cosine_sim": 0.7})
    log.add_result({"experiment_id": "exp_002", "val_loss": 0.3, "val_cosine_sim": 0.85})
    log.add_result({"experiment_id": "exp_003", "val_loss": 0.4, "val_cosine_sim": 0.75})
    best = log.best_result()
    assert best["experiment_id"] == "exp_002"


def test_round1_generates_three_configs():
    configs = generate_round1_configs()
    assert len(configs) == 3
    types = {c.architecture.type for c in configs}
    assert types == {"linear", "mlp", "residual"}


def test_generate_next_configs_varies_knobs():
    log = ExperimentLog()
    log.add_result({
        "experiment_id": "exp_001",
        "val_loss": 0.5,
        "val_cosine_sim": 0.7,
        "config": {
            "experiment_id": "exp_001",
            "architecture": {"type": "mlp", "hidden_dim": 512, "num_blocks": 0, "dropout": 0.1, "activation": "gelu"},
            "training": {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.05, "batch_size": 64, "max_epochs": 20, "early_stop_patience": 5, "grad_clip": 1.0},
            "loss": {"type": "combined", "contrastive_weight": 0.7, "cosine_weight": 0.3, "temperature": 0.07},
            "data": {"subset_size": 50, "val_fraction": 0.15, "clip_sample_frames": 8},
            "vjepa_dim": 1024, "clip_dim": 768,
        },
    })
    next_configs = generate_next_configs(log, num_configs=3, seed=42)
    assert len(next_configs) == 3
    for c in next_configs:
        assert c.experiment_id != "exp_001"


def test_experiment_log_save_load(tmp_path):
    log = ExperimentLog()
    log.add_result({"experiment_id": "exp_001", "val_loss": 0.5})
    path = str(tmp_path / "log.json")
    log.save(path)
    loaded = ExperimentLog.load(path)
    assert loaded.num_experiments() == 1


def test_convergence_detection():
    log = ExperimentLog()
    log.add_result({"experiment_id": "exp_001", "val_loss": 0.5})
    log.add_result({"experiment_id": "exp_002", "val_loss": 0.3})  # best
    log.add_result({"experiment_id": "exp_003", "val_loss": 0.4})
    log.add_result({"experiment_id": "exp_004", "val_loss": 0.35})
    log.add_result({"experiment_id": "exp_005", "val_loss": 0.32})
    assert not log.is_converged(patience_rounds=3)
    # Add 3 more that don't beat 0.3
    log.add_result({"experiment_id": "exp_006", "val_loss": 0.31})
    log.add_result({"experiment_id": "exp_007", "val_loss": 0.35})
    log.add_result({"experiment_id": "exp_008", "val_loss": 0.33})
    assert log.is_converged(patience_rounds=3)
