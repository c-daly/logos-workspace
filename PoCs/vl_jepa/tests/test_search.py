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
