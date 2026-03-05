"""End-to-end test: coordinator -> train -> evaluate pipeline with fake data."""

import json
import os

import h5py
import torch
import pytest

from config import ExperimentConfig, TrainingConfig, DataConfig
from coordinator import ExperimentLog, generate_round1_configs, generate_next_configs
from train import run_experiment
from evaluate import evaluate_checkpoint, compute_retrieval_metrics


@pytest.fixture
def fake_embeddings(tmp_path):
    """Create fake embeddings with learnable structure (not pure random)."""
    n = 40
    path = str(tmp_path / "embeddings.h5")

    torch.manual_seed(42)
    jepa = torch.randn(n, 1024)
    # CLIP image = linear transform of JEPA + small noise (learnable relationship)
    W = torch.randn(1024, 768) * 0.1
    clip_image = torch.nn.functional.normalize(jepa @ W + torch.randn(n, 768) * 0.01, dim=-1)
    # Text embeddings: similar to image but with more noise (5 captions each)
    clip_text = clip_image.unsqueeze(1).expand(-1, 5, -1) + torch.randn(n, 5, 768) * 0.1
    clip_text = torch.nn.functional.normalize(clip_text, dim=-1)

    with h5py.File(path, "w") as f:
        f.create_dataset("jepa_embeddings", data=jepa.numpy())
        f.create_dataset("clip_image_embeddings", data=clip_image.numpy())
        f.create_dataset("clip_text_embeddings", data=clip_text.numpy())
        f.create_dataset("video_ids", data=[f"video{i}".encode() for i in range(n)])
    return path


def test_full_pipeline(fake_embeddings, tmp_path):
    """Simulate a 2-round coordinator loop."""
    checkpoint_dir = str(tmp_path / "checkpoints")
    log_path = str(tmp_path / "log.json")
    log = ExperimentLog()

    # Round 1: initial configs
    round1 = generate_round1_configs()
    for cfg in round1:
        cfg.training.max_epochs = 3
        cfg.training.batch_size = 8
        cfg.data.val_fraction = 0.2
        result = run_experiment(cfg, fake_embeddings, checkpoint_dir)
        log.add_result(result)

    assert log.num_experiments() == 3
    best = log.best_result()
    assert best is not None
    assert "val_loss" in best

    # Round 2: generate next configs based on results
    round2 = generate_next_configs(log, num_configs=2, seed=42)
    assert len(round2) == 2
    for cfg in round2:
        cfg.training.max_epochs = 3
        cfg.training.batch_size = 8
        cfg.data.val_fraction = 0.2
        result = run_experiment(cfg, fake_embeddings, checkpoint_dir)
        log.add_result(result)

    assert log.num_experiments() == 5

    # Evaluate best model
    final_best = log.best_result()
    eval_cfg = ExperimentConfig.from_dict(final_best["config"])
    eval_cfg.data.val_fraction = 0.2
    eval_results = evaluate_checkpoint(eval_cfg, fake_embeddings, final_best["best_checkpoint"])

    assert "image_retrieval" in eval_results
    assert "text_retrieval" in eval_results
    assert "cosine_sim_mean" in eval_results
    assert eval_results["cosine_sim_mean"] > -1  # sanity: not garbage

    # Save log
    log.save(log_path)
    assert os.path.exists(log_path)

    # Verify summary is readable
    summary = log.summary()
    assert "5" in summary  # should mention 5 experiments somewhere
    print("\n" + summary)
    print(f"\nBest eval results: {json.dumps(eval_results, indent=2)}")
