"""Tests for the training script."""

import os

import h5py
import pytest
import torch

from train import run_experiment
from config import ExperimentConfig, TrainingConfig, DataConfig


@pytest.fixture
def fake_embeddings(tmp_path):
    n = 40
    path = str(tmp_path / "test_embeddings.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("jepa_embeddings", data=torch.randn(n, 1024).numpy())
        f.create_dataset("clip_image_embeddings", data=torch.randn(n, 768).numpy())
        f.create_dataset("clip_text_embeddings", data=torch.randn(n, 5, 768).numpy())
        f.create_dataset("video_ids", data=[f"video{i}".encode() for i in range(n)])
    return path


def test_run_experiment_returns_metrics(fake_embeddings, tmp_path):
    cfg = ExperimentConfig(
        experiment_id="test_001",
        training=TrainingConfig(max_epochs=2, batch_size=8),
        data=DataConfig(val_fraction=0.2),
    )
    result = run_experiment(cfg, fake_embeddings, str(tmp_path / "checkpoints"))
    assert "val_loss" in result
    assert "val_cosine_sim" in result
    assert "epochs_trained" in result
    assert result["epochs_trained"] <= 2


def test_checkpoint_saved(fake_embeddings, tmp_path):
    cfg = ExperimentConfig(
        experiment_id="test_002",
        training=TrainingConfig(max_epochs=2, batch_size=8),
    )
    result = run_experiment(cfg, fake_embeddings, str(tmp_path / "checkpoints"))
    assert "best_checkpoint" in result
    assert os.path.exists(result["best_checkpoint"])


def test_early_stopping(fake_embeddings, tmp_path):
    cfg = ExperimentConfig(
        experiment_id="test_003",
        training=TrainingConfig(max_epochs=100, batch_size=8, early_stop_patience=2),
    )
    result = run_experiment(cfg, fake_embeddings, str(tmp_path / "checkpoints"))
    # Should stop well before 100 epochs
    assert result["epochs_trained"] < 100
