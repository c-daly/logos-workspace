"""Tests for embedding pre-computation (HDF5 round-trip and splitting)."""

import os

import h5py
import numpy as np
import pytest
import torch

from precompute_embeddings import EmbeddingPrecomputer


def test_save_and_load_embeddings(tmp_path):
    """Test that embeddings round-trip through HDF5."""
    out_path = str(tmp_path / "test_embeddings.h5")
    jepa_embs = torch.randn(10, 1024)
    clip_image_embs = torch.randn(10, 768)
    clip_text_embs = torch.randn(10, 5, 768)
    video_ids = [f"video{i}" for i in range(10)]

    EmbeddingPrecomputer.save_embeddings(
        out_path, video_ids, jepa_embs, clip_image_embs, clip_text_embs
    )

    assert os.path.exists(out_path)
    with h5py.File(out_path, "r") as f:
        assert f["jepa_embeddings"].shape == (10, 1024)
        assert f["clip_image_embeddings"].shape == (10, 768)
        assert f["clip_text_embeddings"].shape == (10, 5, 768)
        loaded_ids = [s.decode() for s in f["video_ids"][:]]
        assert loaded_ids == video_ids


def test_load_split(tmp_path):
    """Test train/val splitting."""
    out_path = str(tmp_path / "test_embeddings.h5")
    n = 20
    EmbeddingPrecomputer.save_embeddings(
        out_path,
        [f"video{i}" for i in range(n)],
        torch.randn(n, 1024),
        torch.randn(n, 768),
        torch.randn(n, 5, 768),
    )
    train, val = EmbeddingPrecomputer.load_split(out_path, val_fraction=0.2, seed=42)
    assert train["jepa"].shape[0] + val["jepa"].shape[0] == n
    assert val["jepa"].shape[0] == 4  # 20 * 0.2
    assert train["clip_text"].shape[1] == 5  # captions dimension preserved


def test_load_split_deterministic(tmp_path):
    """Same seed produces same split."""
    out_path = str(tmp_path / "test_embeddings.h5")
    n = 20
    EmbeddingPrecomputer.save_embeddings(
        out_path,
        [f"video{i}" for i in range(n)],
        torch.randn(n, 1024),
        torch.randn(n, 768),
        torch.randn(n, 5, 768),
    )
    train1, val1 = EmbeddingPrecomputer.load_split(out_path, seed=42)
    train2, val2 = EmbeddingPrecomputer.load_split(out_path, seed=42)
    assert torch.equal(train1["jepa"], train2["jepa"])
    assert torch.equal(val1["jepa"], val2["jepa"])
