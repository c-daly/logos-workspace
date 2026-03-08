import torch
import pytest
from losses import build_loss_fn
from config import LossConfig


def test_mse_loss_returns_scalar():
    cfg = LossConfig(type="mse")
    loss_fn = build_loss_fn(cfg)
    result = loss_fn(torch.randn(8, 768), torch.randn(8, 768))
    assert result["loss"].shape == ()


def test_cosine_loss_identical_is_zero():
    cfg = LossConfig(type="cosine")
    loss_fn = build_loss_fn(cfg)
    x = torch.nn.functional.normalize(torch.randn(8, 768), dim=-1)
    result = loss_fn(x, x)
    assert result["loss"].item() < 0.01


def test_contrastive_loss_returns_accuracy():
    cfg = LossConfig(type="contrastive", temperature=0.07)
    loss_fn = build_loss_fn(cfg)
    result = loss_fn(torch.randn(8, 768), torch.randn(8, 768))
    assert "loss" in result
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_combined_loss_uses_weights():
    cfg = LossConfig(type="combined", contrastive_weight=0.7, cosine_weight=0.3)
    loss_fn = build_loss_fn(cfg)
    result = loss_fn(torch.randn(8, 768), torch.randn(8, 768))
    assert "loss" in result
    assert "contrastive_loss" in result
    assert "cosine_loss" in result


def test_unknown_loss_raises():
    cfg = LossConfig(type="triplet")
    with pytest.raises(ValueError):
        build_loss_fn(cfg)
