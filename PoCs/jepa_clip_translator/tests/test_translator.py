import torch
import pytest
from translator import build_translator
from config import ArchitectureConfig


def test_linear_translator_shape():
    cfg = ArchitectureConfig(type="linear")
    model = build_translator(cfg, input_dim=1024, output_dim=768)
    out = model(torch.randn(4, 1024))
    assert out.shape == (4, 768)


def test_mlp_translator_shape():
    cfg = ArchitectureConfig(type="mlp", hidden_dim=256, dropout=0.0)
    model = build_translator(cfg, input_dim=1024, output_dim=768)
    out = model(torch.randn(4, 1024))
    assert out.shape == (4, 768)


def test_residual_translator_shape():
    cfg = ArchitectureConfig(type="residual", hidden_dim=256, num_blocks=2, dropout=0.0)
    model = build_translator(cfg, input_dim=1024, output_dim=768)
    out = model(torch.randn(4, 1024))
    assert out.shape == (4, 768)


def test_output_is_l2_normalized():
    cfg = ArchitectureConfig(type="mlp", hidden_dim=256)
    model = build_translator(cfg, input_dim=1024, output_dim=768)
    out = model(torch.randn(4, 1024))
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_transformer_translator_shape():
    cfg = ArchitectureConfig(type="transformer", hidden_dim=256, num_blocks=2, num_heads=8, dropout=0.0)
    model = build_translator(cfg, input_dim=1024, output_dim=768)
    # Token-level input: (B, T, 1024)
    out = model(torch.randn(4, 32, 1024))
    assert out.shape == (4, 32, 768)


def test_transformer_translator_legacy_shape():
    cfg = ArchitectureConfig(type="transformer", hidden_dim=256, num_blocks=1, num_heads=8, dropout=0.0)
    model = build_translator(cfg, input_dim=1024, output_dim=768)
    # Legacy 2D input: (B, 1024)
    out = model(torch.randn(4, 1024))
    assert out.shape == (4, 768)


def test_pipeline_translator_shape():
    cfg = ArchitectureConfig(stages=[
        {"type": "residual", "hidden_dim": 512, "num_blocks": 2, "dropout": 0.0, "activation": "gelu", "use_layer_norm": True},
        {"type": "mlp", "hidden_dim": 256, "num_layers": 2, "dropout": 0.0, "activation": "gelu", "use_layer_norm": True},
    ])
    model = build_translator(cfg, input_dim=1024, output_dim=768)
    out = model(torch.randn(4, 32, 1024))
    assert out.shape == (4, 32, 768)


def test_unknown_type_raises():
    cfg = ArchitectureConfig(type="transformer")
    cfg.type = "unknown"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        build_translator(cfg, input_dim=1024, output_dim=768)
