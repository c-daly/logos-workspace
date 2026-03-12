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
