"""Projector interface for eval/test_alignment.py -- ResidualTranslator (v6)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return x + self.net(x)


class ResidualTranslator(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768, hidden_dim=1024, num_blocks=4, dropout=0.05):
        super().__init__()
        self.proj_in  = nn.Linear(input_dim, hidden_dim)
        self.blocks   = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.proj_out = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        if x.dim() == 3: x = x.mean(dim=1)
        return F.normalize(self.proj_out(self.blocks(self.proj_in(x))), dim=-1)


def load(checkpoint_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResidualTranslator()
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    return model.to(device).eval()


def project(state, vjepa_embs):
    model = state
    device = next(model.parameters()).device
    x = torch.from_numpy(vjepa_embs.astype(np.float32))
    if x.dim() == 3: x = x.mean(dim=1)
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
    result = out.cpu().numpy().astype(np.float32)
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        raise RuntimeError("project() produced NaN/Inf")
    return result
