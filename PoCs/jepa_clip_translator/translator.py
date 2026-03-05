"""Translator model definitions: V-JEPA (1024-dim) -> CLIP (768-dim)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ArchitectureConfig


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class LinearTranslator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(x), dim=-1)


class MLPTranslator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ResidualTranslator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_blocks: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return F.normalize(x, dim=-1)


def build_translator(cfg: ArchitectureConfig, input_dim: int, output_dim: int) -> nn.Module:
    if cfg.type == "linear":
        return LinearTranslator(input_dim, output_dim)
    elif cfg.type == "mlp":
        return MLPTranslator(input_dim, output_dim, cfg.hidden_dim, cfg.dropout)
    elif cfg.type == "residual":
        return ResidualTranslator(input_dim, output_dim, cfg.hidden_dim, cfg.num_blocks, cfg.dropout)
    else:
        raise ValueError(f"Unknown translator type: {cfg.type}")
