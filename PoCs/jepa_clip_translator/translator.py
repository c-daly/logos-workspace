"""Translator model definitions: V-JEPA (1024-dim) -> CLIP (768-dim)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ArchitectureConfig


def _activation(name: str) -> nn.Module:
    return {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[name]()


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, activation: str, use_layer_norm: bool):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), _activation(activation), nn.Dropout(dropout),
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
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int,
        num_layers: int, dropout: float, activation: str, use_layer_norm: bool,
    ):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(_activation(activation))
            layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ResidualTranslator(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int,
        num_blocks: int, dropout: float, activation: str, use_layer_norm: bool,
    ):
        super().__init__()
        proj = [nn.Linear(input_dim, hidden_dim)]
        if use_layer_norm:
            proj.append(nn.LayerNorm(hidden_dim))
        proj.extend([_activation(activation), nn.Dropout(dropout)])
        self.input_proj = nn.Sequential(*proj)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout, activation, use_layer_norm)
            for _ in range(num_blocks)
        ])
        out = []
        if use_layer_norm:
            out.append(nn.LayerNorm(hidden_dim))
        out.append(nn.Linear(hidden_dim, output_dim))
        self.output_proj = nn.Sequential(*out)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return F.normalize(self.output_proj(x), dim=-1)


def build_translator(cfg: ArchitectureConfig, input_dim: int, output_dim: int) -> nn.Module:
    if cfg.type == "linear":
        return LinearTranslator(input_dim, output_dim)
    elif cfg.type == "mlp":
        return MLPTranslator(
            input_dim, output_dim, cfg.hidden_dim, cfg.num_layers,
            cfg.dropout, cfg.activation, cfg.use_layer_norm,
        )
    elif cfg.type == "residual":
        return ResidualTranslator(
            input_dim, output_dim, cfg.hidden_dim, cfg.num_blocks,
            cfg.dropout, cfg.activation, cfg.use_layer_norm,
        )
    else:
        raise ValueError(f"Unknown translator type: {cfg.type}")
