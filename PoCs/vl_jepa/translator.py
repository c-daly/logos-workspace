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
        if x.dim() == 3:
            x = x.mean(dim=1)
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
        if x.dim() == 3:
            x = x.mean(dim=1)
        return F.normalize(self.net(x), dim=-1)


class ResidualTranslator(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int,
        num_blocks: int, dropout: float = 0.1, activation: str = "gelu", use_layer_norm: bool = True,
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
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return F.normalize(self.output_proj(x), dim=-1)


class _TransformerStage(nn.Module):
    """Transformer encoder stage without output normalisation (used in pipelines)."""

    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: int,
        num_blocks: int, num_heads: int, dropout: float, use_layer_norm: bool,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            hidden_dim = max(num_heads, (hidden_dim // num_heads) * num_heads)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=use_layer_norm,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=max(num_blocks, 1), enable_nested_tensor=False,
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        if squeeze:
            x = x.squeeze(1)
        return x


def _build_stage(stage: dict, in_dim: int, out_dim: int) -> nn.Module:
    """Build one pipeline stage (no L2 norm). Dimensions: in_dim -> out_dim."""
    t = stage.get("type", "mlp")
    hidden_dim = stage.get("hidden_dim", 512)
    dropout = stage.get("dropout", 0.1)
    activation = stage.get("activation", "gelu")
    use_ln = stage.get("use_layer_norm", True)
    num_layers = stage.get("num_layers", 2)
    num_blocks = stage.get("num_blocks", 4)
    num_heads = stage.get("num_heads", 8)

    if t == "linear":
        return nn.Linear(in_dim, out_dim)

    elif t == "mlp":
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim)]
            if use_ln:
                layers += [nn.LayerNorm(hidden_dim)]
            layers += [_activation(activation), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        return nn.Sequential(*layers)

    elif t == "residual":
        proj: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if use_ln:
            proj += [nn.LayerNorm(hidden_dim)]
        proj += [_activation(activation), nn.Dropout(dropout)]
        blocks: list[nn.Module] = [
            ResidualBlock(hidden_dim, dropout, activation, use_ln)
            for _ in range(num_blocks)
        ]
        out: list[nn.Module] = []
        if use_ln:
            out += [nn.LayerNorm(hidden_dim)]
        out += [nn.Linear(hidden_dim, out_dim)]
        return nn.Sequential(*proj, *blocks, *out)

    elif t == "transformer":
        return _TransformerStage(in_dim, out_dim, hidden_dim, num_blocks, num_heads, dropout, use_ln)

    else:
        raise ValueError(f"Unknown stage type: {t!r}")


class PipelineTranslator(nn.Module):
    """Chain of heterogeneous stages. Dimensions flow automatically; L2 norm at the end."""

    def __init__(self, stages: list[dict], input_dim: int, output_dim: int):
        super().__init__()
        modules: list[nn.Module] = []
        in_dim = input_dim
        for i, stage in enumerate(stages):
            is_last = (i == len(stages) - 1)
            stage_out = output_dim if is_last else stage.get("hidden_dim", 512)
            modules.append(_build_stage(stage, in_dim, stage_out))
            in_dim = stage_out
        self.pipeline = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        for stage in self.pipeline:
            x = stage(x)
        return F.normalize(x, dim=-1)


class TransformerTranslator(nn.Module):
    """Self-attention over temporal tokens before projecting to output_dim."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int,
        num_blocks: int, num_heads: int, dropout: float, use_layer_norm: bool,
    ):
        super().__init__()
        self.stage = _TransformerStage(input_dim, output_dim, hidden_dim, num_blocks, num_heads, dropout, use_layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stage(x)
        # Pool over token dimension when input has 3 dims (batch, tokens, dim)
        if out.dim() == 3:
            out = out.mean(dim=1)
        return F.normalize(out, dim=-1)


def build_translator(cfg: ArchitectureConfig, input_dim: int, output_dim: int) -> nn.Module:
    if cfg.stages is not None:
        return PipelineTranslator(cfg.stages, input_dim, output_dim)
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
    elif cfg.type == "transformer":
        return TransformerTranslator(
            input_dim, output_dim, cfg.hidden_dim, cfg.num_blocks,
            cfg.num_heads, cfg.dropout, cfg.use_layer_norm,
        )
    else:
        raise ValueError(f"Unknown translator type: {cfg.type}")
