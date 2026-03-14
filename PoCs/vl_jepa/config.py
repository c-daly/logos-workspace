"""Experiment configuration schema and defaults."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class ArchitectureConfig:
    type: Literal["linear", "mlp", "residual", "transformer"] = "linear"
    hidden_dim: int = 512
    num_blocks: int = 0
    num_layers: int = 1
    num_heads: int = 8
    dropout: float = 0.1
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    use_layer_norm: bool = True
    stages: list | None = None  # if set, builds a pipeline; top-level type/etc. ignored


VALID_FUNCTIONS = {"mse", "cosine", "contrastive", "infonce"}
VALID_TARGETS = {"clip_image", "clip_text_mean", "clip_text_first"}

@dataclass
class LossTerm:
    function: str = "mse"  # replaces Literal["mse","cosine","contrastive"]
    target: str = "clip_image"
    weight: float = 1.0
    temperature: float = 0.07
    label_smoothing: float = 0.0

    def __post_init__(self):
        if self.function not in VALID_FUNCTIONS:
            raise ValueError(f"Unknown loss function {self.function!r}. Valid: {VALID_FUNCTIONS}")
        if self.target not in VALID_TARGETS:
            raise ValueError(f"Unknown target {self.target!r}. Valid: {VALID_TARGETS}")


@dataclass
class LossConfig:
    terms: list = field(default_factory=lambda: [
        {"function": "mse", "target": "clip_image", "weight": 1.0,
         "temperature": 0.07, "label_smoothing": 0.0},
    ])
    warmup_terms: list = field(default_factory=list)
    warmup_epochs: int = 0


@dataclass
class TrainingConfig:
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = 3e-4
    lr_min: float = 1e-6
    lr_schedule: Literal["cosine", "step", "none"] = "cosine"
    warmup_epochs: int = 10
    cooldown_epochs: int = 20
    cooldown_lr: float = 1e-6
    weight_decay: float = 0.05
    batch_size: int = 256
    max_epochs: int = 200
    early_stop_patience: int = 10
    grad_clip: float = 1.0

    def __post_init__(self):
        if self.batch_size > 512:
            self.batch_size = 512


@dataclass
class DataConfig:
    noise_std: float = 0.0
    embedding_dropout: float = 0.0
    val_fraction: float = 0.15
    num_tokens: int | None = None  # tokens/frames to use at training time; None = all stored


@dataclass
class ExperimentConfig:
    experiment_id: str = "exp_001"
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vjepa_dim: int = 1024
    clip_dim: int = 768

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        fields = ArchitectureConfig.__dataclass_fields__
        return cls(
            experiment_id=d.get("experiment_id", "exp_001"),
            architecture=ArchitectureConfig(
                **{k: v for k, v in d.get("architecture", {}).items() if k in fields}
            ),
            training=TrainingConfig(
                **{k: v for k, v in d.get("training", {}).items()
                   if k in TrainingConfig.__dataclass_fields__}
            ),
            loss=LossConfig(
                **{k: v for k, v in d.get("loss", {}).items()
                   if k in LossConfig.__dataclass_fields__}
            ),
            data=DataConfig(
                **{k: v for k, v in d.get("data", {}).items()
                   if k in DataConfig.__dataclass_fields__}
            ),
            vjepa_dim=d.get("vjepa_dim", 1024),
            clip_dim=d.get("clip_dim", 768),
        )

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
