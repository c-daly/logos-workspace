"""Experiment configuration schema and defaults."""

from dataclasses import dataclass, field, asdict
from typing import Literal
import json


@dataclass
class ArchitectureConfig:
    type: Literal["linear", "mlp", "residual"] = "linear"
    hidden_dim: int = 512
    num_blocks: int = 0  # only for residual
    dropout: float = 0.1
    activation: str = "gelu"


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.05
    batch_size: int = 64
    max_epochs: int = 20
    early_stop_patience: int = 5
    grad_clip: float = 1.0


@dataclass
class LossConfig:
    type: Literal["mse", "cosine", "contrastive", "combined"] = "combined"
    contrastive_weight: float = 0.7
    cosine_weight: float = 0.3
    temperature: float = 0.07


@dataclass
class DataConfig:
    subset_size: int = 50  # number of videos
    val_fraction: float = 0.15
    clip_sample_frames: int = 8


@dataclass
class ExperimentConfig:
    experiment_id: str = "exp_001"
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vjepa_dim: int = 1024
    clip_dim: int = 768

    def to_dict(self):
        return asdict(self)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            experiment_id=d.get("experiment_id", "exp_001"),
            architecture=ArchitectureConfig(**d.get("architecture", {})),
            training=TrainingConfig(**d.get("training", {})),
            loss=LossConfig(**d.get("loss", {})),
            data=DataConfig(**d.get("data", {})),
            vjepa_dim=d.get("vjepa_dim", 1024),
            clip_dim=d.get("clip_dim", 768),
        )
