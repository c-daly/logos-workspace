"""Loss functions for translator training."""

from typing import Callable

import torch
import torch.nn.functional as F

from config import LossConfig


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> dict:
    return {"loss": F.mse_loss(pred, target)}


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    loss = 1 - (pred_n * target_n).sum(dim=-1).mean()
    return {"loss": loss}


def contrastive_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> dict:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    logits = pred_n @ target_n.T / temperature
    labels = torch.arange(len(pred), device=pred.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return {"loss": loss, "accuracy": acc}


def combined_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07,
                  contrastive_weight: float = 0.7, cosine_weight: float = 0.3) -> dict:
    c = contrastive_loss(pred, target, temperature)
    cos = cosine_loss(pred, target)
    total = contrastive_weight * c["loss"] + cosine_weight * cos["loss"]
    return {
        "loss": total,
        "contrastive_loss": c["loss"].item(),
        "cosine_loss": cos["loss"].item(),
        "accuracy": c["accuracy"],
    }


def build_loss_fn(cfg: LossConfig) -> Callable:
    if cfg.type == "mse":
        return mse_loss
    elif cfg.type == "cosine":
        return cosine_loss
    elif cfg.type == "contrastive":
        return lambda p, t: contrastive_loss(p, t, cfg.temperature)
    elif cfg.type == "combined":
        return lambda p, t: combined_loss(p, t, cfg.temperature, cfg.contrastive_weight, cfg.cosine_weight)
    else:
        raise ValueError(f"Unknown loss type: {cfg.type}")
