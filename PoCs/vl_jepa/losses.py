"""Composable loss functions for translator training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config import LossConfig, LossTerm


def _mse(pred: torch.Tensor, target: torch.Tensor, **_) -> dict:
    return {"loss": F.mse_loss(pred, target)}


def _cosine(pred: torch.Tensor, target: torch.Tensor, **_) -> dict:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return {"loss": 1 - (pred_n * target_n).sum(dim=-1).mean()}


def _contrastive(
    pred: torch.Tensor, target: torch.Tensor,
    temperature: float = 0.07, label_smoothing: float = 0.0, **_
) -> dict:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    logits = pred_n @ target_n.T / temperature
    labels = torch.arange(len(pred), device=pred.device)
    loss = (
        F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        + F.cross_entropy(logits.T, labels, label_smoothing=label_smoothing)
    ) / 2
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return {"loss": loss, "accuracy": acc}


def _infonce(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
    label_smoothing: float = 0.0,
    batch_indices: list | None = None,
    **_,
) -> torch.Tensor:
    """InfoNCE with optional false-negative masking."""
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    B = pred.shape[0]
    sim = pred_n @ target_n.T / temperature
    if batch_indices is not None:
        vi = torch.tensor(batch_indices, device=pred.device)
        fn_mask = (vi.unsqueeze(1) == vi.unsqueeze(0)) & ~torch.eye(B, dtype=torch.bool, device=pred.device)
        sim = sim.masked_fill(fn_mask, -1e9)
    labels = torch.arange(B, device=pred.device)
    return 0.5 * (
        F.cross_entropy(sim, labels, label_smoothing=label_smoothing)
        + F.cross_entropy(sim.T, labels, label_smoothing=label_smoothing)
    )


_PRIMITIVES = {"mse": _mse, "cosine": _cosine, "contrastive": _contrastive, "infonce": _infonce}


def _resolve_target(
    name: str, clip_image: torch.Tensor, clip_text: torch.Tensor
) -> torch.Tensor:
    if name == "clip_image":
        return clip_image
    elif name == "clip_text_mean":
        return clip_text.mean(dim=1) if clip_text.dim() == 3 else clip_text
    elif name == "clip_text_first":
        return clip_text[:, 0, :] if clip_text.dim() == 3 else clip_text
    return clip_image


def compute_loss(
    pred: torch.Tensor,
    clip_image: torch.Tensor,
    clip_text: torch.Tensor,
    terms: list,
    batch_indices: list | None = None,
) -> dict:
    """Weighted sum of loss terms. Each term is a dict or LossTerm."""
    total = torch.tensor(0.0, device=pred.device)
    total_weight = 0.0
    acc = None
    for term in terms:
        if isinstance(term, dict):
            t = LossTerm(**{k: v for k, v in term.items() if k in LossTerm.__dataclass_fields__})
        else:
            t = term
        target = _resolve_target(t.target, clip_image, clip_text)
        # If pred has a token dimension that target lacks, pool pred first.
        # (e.g. clip_text targets are (B, 768) while pred may be (B, T, 768))
        effective_pred = pred.mean(dim=1) if pred.dim() > target.dim() else pred
        raw = _PRIMITIVES[t.function](
            effective_pred, target,
            temperature=t.temperature,
            label_smoothing=t.label_smoothing,
            batch_indices=batch_indices if t.function == "infonce" else None,
        )
        if isinstance(raw, dict):
            loss_val = raw["loss"]
            if "accuracy" in raw and acc is None:
                acc = raw["accuracy"]
        else:
            loss_val = raw
        total = total + t.weight * loss_val
        total_weight += t.weight
    out: dict = {"loss": total / max(total_weight, 1e-8)}
    if acc is not None:
        out["accuracy"] = acc
    return out
