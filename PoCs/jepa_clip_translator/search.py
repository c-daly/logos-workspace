"""Full experiment search loop: load embeddings, run rounds, save results.

Usage:
    python search.py --embeddings msrvtt_embeddings.h5
    python search.py --embeddings msrvtt_embeddings.h5 --rounds 5 --configs-per-round 3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig
from coordinator import LLMConfig, generate_next_configs, generate_round1_configs
from losses import compute_loss
from translator import build_translator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(
    path: str,
    device: torch.device,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    """Load embeddings from flat HDF5 and split into train/val/test."""
    with h5py.File(path, "r") as f:
        jepa = torch.tensor(np.array(f["jepa_embeddings"]), dtype=torch.float32)
        clip_image = torch.tensor(np.array(f["clip_image_embeddings"]), dtype=torch.float32)
        clip_text = torch.tensor(np.array(f["clip_text_embeddings"]), dtype=torch.float32)

    n = jepa.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    def _split(idx):
        return {
            "jepa": jepa[idx].to(device),
            "clip_image": clip_image[idx].to(device),
            "clip_text": clip_text[idx].to(device),
        }

    train, val, test = _split(train_idx), _split(val_idx), _split(test_idx)
    for name, data in (("train", train), ("val", val), ("test", test)):
        logger.info("%s: %d samples  jepa=%s  clip_image=%s",
                    name, data["jepa"].shape[0],
                    tuple(data["jepa"].shape[1:]),
                    tuple(data["clip_image"].shape[1:]))
    return train, val, test


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _get_lr(epoch: int, cfg: ExperimentConfig) -> float:
    t = cfg.training
    if epoch <= t.warmup_epochs:
        return t.lr_min + (t.lr - t.lr_min) * (epoch / max(t.warmup_epochs, 1))
    schedule_end = t.max_epochs - t.cooldown_epochs
    if epoch > schedule_end:
        return t.cooldown_lr
    if t.lr_schedule == "cosine":
        progress = (epoch - t.warmup_epochs) / max(schedule_end - t.warmup_epochs, 1)
        return t.lr_min + 0.5 * (t.lr - t.lr_min) * (1 + math.cos(math.pi * progress))
    elif t.lr_schedule == "step":
        step_size = max((schedule_end - t.warmup_epochs) // 3, 1)
        steps = (epoch - t.warmup_epochs) // step_size
        return max(t.lr * (0.5 ** steps), t.lr_min)
    return t.lr


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def _prepare_batch(
    batch_jepa: torch.Tensor,
    batch_clip_img: torch.Tensor,
    num_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align JEPA and CLIP tensors for the loss, handling both embedding formats.

    Legacy (mean-pooled JEPA): jepa is (B, 1024). CLIP frames are averaged
    down to (B, 768) so the task remains vector-to-vector.

    Token-level JEPA: jepa is (B, T, 1024). CLIP frames are subsampled to
    match, giving (B, K, 768). num_tokens controls K (None = use all T).
    """
    if batch_jepa.dim() == 3:
        T = batch_jepa.shape[1]
        K = min(num_tokens, T) if num_tokens else T
        if K < T:
            idx = torch.linspace(0, T - 1, K, dtype=torch.long, device=batch_jepa.device)
            batch_jepa = batch_jepa[:, idx, :]
            batch_clip_img = batch_clip_img[:, idx, :]
    else:
        # Legacy: average CLIP frames to match the single JEPA vector.
        F = batch_clip_img.shape[1] if batch_clip_img.dim() == 3 else None
        if F is not None:
            K = min(num_tokens, F) if num_tokens else F
            batch_clip_img = batch_clip_img[:, :K, :].mean(dim=1)
    return batch_jepa, batch_clip_img


def _augment(x: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    if cfg.data.noise_std > 0:
        x = x + torch.randn_like(x) * cfg.data.noise_std
    if cfg.data.embedding_dropout > 0:
        mask = torch.bernoulli(torch.full_like(x, 1.0 - cfg.data.embedding_dropout))
        x = x * mask / max(1.0 - cfg.data.embedding_dropout, 1e-8)
    return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: ExperimentConfig,
    train_data: dict,
    val_data: dict,
    device: torch.device,
) -> dict:
    train_loader = DataLoader(
        TensorDataset(train_data["jepa"], train_data["clip_image"], train_data["clip_text"]),
        batch_size=cfg.training.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_data["jepa"], val_data["clip_image"], val_data["clip_text"]),
        batch_size=cfg.training.batch_size, shuffle=False,
    )

    model = build_translator(cfg.architecture, cfg.vjepa_dim, cfg.clip_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt_cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}[cfg.training.optimizer]
    opt_kwargs: dict = {"lr": cfg.training.lr, "weight_decay": cfg.training.weight_decay}
    if cfg.training.optimizer == "sgd":
        opt_kwargs["momentum"] = 0.9
    optimizer = opt_cls(model.parameters(), **opt_kwargs)

    loss_terms = cfg.loss.terms
    warmup_terms = cfg.loss.warmup_terms if cfg.loss.warmup_terms else loss_terms

    term_str = " + ".join(
        f"{t.get('weight', 1):.1f}*{t['function']}->{t['target']}" for t in loss_terms
    )
    logger.info(
        "\n%s\nExperiment: %s\n  arch=%s hidden=%d blocks=%d layers=%d\n"
        "  loss=[%s]\n  opt=%s lr=%.2e schedule=%s bs=%d params=%s\n%s",
        "=" * 70, cfg.experiment_id,
        cfg.architecture.type, cfg.architecture.hidden_dim,
        cfg.architecture.num_blocks, cfg.architecture.num_layers,
        term_str, cfg.training.optimizer, cfg.training.lr,
        cfg.training.lr_schedule, cfg.training.batch_size, f"{num_params:,}",
        "=" * 70,
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, cfg.training.max_epochs + 1):
        current_lr = _get_lr(epoch, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        active_terms = warmup_terms if epoch <= cfg.loss.warmup_epochs else loss_terms

        model.train()
        train_loss_sum = 0.0
        for batch_jepa, batch_clip_img, batch_clip_txt in train_loader:
            batch_jepa, batch_clip_img = _prepare_batch(batch_jepa, batch_clip_img, cfg.data.num_tokens)
            batch_jepa = _augment(batch_jepa, cfg)
            optimizer.zero_grad()
            pred = model(batch_jepa)
            result = compute_loss(pred, batch_clip_img, batch_clip_txt, active_terms)
            result["loss"].backward()
            if cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            train_loss_sum += result["loss"].item()

        model.eval()
        val_loss_sum = val_cos_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_jepa, batch_clip_img, batch_clip_txt in val_loader:
                batch_jepa, batch_clip_img = _prepare_batch(batch_jepa, batch_clip_img, cfg.data.num_tokens)
                pred = model(batch_jepa)
                result = compute_loss(pred, batch_clip_img, batch_clip_txt, active_terms)
                val_loss_sum += result["loss"].item()
                val_cos_sum += (
                    F.normalize(pred, dim=-1) * F.normalize(batch_clip_img, dim=-1)
                ).sum(-1).mean().item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_cos = val_cos_sum / max(val_batches, 1)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(len(train_loader), 1),
            "val_loss": avg_val_loss,
            "val_cosine_sim": avg_val_cos,
            "lr": current_lr,
        })
        logger.info(
            "  Epoch %d/%d  val_loss=%.4f  cos=%.4f  lr=%.2e",
            epoch, cfg.training.max_epochs, avg_val_loss, avg_val_cos, current_lr,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= cfg.training.early_stop_patience:
            logger.info("  Early stopping at epoch %d.", epoch)
            break

    best_cos = history[best_epoch - 1]["val_cosine_sim"] if history else 0.0
    return {
        "experiment_id": cfg.experiment_id,
        "val_loss": best_val_loss,
        "val_cosine_sim": best_cos,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "config": cfg.to_dict(),
        "history": history,
        "best_state": best_state,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(queries: torch.Tensor, targets: torch.Tensor, ks=(1, 5, 10)) -> dict:
    queries_n = F.normalize(queries.float(), dim=-1)
    targets_n = F.normalize(targets.float(), dim=-1)
    sim = queries_n @ targets_n.T
    n = sim.shape[0]
    sorted_idx = sim.argsort(dim=1, descending=True)
    correct = torch.arange(n, device=sim.device).unsqueeze(1)
    ranks = (sorted_idx == correct).nonzero(as_tuple=True)[1] + 1
    return {**{f"R@{k}": (ranks <= k).float().mean().item() for k in ks},
            "median_rank": ranks.float().median().item()}


@torch.no_grad()
def evaluate_best(result: dict, test_data: dict, device: torch.device) -> dict:
    cfg = ExperimentConfig.from_dict(result["config"])
    model = build_translator(cfg.architecture, cfg.vjepa_dim, cfg.clip_dim).to(device)
    model.load_state_dict({k: v.to(device) for k, v in result["best_state"].items()})
    model.eval()

    translated = model(test_data["jepa"])

    # Reduce to (N, D) for retrieval — pool tokens if present.
    translated_pooled = translated.mean(dim=1) if translated.dim() == 3 else translated
    clip_image_pooled = (
        test_data["clip_image"].mean(dim=1)
        if test_data["clip_image"].dim() == 3
        else test_data["clip_image"]
    )

    img_metrics = compute_retrieval_metrics(translated_pooled, clip_image_pooled)
    txt_metrics = compute_retrieval_metrics(translated_pooled, test_data["clip_text"][:, 0, :])

    translated_n = F.normalize(translated_pooled, dim=-1)
    clip_n = F.normalize(clip_image_pooled, dim=-1)
    cos = (translated_n * clip_n).sum(dim=-1)

    return {
        "image_retrieval": img_metrics,
        "text_retrieval": txt_metrics,
        "cosine_sim_mean": cos.mean().item(),
        "cosine_sim_std": cos.std().item(),
    }


# ---------------------------------------------------------------------------
# Search loop
# ---------------------------------------------------------------------------

def run_search(
    embeddings_path: str,
    max_rounds: int = 5,
    configs_per_round: int = 5,
    convergence_patience: int | None = None,
    device: torch.device = torch.device("cpu"),
    output_path: str = "experiment_log.json",
    llm_config: LLMConfig | None = None,
) -> dict:
    train_data, val_data, test_data = load_embeddings(embeddings_path, device)

    all_results: list[dict] = []
    best_val_loss = float("inf")
    rounds_without_improvement = 0

    for round_num in range(1, max_rounds + 1):
        logger.info("\n%s\n# ROUND %d/%d\n%s", "#" * 70, round_num, max_rounds, "#" * 70)

        configs = generate_round1_configs() if round_num == 1 else generate_next_configs(
            all_results, num_configs=configs_per_round, llm_config=llm_config
        )
        logger.info("  Running %d experiments this round.", len(configs))

        round_best = float("inf")
        round_best_cos = 0.0
        for cfg in configs:
            result = run_experiment(cfg, train_data, val_data, device)
            all_results.append(result)
            if result["val_loss"] < round_best:
                round_best = result["val_loss"]
                round_best_cos = result["val_cosine_sim"]

        # Round summary
        logger.info("\n%s\nROUND %d SUMMARY:\n%s", "─" * 70, round_num, "─" * 70)
        for r in sorted(all_results[-len(configs):], key=lambda r: -r["val_cosine_sim"]):
            arch = r["config"]["architecture"]["type"]
            marker = " <-- best" if r["val_cosine_sim"] == round_best_cos else ""
            logger.info(
                "  %-25s %-8s cos=%.4f loss=%.4f ep=%d%s",
                r["experiment_id"], arch, r["val_cosine_sim"], r["val_loss"],
                r["epochs_trained"], marker,
            )

        if round_best < best_val_loss:
            best_val_loss = round_best
            rounds_without_improvement = 0
            overall_best = min(all_results, key=lambda r: r["val_loss"])
            logger.info(
                "  New best: %.4f (cos=%.4f) -- %s",
                best_val_loss, overall_best["val_cosine_sim"], overall_best["experiment_id"],
            )
        else:
            rounds_without_improvement += 1
            patience_str = f"{rounds_without_improvement}/{convergence_patience}" if convergence_patience else str(rounds_without_improvement)
            logger.info("  No improvement (patience %s).", patience_str)

        if convergence_patience is not None and rounds_without_improvement >= convergence_patience:
            logger.info("  Converged after %d rounds.", round_num)
            break

    # Final leaderboard
    logger.info("\n%s\nFINAL LEADERBOARD (%d experiments)\n%s", "=" * 90, len(all_results), "=" * 90)
    best_result = min(all_results, key=lambda r: r["val_loss"])
    for rank, r in enumerate(sorted(all_results, key=lambda r: -r["val_cosine_sim"]), 1):
        arch = r["config"]["architecture"]["type"]
        marker = " *" if r["experiment_id"] == best_result["experiment_id"] else ""
        logger.info(
            "%2d. %-25s %-8s cos=%.4f loss=%.4f ep=%d%s",
            rank, r["experiment_id"], arch, r["val_cosine_sim"], r["val_loss"],
            r["epochs_trained"], marker,
        )

    # Evaluate best on test set
    logger.info("\nEvaluating best model on test set: %s", best_result["experiment_id"])
    eval_results = evaluate_best(best_result, test_data, device)
    img = eval_results["image_retrieval"]
    txt = eval_results["text_retrieval"]
    logger.info(
        "Image retrieval: R@1=%.3f  R@5=%.3f  R@10=%.3f  median_rank=%.0f",
        img["R@1"], img["R@5"], img["R@10"], img["median_rank"],
    )
    logger.info(
        "Text retrieval:  R@1=%.3f  R@5=%.3f  R@10=%.3f  median_rank=%.0f",
        txt["R@1"], txt["R@5"], txt["R@10"], txt["median_rank"],
    )
    logger.info(
        "Cosine sim: mean=%.4f +/- %.4f",
        eval_results["cosine_sim_mean"], eval_results["cosine_sim_std"],
    )

    # Save checkpoint
    checkpoint_path = f"best_translator_{best_result['experiment_id']}.pt"
    torch.save({
        "model_state_dict": best_result["best_state"],
        "config": best_result["config"],
        "val_loss": best_result["val_loss"],
        "val_cosine_sim": best_result["val_cosine_sim"],
        "eval_results": eval_results,
    }, checkpoint_path)
    logger.info("Checkpoint saved: %s", checkpoint_path)

    # Save log
    log_data = {
        "experiments": [{k: v for k, v in r.items() if k != "best_state"} for r in all_results],
        "best_experiment_id": best_result["experiment_id"],
        "eval_results": eval_results,
    }
    with open(output_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info("Experiment log saved: %s", output_path)

    return log_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run JEPA-to-CLIP translator search.")
    parser.add_argument("--embeddings", required=True, help="Path to msrvtt_embeddings.h5")
    parser.add_argument("--rounds", type=int, default=3, help="Max search rounds (default: 3)")
    parser.add_argument("--configs-per-round", type=int, default=2, help="Configs per round after round 1 (default: 2)")
    parser.add_argument("--convergence-patience", type=int, default=None,
                        help="Stop early after N rounds without improvement (default: disabled)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="experiment_log.json")
    parser.add_argument("--llm-provider", default=None, help="LLM provider: openai (default) or anthropic")
    parser.add_argument("--llm-model", default=None, help="Model name (default: gpt-4o for openai, claude-opus-4-6 for anthropic)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(args.device)
    logger.info("Device: %s", device)

    llm_config = LLMConfig(
        provider=args.llm_provider or os.environ.get("COORDINATOR_LLM_PROVIDER", "openai"),
        model=args.llm_model or os.environ.get("COORDINATOR_LLM_MODEL", ""),
    )
    logger.info("LLM coordinator: %s", llm_config)

    run_search(
        embeddings_path=args.embeddings,
        max_rounds=args.rounds,
        configs_per_round=args.configs_per_round,
        convergence_patience=args.convergence_patience,
        device=device,
        output_path=args.output,
        llm_config=llm_config,
    )


if __name__ == "__main__":
    main()
