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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig
from coordinator import LLMConfig, generate_next_configs, generate_round1_configs
from losses import compute_loss
from translator import build_translator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log persistence helpers
# ---------------------------------------------------------------------------

def _save_log(all_results: list[dict], output_path: str) -> None:
    """Write experiment log to JSON, excluding non-serializable best_state."""
    log_data = {"experiments": [{k: v for k, v in r.items() if k != "best_state"}
                                 for r in all_results]}
    with open(output_path, "w") as f:
        json.dump(log_data, f, indent=2)


def _load_existing_log(path: str) -> list[dict]:
    """Load prior results from JSON. Returns [] if file missing or malformed."""
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f).get("experiments", [])
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Could not load existing log %s: %s", path, e)
        return []


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------

def _check_stopping(
    all_results: list[dict],
    round_num: int,
    max_rounds: int,
    max_experiments: int | None,
    target_metric: str | None,
    target_value: float | None,
    convergence_patience: int | None,
    rounds_without_improvement: int,
) -> tuple[bool, str]:
    """Return (should_stop, reason_string)."""
    if max_experiments is not None and len(all_results) >= max_experiments:
        return True, f"max_experiments={max_experiments} reached ({len(all_results)} run)"
    if target_metric and target_value is not None and all_results:
        best = max(r.get(target_metric, 0.0) for r in all_results)
        if best >= target_value:
            return True, f"target {target_metric}>={target_value:.4f} reached (best={best:.4f})"
    if convergence_patience is not None and rounds_without_improvement >= convergence_patience:
        return True, (
            f"convergence patience={convergence_patience} "
            f"({rounds_without_improvement} rounds no improvement)"
        )
    if round_num >= max_rounds:
        return True, f"max_rounds={max_rounds} reached"
    return False, ""



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
        # clip_image: (N,64,768) -> use CLS token (pos 0), no mean pooling
        if clip_image.dim() == 3:
            clip_image = clip_image[:, 0, :]
        # clip_text: (N,5,768) kept as-is; training loop samples one caption per batch

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
# Architecture description helper
# ---------------------------------------------------------------------------

def _arch_desc(arch_cfg) -> str:
    """Short human-readable description of an architecture config."""
    if arch_cfg.stages:
        parts = [f"{s['type']}({s.get('hidden_dim', '?')},{s.get('num_blocks', s.get('num_layers', '?'))})"
                 for s in arch_cfg.stages]
        return "pipeline[" + "→".join(parts) + "]"
    return arch_cfg.type


def _arch_desc_from_dict(arch_dict: dict) -> str:
    """Same but from a raw dict (used when reading saved results)."""
    if arch_dict.get("stages"):
        parts = [f"{s['type']}({s.get('hidden_dim', '?')},{s.get('num_blocks', s.get('num_layers', '?'))})"
                 for s in arch_dict["stages"]]
        return "pipeline[" + "→".join(parts) + "]"
    return arch_dict.get("type", "?")


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
        # clip_img is already (B,768) CLS token — no subsample needed
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
    if torch.cuda.device_count() > 1:
        logger.info("  Using DataParallel across %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)
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
        "\n%s\nExperiment: %s\n  arch=%s\n"
        "  loss=[%s]\n  opt=%s lr=%.2e schedule=%s bs=%d params=%s\n%s",
        "=" * 70, cfg.experiment_id,
        _arch_desc(cfg.architecture),
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
            # Sample one caption per batch (no mean pooling — same approach as v5 training)
            cap_idx = torch.randint(batch_clip_txt.shape[1], (1,)).item()
            batch_clip_txt = batch_clip_txt[:, cap_idx, :]
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
        _val_preds: list = []
        _val_clips: list = []
        with torch.no_grad():
            for batch_jepa, batch_clip_img, batch_clip_txt in val_loader:
                batch_jepa, batch_clip_img = _prepare_batch(batch_jepa, batch_clip_img, cfg.data.num_tokens)
                batch_clip_txt = batch_clip_txt[:, 0, :]
                pred = model(batch_jepa)
                result = compute_loss(pred, batch_clip_img, batch_clip_txt, active_terms)
                val_loss_sum += result["loss"].item()
                val_cos_sum += (
                    F.normalize(pred, dim=-1) * F.normalize(batch_clip_img, dim=-1)
                ).sum(-1).mean().item()
                val_batches += 1
                _val_preds.append(pred.cpu())
                _val_clips.append(batch_clip_img.cpu())

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_cos = val_cos_sum / max(val_batches, 1)
        # Compute R@1 / R@5 on full val set
        _p = F.normalize(torch.cat(_val_preds), dim=-1)
        _c = F.normalize(torch.cat(_val_clips), dim=-1)
        _sim = _p @ _c.T
        _lbl = torch.arange(len(_p))
        val_r1 = (_sim.argmax(1) == _lbl).float().mean().item()
        val_r5 = (_sim.topk(5, 1).indices == _lbl.unsqueeze(1)).any(1).float().mean().item()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss_sum / max(len(train_loader), 1),
            "val_loss": avg_val_loss,
            "val_cosine_sim": avg_val_cos,
            "lr": current_lr,
        })
        logger.info(
            "  Epoch %d/%d  val_loss=%.4f  cos=%.4f  R@1=%.3f  R@5=%.3f  lr=%.2e",
            epoch, cfg.training.max_epochs, avg_val_loss, avg_val_cos, val_r1, val_r5, current_lr,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            _m = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {k: v.cpu().clone() for k, v in _m.state_dict().items()}
            logger.info(
                "  *** best @ epoch %d  loss=%.4f  cos=%.4f  R@1=%.3f  R@5=%.3f",
                epoch, avg_val_loss, avg_val_cos, val_r1, val_r5,
            )
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
    max_experiments: int | None = None,
    target_metric: str | None = None,
    target_value: float | None = None,
    resume: bool = False,
    device: torch.device = torch.device("cpu"),
    output_path: str = "experiment_log.json",
    llm_config: LLMConfig | None = None,
) -> dict:
    train_data, val_data, test_data = load_embeddings(embeddings_path, device)

    all_results: list[dict] = _load_existing_log(output_path) if resume else []
    warm_start = resume and bool(all_results)
    if warm_start:
        logger.info("Resuming with %d existing results -- skipping round 1 baselines.", len(all_results))
    best_val_loss = float("inf")
    rounds_without_improvement = 0

    for round_num in range(1, max_rounds + 1):
        logger.info("\n%s\n# ROUND %d/%d\n%s", "#" * 70, round_num, max_rounds, "#" * 70)

        configs = (
            generate_round1_configs() if round_num == 1 and not warm_start
            else generate_next_configs(
                all_results, num_configs=configs_per_round, llm_config=llm_config,
                rounds_without_improvement=rounds_without_improvement,
            )
        )
        logger.info("  Running %d experiments this round.", len(configs))

        round_best = float("inf")
        round_best_cos = 0.0
        for cfg in configs:
            result = run_experiment(cfg, train_data, val_data, device)
            all_results.append(result)
            _save_log(all_results, output_path)
            if result["val_loss"] < round_best:
                round_best = result["val_loss"]
                round_best_cos = result["val_cosine_sim"]

        # Round summary
        logger.info("\n%s\nROUND %d SUMMARY:\n%s", "─" * 70, round_num, "─" * 70)
        for r in sorted(all_results[-len(configs):], key=lambda r: -r["val_cosine_sim"]):
            arch = _arch_desc_from_dict(r["config"]["architecture"])
            marker = " <-- best" if r["val_cosine_sim"] == round_best_cos else ""
            logger.info(
                "  %-25s %-40s cos=%.4f loss=%.4f ep=%d%s",
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

        should_stop, stop_reason = _check_stopping(
            all_results, round_num, max_rounds,
            max_experiments, target_metric, target_value,
            convergence_patience, rounds_without_improvement,
        )
        if should_stop:
            logger.info("  Stopping: %s", stop_reason)
            break

    # Final leaderboard
    logger.info("\n%s\nFINAL LEADERBOARD (%d experiments)\n%s", "=" * 90, len(all_results), "=" * 90)
    best_result = min(all_results, key=lambda r: r["val_loss"])
    for rank, r in enumerate(sorted(all_results, key=lambda r: -r["val_cosine_sim"]), 1):
        arch = _arch_desc_from_dict(r["config"]["architecture"])
        marker = " *" if r["experiment_id"] == best_result["experiment_id"] else ""
        logger.info(
            "%2d. %-25s %-40s cos=%.4f loss=%.4f ep=%d%s",
            rank, r["experiment_id"], arch, r["val_cosine_sim"], r["val_loss"],
            r["epochs_trained"], marker,
        )

    # Evaluate best on test set (only if best_state available -- skipped on pure-resume runs)
    eval_results: dict = {}
    if best_result.get("best_state") is not None:
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
    else:
        logger.warning(
            "best_state not available for %s (loaded from JSON log) -- skipping evaluate_best",
            best_result["experiment_id"],
        )

    # Save checkpoint
    checkpoint_path = f"best_translator_{best_result['experiment_id']}.pt"
    torch.save({
        "model_state_dict": best_result.get("best_state"),
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
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--target-metric", default=None, choices=["val_cosine_sim", "R@1", "R@5"])
    parser.add_argument("--target-value", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    import os as _os
    from datetime import datetime as _dt
    _log_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "run_logs")
    _os.makedirs(_log_dir, exist_ok=True)
    _run_log = _os.path.join(_log_dir, "run_" + _dt.now().strftime("%Y%m%d_%H%M%S") + ".log")
    _fh = logging.FileHandler(_run_log)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(_fh)
    logging.getLogger(__name__).info("Run log: %s", _run_log)

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
        max_experiments=args.max_experiments,
        target_metric=args.target_metric,
        target_value=args.target_value,
        resume=args.resume,
        device=device,
        output_path=args.output,
        llm_config=llm_config,
    )


if __name__ == "__main__":
    main()
