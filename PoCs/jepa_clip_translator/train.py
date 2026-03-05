"""Training script for the JEPA-to-CLIP translator.

Loads pre-computed embeddings, trains a translator model with early stopping,
saves checkpoints, and returns experiment metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig
from losses import build_loss_fn
from precompute_embeddings import EmbeddingPrecomputer
from translator import build_translator

logger = logging.getLogger(__name__)


def _compute_cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean cosine similarity between predicted and target embeddings."""
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return (pred_n * target_n).sum(dim=-1).mean().item()


def run_experiment(
    cfg: ExperimentConfig,
    embeddings_path: str,
    checkpoint_dir: str,
) -> dict:
    """Run a full training experiment.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    embeddings_path:
        Path to the HDF5 file with pre-computed embeddings.
    checkpoint_dir:
        Directory to save model checkpoints.

    Returns
    -------
    A dict with keys: experiment_id, val_loss, val_cosine_sim, epochs_trained,
    best_epoch, best_checkpoint, config, history.
    """
    # ------------------------------------------------------------------
    # 1. Load train/val data
    # ------------------------------------------------------------------
    train_data, val_data = EmbeddingPrecomputer.load_split(
        embeddings_path, val_fraction=cfg.data.val_fraction
    )

    train_jepa = train_data["jepa"].float()
    train_clip = train_data["clip_image"].float()
    val_jepa = val_data["jepa"].float()
    val_clip = val_data["clip_image"].float()

    logger.info(
        "Data loaded: %d train, %d val samples", len(train_jepa), len(val_jepa)
    )

    # ------------------------------------------------------------------
    # 2. Create DataLoaders
    # ------------------------------------------------------------------
    train_dataset = TensorDataset(train_jepa, train_clip)
    val_dataset = TensorDataset(val_jepa, val_clip)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    model = build_translator(cfg.architecture, cfg.vjepa_dim, cfg.clip_dim)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %s (%d trainable params)", cfg.architecture.type, num_params
    )

    # ------------------------------------------------------------------
    # 4. Build loss function
    # ------------------------------------------------------------------
    loss_fn = build_loss_fn(cfg.loss)

    # ------------------------------------------------------------------
    # 5. Build optimizer
    # ------------------------------------------------------------------
    opt_name = cfg.training.optimizer.lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    os.makedirs(checkpoint_dir, exist_ok=True)

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = ""
    patience_counter = 0

    for epoch in range(1, cfg.training.max_epochs + 1):
        # --- Train ---
        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        for batch_jepa, batch_clip in train_loader:
            optimizer.zero_grad()
            pred = model(batch_jepa)
            result = loss_fn(pred, batch_clip)
            loss = result["loss"]
            loss.backward()

            if cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.grad_clip
                )

            optimizer.step()
            train_loss_accum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_accum / max(train_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss_accum = 0.0
        val_cosine_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_jepa, batch_clip in val_loader:
                pred = model(batch_jepa)
                result = loss_fn(pred, batch_clip)
                val_loss_accum += result["loss"].item()
                val_cosine_accum += _compute_cosine_similarity(pred, batch_clip)
                val_batches += 1

        avg_val_loss = val_loss_accum / max(val_batches, 1)
        avg_val_cosine = val_cosine_accum / max(val_batches, 1)

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_cosine_sim": avg_val_cosine,
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_cosine_sim=%.4f",
            epoch,
            cfg.training.max_epochs,
            avg_train_loss,
            avg_val_loss,
            avg_val_cosine,
        )

        # --- Checkpoint best model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0

            best_checkpoint_path = os.path.join(
                checkpoint_dir, f"{cfg.experiment_id}_best.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_cosine_sim": avg_val_cosine,
                    "config": cfg.to_dict(),
                },
                best_checkpoint_path,
            )
        else:
            patience_counter += 1

        # --- Early stopping ---
        if patience_counter >= cfg.training.early_stop_patience:
            logger.info(
                "Early stopping at epoch %d (patience %d exhausted).",
                epoch,
                cfg.training.early_stop_patience,
            )
            break

    # ------------------------------------------------------------------
    # 7. Return results
    # ------------------------------------------------------------------
    epochs_trained = history[-1]["epoch"] if history else 0
    final_val_loss = history[-1]["val_loss"] if history else float("inf")
    final_val_cosine = history[-1]["val_cosine_sim"] if history else 0.0

    return {
        "experiment_id": cfg.experiment_id,
        "val_loss": final_val_loss,
        "val_cosine_sim": final_val_cosine,
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "best_checkpoint": best_checkpoint_path,
        "config": cfg.to_dict(),
        "history": history,
    }


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train a JEPA-to-CLIP translator model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config JSON file.",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to pre-computed embeddings HDF5 file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints (default: checkpoints).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write results JSON.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    cfg = ExperimentConfig.from_json(args.config)
    logger.info("Loaded config: %s", cfg.experiment_id)

    # Run experiment
    result = run_experiment(cfg, args.embeddings, args.checkpoint_dir)

    # Print summary
    logger.info(
        "Experiment %s complete: val_loss=%.4f, val_cosine_sim=%.4f, "
        "epochs=%d, best_epoch=%d",
        result["experiment_id"],
        result["val_loss"],
        result["val_cosine_sim"],
        result["epochs_trained"],
        result["best_epoch"],
    )

    # Optionally save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
