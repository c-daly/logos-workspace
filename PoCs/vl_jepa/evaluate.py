"""Retrieval evaluation metrics for the JEPA-to-CLIP translator.

Computes Recall@K and median rank for image and text retrieval tasks,
using cosine similarity between translated JEPA embeddings and CLIP targets.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from config import ExperimentConfig
from translator import build_translator

logger = logging.getLogger(__name__)


def compute_retrieval_metrics(
    queries: torch.Tensor,
    targets: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute retrieval metrics between query and target embeddings.

    For each query, the correct target is assumed to be at the same index.

    Parameters
    ----------
    queries:
        ``(N, D)`` tensor of query embeddings.
    targets:
        ``(N, D)`` tensor of target embeddings.
    ks:
        Tuple of K values for Recall@K computation.

    Returns
    -------
    A dict with keys ``"R@{k}"`` for each k in *ks*, plus ``"median_rank"``.
    """
    # L2-normalize both sets of embeddings.
    queries_n = F.normalize(queries.float(), dim=-1)
    targets_n = F.normalize(targets.float(), dim=-1)

    # Cosine similarity matrix: (N, N).
    sim_matrix = queries_n @ targets_n.T

    # For each query, rank targets by similarity (descending).
    # ranks[i] = rank of the correct target (index i) for query i.
    # argsort descending gives indices of targets sorted by similarity.
    n = sim_matrix.shape[0]
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)

    # Find where the correct target (diagonal) appears in the ranking.
    correct_indices = torch.arange(n, device=sim_matrix.device).unsqueeze(1)
    ranks = (sorted_indices == correct_indices).nonzero(as_tuple=True)[1] + 1  # 1-indexed

    metrics: dict[str, float] = {}
    for k in ks:
        recall_at_k = (ranks <= k).float().mean().item()
        metrics[f"R@{k}"] = recall_at_k

    metrics["median_rank"] = ranks.float().median().item()

    return metrics


@torch.no_grad()
def evaluate_checkpoint(
    cfg: ExperimentConfig,
    embeddings_path: str,
    checkpoint_path: str,
) -> dict:
    """Evaluate a trained checkpoint on the validation set.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    embeddings_path:
        Path to the HDF5 file with pre-computed embeddings.
    checkpoint_path:
        Path to the model checkpoint ``.pt`` file.

    Returns
    -------
    A dict with keys ``"image_retrieval"`` (R@K metrics vs CLIP image),
    ``"text_retrieval"`` (R@K metrics vs CLIP text, first caption),
    ``"cosine_sim_mean"``, and ``"cosine_sim_std"``.
    """
    # 1. Load validation data.
    from precompute_embeddings import EmbeddingPrecomputer  # lazy import
    _, val_data = EmbeddingPrecomputer.load_split(
        embeddings_path, val_fraction=cfg.data.val_fraction
    )
    val_jepa = val_data["jepa"].float()
    val_clip_image = val_data["clip_image"].float()
    val_clip_text = val_data["clip_text"].float()

    logger.info("Validation set: %d samples", len(val_jepa))

    # 2. Load model from checkpoint.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_translator(cfg.architecture, cfg.vjepa_dim, cfg.clip_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt.get("best_state", ckpt))
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()

    # 3. Run validation JEPA embeddings through the translator.
    val_jepa = val_jepa.to(device)
    translated = model(val_jepa)

    # 4. Image retrieval metrics: translated vs clip_image.
    image_metrics = compute_retrieval_metrics(translated, val_clip_image)

    # 5. Text retrieval metrics: translated vs clip_text (first caption only).
    # clip_text shape: (N, C, 768) — take first caption.
    clip_text_first = val_clip_text[:, 0, :]
    text_metrics = compute_retrieval_metrics(translated, clip_text_first)

    # 6. Cosine similarity stats (translated vs clip_image).
    translated_n = F.normalize(translated, dim=-1)
    clip_image_n = F.normalize(val_clip_image, dim=-1)
    cosine_sims = (translated_n * clip_image_n).sum(dim=-1)

    return {
        "image_retrieval": image_metrics,
        "text_retrieval": text_metrics,
        "cosine_sim_mean": cosine_sims.mean().item(),
        "cosine_sim_std": cosine_sims.std().item(),
    }


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a JEPA-to-CLIP translator checkpoint."
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
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint .pt file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write evaluation results JSON.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config.
    cfg = ExperimentConfig.from_json(args.config)
    logger.info("Loaded config: %s", cfg.experiment_id)

    # Run evaluation.
    results = evaluate_checkpoint(cfg, args.embeddings, args.checkpoint)

    # Print summary.
    img = results["image_retrieval"]
    txt = results["text_retrieval"]
    logger.info(
        "Image retrieval: R@1=%.3f  R@5=%.3f  R@10=%.3f  median_rank=%.1f",
        img.get("R@1", 0),
        img.get("R@5", 0),
        img.get("R@10", 0),
        img.get("median_rank", 0),
    )
    logger.info(
        "Text retrieval:  R@1=%.3f  R@5=%.3f  R@10=%.3f  median_rank=%.1f",
        txt.get("R@1", 0),
        txt.get("R@5", 0),
        txt.get("R@10", 0),
        txt.get("median_rank", 0),
    )
    logger.info(
        "Cosine similarity: mean=%.4f  std=%.4f",
        results["cosine_sim_mean"],
        results["cosine_sim_std"],
    )

    # Optionally save results.
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
