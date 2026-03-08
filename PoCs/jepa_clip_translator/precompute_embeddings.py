"""Embedding pre-computation module for the JEPA-CLIP translator PoC.

Pre-computes V-JEPA and CLIP embeddings for MSR-VTT videos and stores
them in HDF5 format for efficient training.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any
import pathlib
import h5py
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _get_token_dims(model: Any, total_tokens: int) -> tuple[int, int]:
    """Infer (T_tokens, S_tokens) from V-JEPA model config.

    Returns the number of temporal and spatial patch tokens. Handles the
    optional CLS token by checking whether total_tokens == T*S or T*S+1.
    """
    cfg = model.config
    num_frames = getattr(cfg, "num_frames", None) or getattr(cfg, "frames_per_clip", 64)
    tubelet_size = getattr(cfg, "tubelet_size", 2)
    image_size = getattr(cfg, "image_size", 256)
    patch_size = getattr(cfg, "patch_size", 16)
    if isinstance(image_size, (list, tuple)):
        image_size = image_size[-1]
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[-1]
    t_tok = num_frames // tubelet_size
    s_tok = (image_size // patch_size) ** 2
    expected = t_tok * s_tok
    if total_tokens in (expected, expected + 1):
        return t_tok, s_tok
    # Fallback: divide by t_tok.
    logger.warning(
        "Token count %d doesn't match config-derived T=%d S=%d (expected %d). "
        "Falling back to s_tok = total // t_tok.",
        total_tokens, t_tok, s_tok, expected,
    )
    return t_tok, total_tokens // t_tok


class EmbeddingPrecomputer:
    """Pre-compute and manage V-JEPA and CLIP embeddings.

    Parameters
    ----------
    device:
        Torch device string (e.g. ``"cpu"``, ``"cuda"``).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.vjepa_model: Any = None
        self.vjepa_processor: Any = None
        self.clip_model: Any = None
        self.clip_processor: Any = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load V-JEPA and CLIP models (frozen, eval, float32)."""
        from transformers import AutoModel, AutoVideoProcessor, CLIPModel, CLIPProcessor

        logger.info("Loading V-JEPA model (facebook/vjepa2-vitl-fpc64-256) ...")
        self.vjepa_processor = AutoVideoProcessor.from_pretrained(
            "facebook/vjepa2-vitl-fpc64-256"
        )
        self.vjepa_model = AutoModel.from_pretrained(
            "facebook/vjepa2-vitl-fpc64-256"
        )
        self.vjepa_model.eval()
        self.vjepa_model.to(self.device)
        for param in self.vjepa_model.parameters():
            param.requires_grad = False

        logger.info("Loading CLIP model (openai/clip-vit-large-patch14) ...")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_model.eval()
        self.clip_model.to(self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        logger.info("All models loaded (frozen, eval, float32).")

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_vjepa_embedding(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Compute V-JEPA temporal token embeddings.

        Parameters
        ----------
        frames:
            List of RGB numpy arrays, each ``(H, W, 3)`` uint8.

        Returns
        -------
        A ``(T, 1024)`` tensor — T temporal tokens, each spatially mean-pooled
        over the spatial patch dimension.
        """
        inputs = self.vjepa_processor(frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.vjepa_model(**inputs)
        hidden = outputs.last_hidden_state.squeeze(0)  # (N_tok, 1024)
        t_tok, s_tok = _get_token_dims(self.vjepa_model, hidden.shape[0])
        # Strip CLS token if present.
        if hidden.shape[0] == t_tok * s_tok + 1:
            hidden = hidden[1:]
        hidden = hidden.view(t_tok, s_tok, hidden.shape[-1])  # (T, S, 1024)
        return hidden.mean(dim=1).cpu()  # (T, 1024)

    @torch.no_grad()
    def compute_clip_image_embedding(
        self, frames: list[np.ndarray], sample_frames: int = 64
    ) -> torch.Tensor:
        """Compute per-frame CLIP image embeddings.

        Parameters
        ----------
        frames:
            List of RGB numpy arrays.
        sample_frames:
            Number of frames to uniformly sample.  Always returns exactly this
            many embeddings — short videos will have repeated frames to ensure
            a uniform output shape across all videos.

        Returns
        -------
        A ``(sample_frames, 768)`` L2-normalised tensor (one row per frame).
        """
        n = len(frames)
        indices = np.linspace(0, n - 1, sample_frames, dtype=int)
        sampled = [frames[i] for i in indices]

        embeddings = []
        for frame in sampled:
            inputs = self.clip_processor(images=frame, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            raw = self.clip_model.get_image_features(**inputs)
            # Older transformers versions return a tensor; newer may return a
            # ModelOutput with .pooler_output.
            feat = raw.pooler_output if hasattr(raw, "pooler_output") else raw
            embeddings.append(F.normalize(feat.squeeze(0), p=2, dim=-1))

        return torch.stack(embeddings).cpu()  # (sample_frames, 768)

    @torch.no_grad()
    def compute_clip_text_embeddings(
        self, captions: list[str]
    ) -> torch.Tensor:
        """Compute CLIP text embeddings for a list of captions.

        Parameters
        ----------
        captions:
            List of text captions.

        Returns
        -------
        A ``(num_captions, 768)`` L2-normalised tensor.
        """
        inputs = self.clip_processor(
            text=captions, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        raw = self.clip_model.get_text_features(**inputs)
        feat = raw.pooler_output if hasattr(raw, "pooler_output") else raw
        return F.normalize(feat, p=2, dim=-1).cpu()

    # ------------------------------------------------------------------
    # HDF5 persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_embeddings(
        path: str,
        video_ids: list[str],
        jepa_embs: torch.Tensor,
        clip_image_embs: torch.Tensor,
        clip_text_embs: torch.Tensor,
    ) -> None:
        """Save pre-computed embeddings to an HDF5 file.

        Parameters
        ----------
        path:
            Output ``.h5`` file path.
        video_ids:
            List of video identifier strings.
        jepa_embs:
            ``(N, T, 1024)`` V-JEPA temporal token embeddings.
        clip_image_embs:
            ``(N, F, 768)`` per-frame CLIP image embeddings.
        clip_text_embs:
            ``(N, C, 768)`` CLIP text embeddings (C captions per video).
        """
        with h5py.File(path, "w") as f:
            f.create_dataset("jepa_embeddings", data=jepa_embs.numpy())
            f.create_dataset("clip_image_embeddings", data=clip_image_embs.numpy())
            f.create_dataset("clip_text_embeddings", data=clip_text_embs.numpy())
            # Store video IDs as variable-length UTF-8 strings.
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("video_ids", data=video_ids, dtype=dt)

    @staticmethod
    def load_split(
        path: str, val_fraction: float = 0.15, seed: int = 42
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Load embeddings from HDF5 and split into train / val.

        Parameters
        ----------
        path:
            Path to the ``.h5`` file written by :meth:`save_embeddings`.
        val_fraction:
            Fraction of samples to hold out for validation.
        seed:
            Random seed for reproducible splitting.

        Returns
        -------
        train, val:
            Each is a dict with keys ``"jepa"``, ``"clip_image"``,
            ``"clip_text"`` mapping to tensors.
        """
        with h5py.File(path, "r") as f:
            jepa = torch.from_numpy(np.array(f["jepa_embeddings"]))
            clip_image = torch.from_numpy(
                np.array(f["clip_image_embeddings"])
            )  # (N, F, 768) per-frame
            clip_text = torch.from_numpy(np.array(f["clip_text_embeddings"]))

        n = jepa.shape[0]
        n_val = int(n * val_fraction)

        rng = np.random.RandomState(seed)
        perm = torch.from_numpy(rng.permutation(n))

        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train = {
            "jepa": jepa[train_idx],
            "clip_image": clip_image[train_idx],
            "clip_text": clip_text[train_idx],
        }
        val = {
            "jepa": jepa[val_idx],
            "clip_image": clip_image[val_idx],
            "clip_text": clip_text[val_idx],
        }
        return train, val


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main() -> None:
    """Orchestrate: load data -> load models -> compute embeddings -> save."""
    parser = argparse.ArgumentParser(
        description="Pre-compute V-JEPA and CLIP embeddings for MSR-VTT videos."
    )
    parser.add_argument(
        "--data",
        "--data-dir",
        dest="data_dir",
        type=str,
        default="data",
        help="Root directory for MSR-VTT data (default: data).  "
             "Expects <data>/videos/*.mp4 and <data>/annotations.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="msrvtt_embeddings.h5",
        help="Output HDF5 file path (default: msrvtt_embeddings.h5).",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of videos to process (default: all).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=64,
        help="Max frames to extract per video for V-JEPA (default: 64).",
    )
    parser.add_argument(
        "--clip-sample-frames",
        type=int,
        default=64,
        help="Frames to sample for CLIP image encoder; stored per-frame as "
             "(N, F, 768) (default: 64).",
    )
    parser.add_argument(
        "--max-captions",
        type=int,
        default=5,
        help="Maximum number of captions per video (default: 5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 1. Load data --------------------------------------------------------
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from data_loader import MSRVTTLoader

    loader = MSRVTTLoader(data_dir=args.data_dir, subset_size=args.subset_size or 0)
    ann_path = loader.ensure_data()
    videos, captions = loader.parse_annotations(str(ann_path))

    # Take the requested subset (None = all).
    if args.subset_size:
        videos = videos[: args.subset_size]
    video_ids = [v["video_id"] for v in videos]
    logger.info("Processing %d videos.", len(video_ids))

    # 2. Load models ------------------------------------------------------
    precomputer = EmbeddingPrecomputer(device=args.device)
    precomputer.load_models()

    # 3. Compute embeddings -----------------------------------------------
    jepa_list: list[torch.Tensor] = []
    clip_image_list: list[torch.Tensor] = []
    clip_text_list: list[torch.Tensor] = []
    valid_ids: list[str] = []

    for i, vid_id in enumerate(video_ids):
        frames = loader.load_video_frames(vid_id, max_frames=args.max_frames)
        if frames is None:
            logger.warning("Skipping %s — no video frames available.", vid_id)
            continue

        vid_captions = captions.get(vid_id, [])
        if not vid_captions:
            logger.warning("Skipping %s — no captions found.", vid_id)
            continue

        # Pad or truncate captions to a fixed count.
        vid_captions = vid_captions[: args.max_captions]
        while len(vid_captions) < args.max_captions:
            vid_captions.append(vid_captions[-1])

        logger.info(
            "[%d/%d] Computing embeddings for %s ...", i + 1, len(video_ids), vid_id
        )

        jepa_emb = precomputer.compute_vjepa_embedding(frames)
        clip_img_emb = precomputer.compute_clip_image_embedding(
            frames, sample_frames=args.clip_sample_frames
        )
        clip_txt_emb = precomputer.compute_clip_text_embeddings(vid_captions)

        jepa_list.append(jepa_emb)
        clip_image_list.append(clip_img_emb)
        clip_text_list.append(clip_txt_emb)
        valid_ids.append(vid_id)

    if not valid_ids:
        logger.error("No valid videos processed. Exiting.")
        return

    # 4. Save -------------------------------------------------------------
    jepa_embs = torch.stack(jepa_list)
    clip_image_embs = torch.stack(clip_image_list)
    clip_text_embs = torch.stack(clip_text_list)

    logger.info(
        "Saving %d embeddings to %s (JEPA %s, CLIP-img %s, CLIP-txt %s)",
        len(valid_ids),
        args.output,
        tuple(jepa_embs.shape),
        tuple(clip_image_embs.shape),
        tuple(clip_text_embs.shape),
    )
    EmbeddingPrecomputer.save_embeddings(
        args.output, valid_ids, jepa_embs, clip_image_embs, clip_text_embs
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
