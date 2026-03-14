# Current State — vjepa_clip_alignment
Updated: 2026-03-10

## Direction Change

Original approach (frozen MLP projector) abandoned after 3 training runs all plateaued at cosine≈0.30-0.33 and R@5≈0.24.

Root cause confirmed via embedding analysis: frozen V-JEPA and CLIP text embeddings have **zero semantic alignment** — R@1=0% even with theoretically optimal linear projection. Fine-tuning the text encoder is required.

New approach: **implement VL-JEPA (arXiv 2512.10942) faithfully**.

## What's Being Built (in progress)

**impl-model-2 implementing:**

1. `workspace/model_vljepa.py`
   - `TransformerPredictor`: (B, 32, 1024) tokens → 4-layer TransformerEncoder → mean pool → Linear → (B, 768) L2-normed
   - `TextProjector`: CLIP encode_text + Linear(768,768) + L2-norm
   - save/load checkpoint

2. `workspace/train_vljepa.py`
   - Data: V-JEPA tokens from HDF5 (7010, 32, 1024) + raw MSR-VTT captions (140,200 sentences)
   - Two optimizer groups: predictor at base_lr, CLIP text at 0.05×base_lr
   - Loss: bi-directional InfoNCE only (no MSE warmup)
   - Checkpoint by val_R@5 (retrieval is the primary metric)
   - Default: 100 epochs, batch_size=64

3. `workspace/projector.py` (updated)
   - project() handles (N, 1024) or (N, 32, 1024) input
   - Uses TransformerPredictor only at inference (CLIP not needed)

## Key Decisions

- **Primary metric: R@1, R@5 (not cosine similarity)**
  Cosine sim can mislead when embedding space geometry differs between methods.
- **No MSE warmup** — text encoder fine-tuning provides alignment signal from epoch 1
- **Shared space: 768-dim** (same as CLIP text output, eval compatible)
- **CLIP ViT-L/14**: fine-tuned at 0.05× base LR (~123M text params, fits in 8.6GB)

## Previous Runs Summary

| Run | Approach | Best cosine | R@5 | R@1 |
|-----|----------|-------------|-----|-----|
| 1 | MLP, hard MSE→InfoNCE switch | 0.862 | 0.24 | 0.08 |
| 2 | MLP, smooth interpolation | -0.006 | — | — |
| 3 | MLP, flat pairs, alpha_max=0.7 | 0.331 | — | — |

All failed: zero semantic alignment in frozen feature spaces.

## What's Available

- Raw captions: `~/projects/LOGOS/PoCs/jepa_clip_translator/data/MSR-VTT/train_val_annotation/train_val_videodatainfo.json`
- V-JEPA tokens: HDF5 `jepa_embeddings` (7010, 32, 1024)
- CLIP: installed in venv, ViT-L/14 downloads to ~/.cache
- GPU: RTX 3070, 8.6GB

## After Implementation

When impl-model-2 reports:
1. Run: `python workspace/train_vljepa.py --epochs 100 --batch-size 64 --checkpoint best_vljepa.pt`
2. Generate new clip_val.npy using trained text encoder (needed for eval)
3. Run eval: `python eval/test_alignment.py --projector workspace/projector.py --checkpoint best_vljepa.pt`
4. Read [METRIC] lines directly (eval script PASS/FAIL is unreliable — see session-notes.md)
5. If R@5 ≥ 0.80 and R@1 ≥ 0.50: proceed to review
6. If not: diagnose and iterate
