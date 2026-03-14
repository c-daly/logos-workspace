# Research Findings — vjepa_clip_alignment

## Projector Interface (from eval/test_alignment.py)

```python
def load(checkpoint_path: str = None) -> Any:
    # Initialize projector. Return state object passed to project().

def project(state: Any, vjepa_embeddings: np.ndarray) -> np.ndarray:
    # Input:  (N, 1024) numpy array — mean-pooled V-JEPA embeddings
    # Output: (N, 768)  numpy array — CLIP text space, L2-normalized
    # Must not contain NaN/Inf
```

## Validation Data Requirements

- Format: 2D `.npy` files
- Location: `experiments/vjepa_clip_alignment/data/val/`
- V-JEPA file: name must match `*vjepa*` or `*jepa*` or `*source*` or `*input*`
- CLIP file: name must match `*clip*` or `*target*` or `*output*`
- Both must have same N (number of samples)
- **Status: data dir does not exist yet** — must extract from HDF5

## HDF5 Source Data

Path: `~/projects/LOGOS/PoCs/jepa_clip_translator/msrvtt_embeddings.h5`
- `jepa`: (7010, 32, 1024) — mean-pool over token dim → (7010, 1024)
- `clip_text`: (7010, 5, 768) — mean over caption dim → (7010, 768) OR keep all 5
- `clip_image`: (7010, 64, 768) — DO NOT USE (dead end)

## Constraints (from constraints.yaml)

- Max 4h per run, 40h total GPU
- NaN early stop after 10 batches
- GPU target: 12GB (RunPod as fallback)
- Do NOT fine-tune V-JEPA or CLIP encoders
- Do NOT target CLIP image embeddings
- Do NOT use pure InfoNCE without warmup (collapse risk)
- Escalate if: no GPU, InfoNCE not decreasing after 20 epochs, R@5 < 0.30 after 3 runs

## Best Architecture (from 75 prior experiments)

**Residual translator — 4 blocks, hidden_dim=1024, GELU, LayerNorm, dropout=0.1**

```
Input (1024) → linear → 4× [linear → GELU → LayerNorm → residual add] → linear → L2 norm → Output (768)
```

## Best Hyperparameters

- lr=4.5e-5, AdamW, weight_decay=0.05
- grad_clip=0.4
- Cosine LR schedule with warmup_epochs=30
- batch_size=256 (may need reduction for 12GB)
- early_stopping patience=10

## Loss Schedule

1. Warmup (epochs 0–N): MSE loss only (stable)
2. Main (epoch N+): InfoNCE (temperature=0.07) + optional cosine term

## Checkpoint Format

```python
{
    "model_state_dict": model.state_dict(),
    "config": cfg,
    "epoch": epoch,
    "val_loss": float,
    "val_cosine_sim": float
}
```

## Environment

- venv: `~/projects/LOGOS/PoCs/jepa_clip_translator/.venv` (Python 3.13, torch, h5py, numpy, clip)
- Harness: pyyaml, numpy, pytest
- GPU: needs verification (handoff says ~12GB local; researcher could not confirm)

## Key Difference from Prior Work

All 75 prior experiments targeted `clip_image` embeddings → text R@1=0.074.
This experiment targets `clip_text` embeddings. InfoNCE against text anchors directly should close the modality gap.
