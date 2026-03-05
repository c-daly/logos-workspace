# JEPA-to-CLIP Translator PoC

Iterative agent-driven search for a V-JEPA -> CLIP embedding translator.

The translator maps V-JEPA's grounded sensory representations (1024-dim) into CLIP's joint vision-language space (768-dim), creating a unified embedding queryable by text, video, or other JEPA embeddings.

## Quick Start (CPU, Phase A)

```bash
# Install dependencies
cd PoCs/jepa_clip_translator
pip install -r requirements.txt

# 1. Pre-compute embeddings (slow on CPU, one-time)
#    First place MSR-VTT videos in data/videos/ or set MSRVTT_VIDEOS_URL
python precompute_embeddings.py --subset-size 50 --device cpu

# 2. Run a single experiment
python train.py --config configs/exp_001.json --embeddings embeddings/msrvtt_embeddings.h5 -v

# 3. Evaluate best checkpoint
python evaluate.py --config configs/exp_001.json \
    --checkpoint checkpoints/exp_001_best.pt \
    --embeddings embeddings/msrvtt_embeddings.h5

# 4. Run tests
.venv/bin/python -m pytest tests/ -v
```

## Agent Loop

A coordinator agent manages the experiment loop:

1. **Round 1**: Spawns 3 workers (linear, MLP, residual translator)
2. **Analysis**: Reads experiment log, identifies best config
3. **Round 2+**: Mutates best config (architecture, loss, hyperparams, data strategy)
4. **Repeats** until convergence or user stop

See `docs/plans/2026-03-05-jepa-clip-translator-design.md` for full design.

## Models

| Model | Embedding Dim | Source |
|-------|--------------|--------|
| V-JEPA | 1024 | `facebook/vjepa2-vitl-fpc64-256` |
| CLIP | 768 | `openai/clip-vit-large-patch14` |

## Modules

| File | Purpose |
|------|---------|
| `config.py` | Experiment config schema |
| `data_loader.py` | MSR-VTT download and loading |
| `precompute_embeddings.py` | V-JEPA + CLIP embedding extraction |
| `translator.py` | Model definitions (linear, MLP, residual) |
| `train.py` | Training with early stopping (`-v` for batch logging) |
| `evaluate.py` | Retrieval metrics (R@1, R@5, R@10) |
| `coordinator.py` | Experiment log and config generation |
| `losses.py` | Loss functions (MSE, cosine, contrastive, combined) |

## Phased Rollout

- **Phase A** (current): CPU proof of concept, 50-100 video subset, validate pipeline
- **Phase B**: Cloud GPU, full MSR-VTT, full search space
