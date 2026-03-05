# JEPA-to-CLIP Translator: Agent Feedback Loop Design

**Date**: 2026-03-05
**Location**: `LOGOS/PoCs/jepa_clip_translator/`
**Status**: Design approved

## Goal

Build an agent feedback loop where a coordinator agent spawns parallel worker agents to iteratively search for a successful V-JEPA to CLIP embedding translator. Workers vary architecture, hyperparameters, and data strategy. The coordinator analyzes results and proposes new experiments until a quality threshold is met.

## Context

LOGOS uses JEPA models for grounded working memory (CWM-G) -- physical/sensory representations learned without text. To make CWM-G queryable from the language side, we need a translator that maps V-JEPA embeddings into CLIP's joint vision-language space, where text queries can find matching sensory representations.

### Prior Work

`~/cdaly/Downloads/jepa_to_clip_v2.ipynb` established a baseline:
- **V-JEPA**: `facebook/vjepa2-vitl-fpc64-256` (1024-dim)
- **CLIP**: `openai/clip-vit-large-patch14` (768-dim)
- **Translator**: 4 residual blocks, 1024->512->768, ~2M params, L2-normalized output
- **Loss**: 0.7 x contrastive (InfoNCE, t=0.07) + 0.3 x cosine
- **Training**: AdamW, lr=3e-4, cosine annealing, warmup 30 epochs, grad clip 1.0, AMP

## Dataset

**MSR-VTT**: 10K video clips, 200K text captions, ~6GB. Standard video-text retrieval benchmark.

For each video:
- V-JEPA: 64-frame windows, stride 32
- CLIP image encoder: 8 uniformly sampled frames per window
- CLIP text encoder: encode associated captions

Three embedding sets enable both reconstruction evaluation (JEPA->CLIP image) and the real task (JEPA->CLIP text retrieval).

## Architecture: Coordinator + Worker Agents (Approach B)

### Coordinator Agent
- Maintains experiment log (`log.json`)
- Analyzes results from completed experiments
- Proposes 2-4 new experiment configs per round
- Spawns worker agents in parallel
- Escalates complexity only when simpler approaches fail
- Decides when to stop (convergence or user flag)

### Worker Agents
- Stateless and disposable
- Receive config, train translator, evaluate, return metrics
- Each runs `train.py` with a specific config
- Returns: val loss, cosine similarity, retrieval accuracy

### Orchestration Flow

```
Coordinator spawns
    |
    +-- 1. Check: does MSR-VTT data exist locally?
    |      Yes -> use it
    |      No  -> download from source
    |
    +-- 2. Check: do pre-computed embeddings exist?
    |      No  -> run precompute_embeddings.py
    |      Yes -> skip
    |
    +-- 3. Round 1: spawn 3 workers in parallel
    |      Worker A: v2 baseline (4 residual blocks)
    |      Worker B: linear probe (1024->768)
    |      Worker C: 1-layer MLP (1024->512->768)
    |
    +-- 4. Collect results -> update log.json
    |      Coordinator analyzes and proposes next round
    |
    +-- 5. Round 2+: spawn 3-4 workers guided by prior results
    |
    +-- 6. Repeat until stopping criteria met
    |
    +-- 7. Final report: best config, metrics, checkpoint path
```

### Stopping Criteria
- Text retrieval R@5 exceeds target threshold
- Cosine similarity > 0.9 on validation set
- No improvement for 3 consecutive rounds
- Coordinator flags user for input

## Experiment Search Space

### Tier 1: Validate baseline on MSR-VTT (Round 1)
- v2 translator architecture, same hyperparams
- Linear probe (lower bound)
- 1-layer MLP (simplicity check)

### Tier 2: Knob turning (Round 2+, guided by Tier 1)

| Knob | Range |
|------|-------|
| Hidden dim | 256, 512, 768, 1024 |
| Num residual blocks | 0, 2, 4, 6 |
| Loss recipe | pure MSE, pure contrastive, pure cosine, blends |
| Contrastive temperature | 0.03, 0.07, 0.1, 0.2 |
| Learning rate | 1e-5 to 1e-3 |
| Dropout | 0.0, 0.1, 0.2, 0.3 |
| Batch size | 64, 128, 256, 512 |
| Data fraction | 10%, 25%, 50%, 100% |

## Evaluation Metrics

- **Cosine similarity**: translated JEPA vs true CLIP image embedding (reconstruction)
- **Text retrieval R@1/R@5/R@10**: translated JEPA vs CLIP text embeddings (the real goal)
- **Video retrieval R@1/R@5/R@10**: query with text, retrieve via translated embeddings
- MSR-VTT has published baselines for comparison

## Phased Rollout

### Phase A: Local CPU (prove the pipeline)
- 50-100 video subset of MSR-VTT
- Pre-compute embeddings once (slow on CPU, but one-time)
- Small translators: linear probe, 1-layer MLP
- Short training: 10-20 epochs
- Float32, no AMP
- Goal: every piece works end-to-end -- data fetch, precompute, training, eval, coordinator loop

### Phase B: Cloud GPU (real experiments)
- Full MSR-VTT (10K videos)
- Full search space including deep residual architectures
- Long training with early stopping (up to 500 epochs)
- AMP enabled
- Goal: find the best translator

## File Structure

```
LOGOS/PoCs/jepa_clip_translator/
    README.md
    precompute_embeddings.py    # V-JEPA + CLIP over MSR-VTT
    translator.py               # Model definitions (v2 baseline + variants)
    train.py                    # Worker training script
    evaluate.py                 # Metrics: loss, cosine sim, retrieval
    coordinator.py              # Coordinator logic
    config.py                   # Experiment config schema
    log.json                    # Experiment log (auto-generated)
    data/                       # MSR-VTT download (gitignored)
    embeddings/                 # Pre-computed embeddings (gitignored)
    checkpoints/                # Model checkpoints (gitignored)
```

## Future Integration

Once a winning translator is found, it gets packaged into Hermes as a `jepa_translator` module -- the bridge between CWM-G embeddings and the text-queryable CLIP space.
