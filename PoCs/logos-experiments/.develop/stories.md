# User Stories — vjepa_clip_alignment

## Story 1 — Shared Embedding Interface
As LOGOS, I want V-JEPA video embeddings projected into the CLIP text embedding space via a standard interface, so that video and text can be retrieved against each other in a unified space.

**Acceptance criteria:**
- `workspace/projector.py` exposes `load(checkpoint_path=None)` and `project(state, vjepa_embs)`
- Interface passes `--synthetic` smoke test on the eval script

## Story 2 — Top-5 Retrieval
As a researcher, I want the projector to achieve retrieval_accuracy_at_5 ≥ 0.80, so that the alignment is strong enough for downstream LOGOS use.

**Acceptance criteria:**
- `eval/test_alignment.py` reports `retrieval_accuracy_at_5 >= 0.80`

## Story 3 — Top-1 Retrieval
As a researcher, I want retrieval_accuracy_at_1 ≥ 0.50, so that top-1 text→video retrieval is meaningful.

**Acceptance criteria:**
- `eval/test_alignment.py` reports `retrieval_accuracy_at_1 >= 0.50`

## Story 4 — Embedding Alignment Quality
As a researcher, I want mean_cosine_similarity ≥ 0.82, so that projected video embeddings are tightly aligned with CLIP text embeddings in the shared space.

**Acceptance criteria:**
- `eval/test_alignment.py` reports `mean_cosine_similarity >= 0.82`

## Story 5 — Training Stability
As a researcher, I want training to be numerically stable with no NaN or Inf, so that results are reproducible.

**Acceptance criteria:**
- `training_stability == 1.0`
- Harness monitor confirms no NaN/Inf during training
