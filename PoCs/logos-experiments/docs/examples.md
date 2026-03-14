# Examples

The harness ships with two experiments that demonstrate different patterns.

## doc_search — Pytest Eval (start here)

A document search engine. The simplest possible experiment: no data to acquire, no GPU needed, no ML. Just read the ticket, read the tests, write code.

**Ticket** (`experiments/doc_search/goal.yaml`):
```yaml
objective: |
  Build a document search engine that indexes a small corpus of 
  programming topic documents and returns ranked results for text queries.
  
success_criteria:
  - metric: pass_rate
    threshold: 1.0
    primary: true
  - metric: tests_total
    threshold: 15

eval: eval/test_search.py
```

**Eval** (`experiments/doc_search/eval/test_search.py`): 15 pytest tests across three classes:

- `TestBasicSearch` — returns list, correct structure, scores descending, top_k limit, empty/no-match queries
- `TestRelevance` — title matching, keyword matching, partial matching, case insensitivity, multi-word queries
- `TestEdgeCases` — special characters, very long queries, numeric queries, positive scores

**Solution interface**: The tests expect `workspace/solution.py` with:
```python
def load(checkpoint_path=None) -> state
def search(state, query: str, top_k: int = 5) -> list[dict]
# Each result: {"id": str, "score": float, "title": str}
```

**Try it**:
```bash
cd logos-experiments

# See the current state
harness-eval --tests experiments/doc_search/eval/test_search.py
# → FAIL (no solution yet)

# Let an agent solve it
claude
# "Work on the doc_search experiment. Read goal.yaml and the eval, build a solution."
```

**What a solution looks like**: A TF-IDF search engine in ~80 lines. The agent needs to create a document corpus, build an inverted index, implement scored retrieval, and handle edge cases. Typically solved in one attempt.

## vjepa_clip_alignment — Metric Eval (research problem)

Map V-JEPA video embeddings into CLIP's vision-language space. A real ML research problem with multiple possible approaches, data bootstrapping, and measurable success criteria.

**Ticket** (`experiments/vjepa_clip_alignment/goal.yaml`):
```yaml
objective: |
  Build a mapping from V-JEPA's video representation space into CLIP's 
  vision-language embedding space.

success_criteria:
  - metric: mean_cosine_similarity
    threshold: 0.70
    primary: true
  - metric: retrieval_accuracy_at_5
    threshold: 0.60
  - metric: training_stability
    threshold: 1.0
```

**Eval** (`experiments/vjepa_clip_alignment/eval/test_alignment.py`): A custom metric eval that:

- Discovers validation data by scanning for `.npy` files matching naming patterns
- Discovers all dimensions from the data itself
- Supports `--synthetic` mode for interface testing
- Reports cosine similarity, retrieval accuracy, and stability
- Gives actionable errors when data is missing

**Solution interface**:
```python
def load(checkpoint_path=None) -> state
def project(state, vjepa_embeddings: np.ndarray) -> np.ndarray
# Input: (N, D_vjepa), Output: (N, D_clip)
```

**Constraints** (`experiments/vjepa_clip_alignment/constraints.yaml`): Five known failure modes from prior manual work — device placement, gradient checkpointing, NaN at batch 2, gradient explosion, tensor format mismatch.

**What makes this different from doc_search**:

1. **Data bootstrapping.** The agent needs to find video data, run both encoders, and generate paired embeddings before it can even evaluate.
2. **Multiple viable approaches.** Procrustes rotation, ridge regression, CCA, neural projection heads — the agent chooses.
3. **GPU required.** Encoding videos needs a GPU. The agent can use RunPod via MCP or a remote machine.
4. **Prior failure history.** The constraints file contains real bugs encountered during manual attempts.
5. **Multi-phase work.** Data prep, approach selection, implementation, evaluation — agent teams can parallelize this.

**Try it**:
```bash
# Test the eval interface with synthetic data
python experiments/vjepa_clip_alignment/eval/test_alignment.py \
    --projector <your_projector.py> --synthetic

# See what data exists
python experiments/vjepa_clip_alignment/eval/test_alignment.py \
    --projector <anything> --discover-only
```

## Creating Your Own

```bash
harness-new my_experiment --goal "Do the thing"
```

Then:
1. Edit `experiments/my_experiment/goal.yaml` — define the ticket
2. Write the eval in `experiments/my_experiment/eval/`
3. Optionally add `constraints.yaml` with known failure modes
4. Point Claude Code at the project

See [Writing Evals](writing-evals.md) for detailed guidance on the eval.
