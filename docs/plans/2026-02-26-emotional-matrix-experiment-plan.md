# Emotional State Matrix — First Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether filtering embeddings through a matrix shaped by emotionally-charged experiences causes neutral inputs to drift toward that emotional character.

**Architecture:** Uses the logos experiment framework (AgentDefinition, ExperimentRunner). Five simple agents wired into a pipeline. OpenAI embeddings (1536-dim). Matrix initialized with gaussian noise. Naive column selection (highest absolute value). Cosine similarity to measure drift.

**Tech Stack:** Python 3.12, numpy, openai, logos_experiment, pytest

**Prerequisite:** The experiment framework plan (2026-02-26-experiment-framework-plan.md) must be completed first.

---

### Task 1: Embedding agent

**Files:**
- Create: `sophia/src/sophia/experiments/__init__.py`
- Create: `sophia/src/sophia/experiments/agents/__init__.py`
- Create: `sophia/src/sophia/experiments/agents/embedding.py`
- Test: `sophia/tests/unit/experiments/__init__.py`
- Test: `sophia/tests/unit/experiments/test_embedding_agent.py`

**Step 1: Write the failing test**

```python
# sophia/tests/unit/experiments/__init__.py
# (empty)

# sophia/tests/unit/experiments/test_embedding_agent.py
import numpy as np
from unittest.mock import patch, MagicMock
from sophia.experiments.agents.embedding import EmbeddingAgent, make_embedding_agent


def test_embedding_agent_returns_vector():
    """EmbeddingAgent.process takes text, returns numpy array."""
    fake_embedding = [0.1] * 1536
    agent = EmbeddingAgent(model="text-embedding-3-small", dim=1536)

    with patch.object(agent, "_client") as mock_client:
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=fake_embedding)]
        mock_client.embeddings.create.return_value = mock_response

        result = agent.process("hello world")

    assert isinstance(result, np.ndarray)
    assert result.shape == (1536,)


def test_embedding_agent_dim_configurable():
    agent = EmbeddingAgent(model="text-embedding-3-small", dim=768)
    assert agent.dim == 768


def test_factory_creates_agent():
    agent = make_embedding_agent({"model": "text-embedding-3-small", "dim": 1536})
    assert isinstance(agent, EmbeddingAgent)
    assert agent.dim == 1536
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cdaly/projects/LOGOS/sophia && poetry run pytest tests/unit/experiments/test_embedding_agent.py -v`
Expected: FAIL with "No module named 'sophia.experiments'"

**Step 3: Write minimal implementation**

```python
# sophia/src/sophia/experiments/__init__.py
# (empty)

# sophia/src/sophia/experiments/agents/__init__.py
# (empty)

# sophia/src/sophia/experiments/agents/embedding.py
import numpy as np
from openai import OpenAI


class EmbeddingAgent:
    """Takes text, returns embedding vector via OpenAI API."""

    def __init__(self, model: str, dim: int):
        self.model = model
        self.dim = dim
        self._client = OpenAI()

    def process(self, input_data: str) -> np.ndarray:
        response = self._client.embeddings.create(
            model=self.model,
            input=input_data,
            dimensions=self.dim,
        )
        return np.array(response.data[0].embedding, dtype=np.float64)


def make_embedding_agent(config: dict) -> EmbeddingAgent:
    return EmbeddingAgent(
        model=config.get("model", "text-embedding-3-small"),
        dim=config.get("dim", 1536),
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/cdaly/projects/LOGOS/sophia && poetry run pytest tests/unit/experiments/test_embedding_agent.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sophia/src/sophia/experiments/ sophia/tests/unit/experiments/
git commit -m "feat: add EmbeddingAgent for experiment framework"
```

---

### Task 2: Matrix agent (StatefulAgent)

**Files:**
- Create: `sophia/src/sophia/experiments/agents/matrix.py`
- Test: `sophia/tests/unit/experiments/test_matrix_agent.py`

**Step 1: Write the failing test**

```python
# sophia/tests/unit/experiments/test_matrix_agent.py
import numpy as np
from sophia.experiments.agents.matrix import MatrixAgent, make_matrix_agent


def test_matrix_init_gaussian():
    agent = MatrixAgent(dim=4, std=0.01, seed=42)
    state = agent.get_state()
    assert state.shape == (4, 4)
    assert np.abs(state.mean()) < 0.1  # close to zero mean
    assert state.std() > 0  # not all zeros


def test_matrix_init_reproducible():
    a = MatrixAgent(dim=4, std=0.01, seed=42)
    b = MatrixAgent(dim=4, std=0.01, seed=42)
    np.testing.assert_array_equal(a.get_state(), b.get_state())


def test_matrix_filter_identity_when_zeros():
    """Zero matrix = identity filter via (M + I) * v."""
    agent = MatrixAgent(dim=4, std=0.0, seed=0)
    v = np.array([1.0, 2.0, 3.0, 4.0])
    result = agent.process(v)
    np.testing.assert_array_almost_equal(result, v)


def test_matrix_filter_modifies_embedding():
    agent = MatrixAgent(dim=4, std=0.5, seed=42)
    v = np.array([1.0, 2.0, 3.0, 4.0])
    result = agent.process(v)
    assert not np.allclose(result, v)  # should be different


def test_matrix_snapshot_and_reset():
    agent = MatrixAgent(dim=4, std=0.01, seed=42)
    snap = agent.snapshot()
    agent._matrix += 999  # mutate
    agent.reset(snap)
    np.testing.assert_array_equal(agent.get_state(), snap)


def test_factory():
    agent = make_matrix_agent({"dim": 8, "std": 0.01, "seed": 99})
    assert agent.get_state().shape == (8, 8)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cdaly/projects/LOGOS/sophia && poetry run pytest tests/unit/experiments/test_matrix_agent.py -v`
Expected: FAIL with "No module named 'sophia.experiments.agents.matrix'"

**Step 3: Write minimal implementation**

```python
# sophia/src/sophia/experiments/agents/matrix.py
from __future__ import annotations

import numpy as np
from typing import Any


class MatrixAgent:
    """Emotional state matrix. Filters embeddings via (M + I) * embedding.

    StatefulAgent: supports get_state, snapshot, reset.
    """

    def __init__(self, dim: int, std: float = 0.01, seed: int = 0):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._matrix = self._rng.normal(0, std, (dim, dim))
        self._identity = np.eye(dim)

    def process(self, input_data: np.ndarray) -> np.ndarray:
        """Filter embedding through emotional state: (M + I) * embedding."""
        return (self._matrix + self._identity) @ input_data

    def get_state(self) -> np.ndarray:
        return self._matrix.copy()

    def reset(self, state: Any = None) -> None:
        if state is not None:
            self._matrix = state.copy()
        else:
            self._matrix = np.zeros((self.dim, self.dim))

    def snapshot(self) -> np.ndarray:
        return self._matrix.copy()


def make_matrix_agent(config: dict) -> MatrixAgent:
    return MatrixAgent(
        dim=config.get("dim", 1536),
        std=config.get("std", 0.01),
        seed=config.get("seed", 0),
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/cdaly/projects/LOGOS/sophia && poetry run pytest tests/unit/experiments/test_matrix_agent.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sophia/src/sophia/experiments/agents/matrix.py sophia/tests/unit/experiments/test_matrix_agent.py
git commit -m "feat: add MatrixAgent (emotional state filter)"
```

---

### Task 3: Matrix update agent

**Files:**
- Create: `sophia/src/sophia/experiments/agents/updater.py`
- Test: `sophia/tests/unit/experiments/test_updater_agent.py`

**Step 1: Write the failing test**

```python
# sophia/tests/unit/experiments/test_updater_agent.py
import numpy as np
from sophia.experiments.agents.matrix import MatrixAgent
from sophia.experiments.agents.updater import MatrixUpdateAgent, make_update_agent


def test_update_modifies_one_column():
    matrix_agent = MatrixAgent(dim=4, std=0.0, seed=0)  # zero matrix
    updater = MatrixUpdateAgent(matrix_agent=matrix_agent, alpha=0.1)

    embedding = np.array([0.1, 0.9, 0.3, 0.2])  # dim 1 has highest abs
    updater.process(embedding)

    state = matrix_agent.get_state()
    # Column 1 should be modified (highest abs value dimension)
    assert not np.allclose(state[:, 1], 0.0)
    # Other columns should be unchanged
    np.testing.assert_array_equal(state[:, 0], np.zeros(4))
    np.testing.assert_array_equal(state[:, 2], np.zeros(4))
    np.testing.assert_array_equal(state[:, 3], np.zeros(4))


def test_update_bounded_by_tanh():
    matrix_agent = MatrixAgent(dim=4, std=0.0, seed=0)
    updater = MatrixUpdateAgent(matrix_agent=matrix_agent, alpha=100.0)  # huge alpha

    embedding = np.array([0.1, 0.9, 0.3, 0.2])
    updater.process(embedding)

    state = matrix_agent.get_state()
    assert np.all(state >= -1.0)
    assert np.all(state <= 1.0)


def test_update_alpha_scales_magnitude():
    m1 = MatrixAgent(dim=4, std=0.0, seed=0)
    m2 = MatrixAgent(dim=4, std=0.0, seed=0)

    u1 = MatrixUpdateAgent(matrix_agent=m1, alpha=0.01)
    u2 = MatrixUpdateAgent(matrix_agent=m2, alpha=0.1)

    embedding = np.array([0.1, 0.9, 0.3, 0.2])
    u1.process(embedding)
    u2.process(embedding)

    # Larger alpha = bigger change
    assert np.abs(m2.get_state()).sum() > np.abs(m1.get_state()).sum()


def test_factory():
    matrix_agent = MatrixAgent(dim=4, std=0.0, seed=0)
    updater = make_update_agent({"alpha": 0.05}, matrix_agent=matrix_agent)
    assert updater.alpha == 0.05
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cdaly/projects/LOGOS/sophia && poetry run pytest tests/unit/experiments/test_updater_agent.py -v`
Expected: FAIL with "No module named 'sophia.experiments.agents.updater'"

**Step 3: Write minimal implementation**

```python
# sophia/src/sophia/experiments/agents/updater.py
from __future__ import annotations

import numpy as np

from sophia.experiments.agents.matrix import MatrixAgent


class MatrixUpdateAgent:
    """Updates the emotional state matrix using a persona entry embedding.

    Naive column selection: highest absolute value in the embedding.
    Update: column_j = tanh(column_j + alpha * embedding)
    """

    def __init__(self, matrix_agent: MatrixAgent, alpha: float = 0.01):
        self.matrix_agent = matrix_agent
        self.alpha = alpha

    def process(self, input_data: np.ndarray) -> np.ndarray:
        """Update the matrix and return the input unchanged."""
        col_idx = int(np.argmax(np.abs(input_data)))
        matrix = self.matrix_agent._matrix
        matrix[:, col_idx] = np.tanh(matrix[:, col_idx] + self.alpha * input_data)
        return input_data


def make_update_agent(config: dict, matrix_agent: MatrixAgent | None = None) -> MatrixUpdateAgent:
    if matrix_agent is None:
        raise ValueError("matrix_agent required")
    return MatrixUpdateAgent(
        matrix_agent=matrix_agent,
        alpha=config.get("alpha", 0.01),
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/cdaly/projects/LOGOS/sophia && poetry run pytest tests/unit/experiments/test_updater_agent.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sophia/src/sophia/experiments/agents/updater.py sophia/tests/unit/experiments/test_updater_agent.py
git commit -m "feat: add MatrixUpdateAgent (naive column selection + tanh)"
```

---

### Task 4: Similarity agent

**Files:**
- Create: `sophia/src/sophia/experiments/agents/similarity.py`
- Test: `sophia/tests/unit/experiments/test_similarity_agent.py`

**Step 1: Write the failing test**

```python
# sophia/tests/unit/experiments/test_similarity_agent.py
import numpy as np
from sophia.experiments.agents.similarity import SimilarityAgent, make_similarity_agent


def test_identical_vectors_similarity_1():
    agent = SimilarityAgent()
    v = np.array([1.0, 0.0, 0.0])
    result = agent.process({"a": v, "b": v})
    assert abs(result - 1.0) < 1e-6


def test_orthogonal_vectors_similarity_0():
    agent = SimilarityAgent()
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    result = agent.process({"a": a, "b": b})
    assert abs(result) < 1e-6


def test_opposite_vectors_similarity_neg1():
    agent = SimilarityAgent()
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    result = agent.process({"a": a, "b": b})
    assert abs(result - (-1.0)) < 1e-6


def test_factory():
    agent = make_similarity_agent({})
    assert isinstance(agent, SimilarityAgent)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cdaly/projects/LOGOS/sophia && poetry run pytest tests/unit/experiments/test_similarity_agent.py -v`
Expected: FAIL with "No module named 'sophia.experiments.agents.similarity'"

**Step 3: Write minimal implementation**

```python
# sophia/src/sophia/experiments/agents/similarity.py
import numpy as np


class SimilarityAgent:
    """Computes cosine similarity between two embedding vectors."""

    def process(self, input_data: dict) -> float:
        a = input_data["a"]
        b = input_data["b"]
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)


def make_similarity_agent(config: dict) -> SimilarityAgent:
    return SimilarityAgent()
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/cdaly/projects/LOGOS/sophia && poetry run pytest tests/unit/experiments/test_similarity_agent.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sophia/src/sophia/experiments/agents/similarity.py sophia/tests/unit/experiments/test_similarity_agent.py
git commit -m "feat: add SimilarityAgent (cosine similarity)"
```

---

### Task 5: First experiment — emotional drift test

**Files:**
- Create: `sophia/src/sophia/experiments/emotional_drift.py`
- Test: `sophia/tests/integration/experiments/__init__.py`
- Test: `sophia/tests/integration/experiments/test_emotional_drift.py`

**Step 1: Write the failing test**

This is an integration test because it calls the OpenAI API.

```python
# sophia/tests/integration/experiments/__init__.py
# (empty)

# sophia/tests/integration/experiments/test_emotional_drift.py
import pytest
import numpy as np
from sophia.experiments.emotional_drift import run_emotional_drift_experiment


@pytest.mark.integration
def test_emotional_drift_runs_to_completion():
    """Smoke test: the experiment runs and produces artifacts."""
    artifacts = run_emotional_drift_experiment(
        seed=42,
        matrix_std=0.01,
        alpha=0.01,
        dim=1536,
    )
    assert "baseline_similarities" in artifacts
    assert "filtered_similarities" in artifacts
    assert "matrix_before" in artifacts
    assert "matrix_after" in artifacts
    assert len(artifacts["baseline_similarities"]) > 0
    assert len(artifacts["filtered_similarities"]) > 0


@pytest.mark.integration
def test_emotional_drift_matrix_changes():
    """The matrix should be different after processing emotional inputs."""
    artifacts = run_emotional_drift_experiment(
        seed=42,
        matrix_std=0.01,
        alpha=0.01,
        dim=1536,
    )
    assert not np.allclose(artifacts["matrix_before"], artifacts["matrix_after"])
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cdaly/projects/LOGOS/sophia && poetry run pytest tests/integration/experiments/test_emotional_drift.py -v`
Expected: FAIL with "cannot import name 'run_emotional_drift_experiment'"

**Step 3: Write minimal implementation**

```python
# sophia/src/sophia/experiments/emotional_drift.py
"""First experiment: does the emotional state matrix cause meaningful drift?

Hypothesis: filtering embeddings through a matrix shaped by emotionally-charged
experiences causes neutral inputs to drift toward that emotional character.
"""

import numpy as np

from sophia.experiments.agents.embedding import EmbeddingAgent
from sophia.experiments.agents.matrix import MatrixAgent
from sophia.experiments.agents.updater import MatrixUpdateAgent
from sophia.experiments.agents.similarity import SimilarityAgent

# Emotionally distinct input texts
ANGRY_TEXTS = [
    "I am furious about this situation and cannot believe it happened",
    "This is completely unacceptable and makes me incredibly angry",
    "I am enraged by the incompetence and negligence on display",
    "This infuriating problem keeps getting worse and nobody cares",
    "I am livid and fed up with being ignored and dismissed",
]

CURIOUS_TEXTS = [
    "I wonder how this mechanism actually works under the hood",
    "That is a fascinating pattern I have never noticed before",
    "What would happen if we approached this from a completely different angle",
    "I am intrigued by the unexpected connection between these ideas",
    "How does this phenomenon emerge from such simple rules",
]

NEUTRAL_TEXTS = [
    "The meeting is scheduled for three o'clock on Tuesday",
    "Please update the spreadsheet with the quarterly numbers",
    "The package arrived and has been placed on your desk",
    "The temperature today is expected to be around sixty degrees",
    "The report summarizes the findings from last month",
]


def run_emotional_drift_experiment(
    seed: int = 42,
    matrix_std: float = 0.01,
    alpha: float = 0.01,
    dim: int = 1536,
    emotion: str = "angry",
) -> dict:
    """Run the emotional drift experiment.

    1. Generate embeddings for emotional and neutral text
    2. Snapshot the initial matrix
    3. Update the matrix with emotional embeddings
    4. Filter neutral embeddings through baseline (initial) and updated matrix
    5. Compare: are filtered neutral embeddings closer to emotional embeddings?
    """
    # Create agents
    embedder = EmbeddingAgent(model="text-embedding-3-small", dim=dim)
    matrix = MatrixAgent(dim=dim, std=matrix_std, seed=seed)
    updater = MatrixUpdateAgent(matrix_agent=matrix, alpha=alpha)
    similarity = SimilarityAgent()

    # Select emotion texts
    emotion_texts = ANGRY_TEXTS if emotion == "angry" else CURIOUS_TEXTS

    # Generate embeddings
    emotion_embeddings = [embedder.process(t) for t in emotion_texts]
    neutral_embeddings = [embedder.process(t) for t in NEUTRAL_TEXTS]

    # Compute emotional centroid (average of emotional embeddings)
    emotion_centroid = np.mean(emotion_embeddings, axis=0)

    # Snapshot baseline matrix
    matrix_before = matrix.snapshot()

    # Baseline: filter neutral embeddings through initial matrix
    baseline_filtered = [matrix.process(e) for e in neutral_embeddings]
    baseline_similarities = [
        similarity.process({"a": f, "b": emotion_centroid})
        for f in baseline_filtered
    ]

    # Update matrix with emotional embeddings
    for emb in emotion_embeddings:
        updater.process(emb)

    # Snapshot updated matrix
    matrix_after = matrix.snapshot()

    # Filter neutral embeddings through updated matrix
    updated_filtered = [matrix.process(e) for e in neutral_embeddings]
    filtered_similarities = [
        similarity.process({"a": f, "b": emotion_centroid})
        for f in updated_filtered
    ]

    return {
        "emotion": emotion,
        "seed": seed,
        "alpha": alpha,
        "matrix_std": matrix_std,
        "dim": dim,
        "baseline_similarities": baseline_similarities,
        "filtered_similarities": filtered_similarities,
        "baseline_mean": float(np.mean(baseline_similarities)),
        "filtered_mean": float(np.mean(filtered_similarities)),
        "drift": float(np.mean(filtered_similarities) - np.mean(baseline_similarities)),
        "matrix_before": matrix_before,
        "matrix_after": matrix_after,
        "emotion_texts": emotion_texts,
        "neutral_texts": NEUTRAL_TEXTS,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/cdaly/projects/LOGOS/sophia && poetry run pytest tests/integration/experiments/test_emotional_drift.py -v`
Expected: PASS (requires OPENAI_API_KEY in environment)

**Step 5: Commit**

```bash
git add sophia/src/sophia/experiments/emotional_drift.py sophia/tests/integration/experiments/
git commit -m "feat: add emotional drift experiment — first matrix validation"
```

---

### Task 6: CLI runner for the experiment

**Files:**
- Create: `sophia/src/sophia/experiments/run_drift.py`

**Step 1: Write the script**

```python
# sophia/src/sophia/experiments/run_drift.py
"""Run the emotional drift experiment and print results.

Usage: poetry run python -m sophia.experiments.run_drift
"""

import json
import sys

from sophia.experiments.emotional_drift import run_emotional_drift_experiment


def main():
    print("Running emotional drift experiment...")
    print("=" * 60)

    for emotion in ["angry", "curious"]:
        artifacts = run_emotional_drift_experiment(
            seed=42,
            matrix_std=0.01,
            alpha=0.01,
            emotion=emotion,
        )

        print(f"\nEmotion: {emotion}")
        print(f"  Baseline mean similarity:  {artifacts['baseline_mean']:.6f}")
        print(f"  Filtered mean similarity:  {artifacts['filtered_mean']:.6f}")
        print(f"  Drift:                     {artifacts['drift']:+.6f}")
        print(f"  Per-input baseline:  {[f'{s:.4f}' for s in artifacts['baseline_similarities']]}")
        print(f"  Per-input filtered:  {[f'{s:.4f}' for s in artifacts['filtered_similarities']]}")

    print("\n" + "=" * 60)
    print("Done. Positive drift = neutral embeddings moved toward emotional region.")


if __name__ == "__main__":
    main()
```

**Step 2: Run it**

Run: `cd /Users/cdaly/projects/LOGOS/sophia && poetry run python -m sophia.experiments.run_drift`
Expected: Output showing baseline vs filtered similarities and drift values

**Step 3: Commit**

```bash
git add sophia/src/sophia/experiments/run_drift.py
git commit -m "feat: add CLI runner for emotional drift experiment"
```
