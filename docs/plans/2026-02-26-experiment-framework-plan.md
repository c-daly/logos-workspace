# Experiment Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the foundry-level experiment framework with agent definitions, experiment runner, and arrange/act/assert lifecycle.

**Architecture:** Minimal agent protocol with `process` method as the only contract. Factory functions create agents from config. Experiment runner orchestrates arrange/act/assert phases. Everything configurable, nothing hardcoded.

**Tech Stack:** Python 3.12, pytest, pydantic for config models

---

### Task 1: AgentDefinition base protocol

**Files:**
- Create: `logos/logos_experiment/__init__.py`
- Create: `logos/logos_experiment/agent.py`
- Test: `logos/tests/unit/experiment/test_agent.py`

**Step 1: Write the failing test**

```python
# logos/tests/unit/experiment/__init__.py
# (empty)

# logos/tests/unit/experiment/test_agent.py
from logos_experiment.agent import AgentDefinition


class SimpleAgent:
    """Minimal agent that doubles its input."""

    def process(self, input_data):
        return input_data * 2


def test_agent_satisfies_protocol():
    agent = SimpleAgent()
    assert isinstance(agent, AgentDefinition)


def test_agent_process():
    agent = SimpleAgent()
    assert agent.process(5) == 10
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cdaly/projects/LOGOS/logos && poetry run pytest tests/unit/experiment/test_agent.py -v`
Expected: FAIL with "No module named 'logos_experiment'"

**Step 3: Write minimal implementation**

```python
# logos/logos_experiment/__init__.py
from logos_experiment.agent import AgentDefinition, StatefulAgent

__all__ = ["AgentDefinition", "StatefulAgent"]

# logos/logos_experiment/agent.py
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentDefinition(Protocol):
    """Minimal contract for experiment pipeline participation.

    The framework calls process(). That's the whole contract.
    Agents are free to have any additional methods, state, or dependencies.
    """

    def process(self, input_data: Any) -> Any: ...


@runtime_checkable
class StatefulAgent(AgentDefinition, Protocol):
    """Agent with observable, snapshottable, resettable state."""

    def get_state(self) -> Any: ...
    def reset(self, state: Any = None) -> None: ...
    def snapshot(self) -> Any: ...
```

**Step 4: Add logos_experiment to pyproject.toml packages**

In `/Users/cdaly/projects/LOGOS/logos/pyproject.toml`, add `logos_experiment` to the packages list.

**Step 5: Run test to verify it passes**

Run: `cd /Users/cdaly/projects/LOGOS/logos && poetry run pytest tests/unit/experiment/test_agent.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add logos/logos_experiment/ logos/tests/unit/experiment/
git commit -m "feat: add AgentDefinition protocol for experiment framework"
```

---

### Task 2: Experiment config model

**Files:**
- Create: `logos/logos_experiment/config.py`
- Test: `logos/tests/unit/experiment/test_config.py`

**Step 1: Write the failing test**

```python
# logos/tests/unit/experiment/test_config.py
from logos_experiment.config import ExperimentConfig, PipelineStep


def test_experiment_config_minimal():
    config = ExperimentConfig(
        name="test-experiment",
        seed=42,
        pipeline=[
            PipelineStep(name="step1", factory="module.path:make_agent", config={}),
        ],
    )
    assert config.name == "test-experiment"
    assert config.seed == 42
    assert len(config.pipeline) == 1


def test_experiment_config_defaults():
    config = ExperimentConfig(
        name="test",
        seed=0,
        pipeline=[],
    )
    assert config.description == ""
    assert config.init_config == {}
    assert config.logging == {}


def test_pipeline_step():
    step = PipelineStep(
        name="filter",
        factory="sophia.experiments.agents:make_filter",
        config={"alpha": 0.01},
    )
    assert step.name == "filter"
    assert step.config["alpha"] == 0.01
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cdaly/projects/LOGOS/logos && poetry run pytest tests/unit/experiment/test_config.py -v`
Expected: FAIL with "cannot import name 'ExperimentConfig'"

**Step 3: Write minimal implementation**

```python
# logos/logos_experiment/config.py
from pydantic import BaseModel, Field


class PipelineStep(BaseModel):
    """One step in an experiment pipeline."""

    name: str
    factory: str  # dotted.path:function_name
    config: dict = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Full specification of an experiment."""

    name: str
    description: str = ""
    seed: int
    init_config: dict = Field(default_factory=dict)
    pipeline: list[PipelineStep]
    input_corpus: str = ""  # path or generator reference
    logging: dict = Field(default_factory=dict)
    evaluators: list[str] = Field(default_factory=list)
    comparison_baseline: str = ""
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/cdaly/projects/LOGOS/logos && poetry run pytest tests/unit/experiment/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add logos/logos_experiment/config.py logos/tests/unit/experiment/test_config.py
git commit -m "feat: add ExperimentConfig and PipelineStep models"
```

---

### Task 3: Experiment runner

**Files:**
- Create: `logos/logos_experiment/runner.py`
- Test: `logos/tests/unit/experiment/test_runner.py`

**Step 1: Write the failing test**

```python
# logos/tests/unit/experiment/test_runner.py
from logos_experiment.runner import ExperimentRunner
from logos_experiment.config import ExperimentConfig, PipelineStep


class DoublerAgent:
    def process(self, input_data):
        return input_data * 2


class AdderAgent:
    def process(self, input_data):
        return input_data + 1


def make_doubler(config):
    return DoublerAgent()


def make_adder(config):
    return AdderAgent()


def test_runner_arrange_creates_agents():
    config = ExperimentConfig(
        name="test",
        seed=42,
        pipeline=[
            PipelineStep(name="double", factory="unused", config={}),
            PipelineStep(name="add", factory="unused", config={}),
        ],
    )
    runner = ExperimentRunner(config)
    runner.arrange(factories={"double": make_doubler, "add": make_adder})
    assert len(runner.agents) == 2


def test_runner_act_runs_pipeline():
    config = ExperimentConfig(
        name="test",
        seed=42,
        pipeline=[
            PipelineStep(name="double", factory="unused", config={}),
            PipelineStep(name="add", factory="unused", config={}),
        ],
    )
    runner = ExperimentRunner(config)
    runner.arrange(factories={"double": make_doubler, "add": make_adder})
    results = runner.act(input_corpus=[5, 10, 15])
    # 5 * 2 + 1 = 11, 10 * 2 + 1 = 21, 15 * 2 + 1 = 31
    assert results == [11, 21, 31]


def test_runner_assert_captures_results():
    config = ExperimentConfig(
        name="test",
        seed=42,
        pipeline=[
            PipelineStep(name="double", factory="unused", config={}),
        ],
    )
    runner = ExperimentRunner(config)
    runner.arrange(factories={"double": make_doubler})
    runner.act(input_corpus=[5])
    artifacts = runner.assert_results()
    assert "results" in artifacts
    assert artifacts["results"] == [10]
    assert artifacts["config"] == config
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/cdaly/projects/LOGOS/logos && poetry run pytest tests/unit/experiment/test_runner.py -v`
Expected: FAIL with "No module named 'logos_experiment.runner'"

**Step 3: Write minimal implementation**

```python
# logos/logos_experiment/runner.py
from __future__ import annotations

import numpy as np
from typing import Any, Callable

from logos_experiment.agent import AgentDefinition
from logos_experiment.config import ExperimentConfig


class ExperimentRunner:
    """Orchestrates arrange/act/assert lifecycle for an experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.agents: list[AgentDefinition] = []
        self._results: list[Any] = []
        self._rng = np.random.default_rng(config.seed)

    def arrange(self, factories: dict[str, Callable] | None = None) -> None:
        """Create agents from factories using pipeline config."""
        factories = factories or {}
        for step in self.config.pipeline:
            factory = factories.get(step.name)
            if factory is None:
                raise ValueError(f"No factory provided for step '{step.name}'")
            agent = factory(step.config)
            self.agents.append(agent)

    def act(self, input_corpus: list[Any]) -> list[Any]:
        """Run each input through the pipeline of agents."""
        self._results = []
        for input_data in input_corpus:
            current = input_data
            for agent in self.agents:
                current = agent.process(current)
            self._results.append(current)
        return self._results

    def assert_results(self) -> dict[str, Any]:
        """Return experiment artifacts for evaluation."""
        return {
            "config": self.config,
            "results": list(self._results),
            "seed": self.config.seed,
        }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/cdaly/projects/LOGOS/logos && poetry run pytest tests/unit/experiment/test_runner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add logos/logos_experiment/runner.py logos/tests/unit/experiment/test_runner.py
git commit -m "feat: add ExperimentRunner with arrange/act/assert lifecycle"
```

---

### Task 4: Update logos_experiment __init__.py exports

**Files:**
- Modify: `logos/logos_experiment/__init__.py`

**Step 1: Update exports**

```python
# logos/logos_experiment/__init__.py
from logos_experiment.agent import AgentDefinition, StatefulAgent
from logos_experiment.config import ExperimentConfig, PipelineStep
from logos_experiment.runner import ExperimentRunner

__all__ = [
    "AgentDefinition",
    "StatefulAgent",
    "ExperimentConfig",
    "PipelineStep",
    "ExperimentRunner",
]
```

**Step 2: Run all experiment tests**

Run: `cd /Users/cdaly/projects/LOGOS/logos && poetry run pytest tests/unit/experiment/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add logos/logos_experiment/__init__.py
git commit -m "feat: export all experiment framework components"
```
