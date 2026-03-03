# Experiment Framework — Design Exploration

**Date:** 2026-02-26
**Status:** In progress
**Scope:** Foundry-level infrastructure — not specific to any one component

## Purpose

A general-purpose framework for running controlled experiments against any LOGOS component or pipeline. The framework handles orchestration; the experiment definition provides the specifics.

## Core Structure: Arrange, Act, Assert

Every experiment follows three phases:

### Arrange
Set up initial state from configuration.
- Use factory functions to create agents for each pipeline step
- Initialize state (matrix, graph, whatever the experiment needs)
- Load input corpus
- Record all configuration for reproducibility (including random seeds)

### Act
Run the input through the pipeline.
- Iterate over the input corpus
- For each input, execute the pipeline steps in order
- Each step calls its agent's `process` method, passes output to the next step
- Log/capture intermediate state as configured

### Assert
Evaluate or capture output.
- Early experiments: just capture artifacts for manual inspection
- Later: automated checks against learned criteria
- Compare output across configurations using the same input
- Assert phase is itself pluggable — swap in different evaluation strategies

## Agent Definition

### Base: AgentDefinition

The minimal contract for pipeline participation. The framework only calls `process`. That's the one thing every agent must have.

```python
class AgentDefinition(Protocol):
    """Minimal contract — the framework calls process, nothing else."""
    
    def process(self, input_data: Any) -> Any: ...
```

Agents are free to have any additional methods, state, dependencies, or side effects. The base definition is a floor, not a ceiling.

### Subclasses: Richer interfaces

Subclasses expose richer capabilities for agents that need them. The framework still only calls `process`. The experiment definition and other agents use the richer interface for setup, inspection, and coordination.

```python
class StatefulAgent(AgentDefinition):
    """Agent with observable, snapshottable, resettable state."""
    
    def process(self, input_data: Any) -> Any: ...
    def get_state(self) -> Any: ...
    def reset(self, state: Any = None) -> None: ...
    def snapshot(self) -> Any: ...

class GraphAgent(AgentDefinition):
    """Agent that interacts with the HCG."""
    
    def process(self, input_data: Any) -> Any: ...
    def query(self, **kwargs) -> Any: ...
    def insert(self, **kwargs) -> Any: ...

class ReflectiveAgent(AgentDefinition):
    """Agent capable of self-inspection and reflection."""
    
    def process(self, input_data: Any) -> Any: ...
    def reflect(self, entries: list) -> Any: ...
    def introspect(self) -> Any: ...
```

These are examples, not exhaustive. New subclasses can be added without touching the base or the framework. The hierarchy grows with the project.

An experiment step that needs a stateful agent type-hints for `StatefulAgent`. The experiment definition uses the richer interface for arrange/assert. Everything composes — pick the level of capability you need.

### Factory functions

Agents are created by factory functions. The experiment definition specifies which factory to use for each pipeline step.

```python
def make_naive_matrix_updater(config: dict) -> StatefulAgent:
    """Factory returns an agent satisfying StatefulAgent interface."""
    return NaiveMatrixUpdater(
        alpha=config.get("alpha", 0.01),
        activation=config.get("activation", "tanh"),
    )
```

Swap the factory in the experiment definition, get completely different behavior. The pipeline doesn't care which factory produced the agent.

## Experiment Definition

An experiment is fully specified by:

```yaml
experiment:
  name: string
  description: string
  seed: int
  
  arrange:
    init_config: dict              # initialization parameters
    pipeline:                      # ordered steps
      - name: step_name
        factory: function_ref      # factory function that returns an agent
        config: dict               # passed to the factory
    
  act:
    input_corpus: path | generator
    logging: dict                  # what intermediate state to capture
    
  assert:
    evaluators: list               # what to check/measure/capture
    comparison_baseline: path      # optional: previous run to compare against
```

## Key Design Principles

### Minimal base, rich subclasses
The base agent definition is one method: `process`. Subclasses add capabilities as needed. The framework uses only the base. Experiment definitions and other agents use whatever interface they need.

### Agents are free to vary
Beyond `process`, agents can do anything. Simple agents wrap a numpy operation. Complex agents talk to Neo4j, Milvus, and Hermes. The framework sees them identically.

### Domain-agnostic
The framework doesn't know about persona entries, matrices, embeddings, or graphs. It knows about configurations, pipelines, input, and output.

### Reproducibility by default
- Random seeds recorded and replayable
- Configurations captured as artifacts
- Input corpora are fixed per experiment
- Same input + same config = same output (deterministic where possible)

## Experiment Lifecycle

1. Load experiment definition
2. **Arrange:** Call factory functions with config to create agents, initialize state, load input
3. **Act:** For each input in corpus, call `process` on each agent in pipeline order, log as configured
4. **Assert:** Run evaluators against output artifacts
5. Persist all artifacts (config, input reference, outputs, evaluations, final state)
6. Optionally compare against baseline run

## Configuration Sweep (Future)

1. Define a sweep: which factory to vary, or which config parameter
2. Framework generates N experiment definitions (one per configuration)
3. Run all experiments against the same input corpus
4. Compare results across the sweep
5. Surface patterns — which configurations produce what behaviors

## Open Questions

### 1. Data passing between steps
Does step N+2 need access to step N's output? Options: each step sees only the previous output, or a context/accumulator object carries everything forward.

### 2. State snapshots
For stateful agents, when does the framework snapshot? Only at start/end, or configurable checkpoints mid-experiment?

### 3. Parallelism
Can experiments in a sweep run in parallel? Shared infrastructure (Neo4j, Milvus) may conflict.

### 4. Where does this live?
Foundry (logos repo) seems right — it's cross-cutting infrastructure usable by any component.

### 5. Subclass discovery
How do experiment definitions reference agent subclasses? Import paths? A registry? Convention-based discovery?
