# Persona Entries & Emotional State Matrix — Design Exploration

**Date:** 2026-02-26
**Status:** In progress — open questions remain

## Starting Problem

Sophia is the non-linguistic cognitive core. She operates on embeddings, graph structures, and relationships — never on text. But persona entries (as currently defined) have a `summary` (text) and `sentiment` (a named label like "confident", "cautious"). Neither of these is something Sophia should be generating.

The question: how does Sophia create persona entries from interactions?

## Key Architectural Constraints

- **Sophia is non-linguistic.** She never reads or generates text. She reasons in embedding space and graph topology.
- **Hermes is the language component.** STT, TTS, NLP, embeddings, LLM calls — all Hermes.
- **Hermes is stateless.** He processes and passes through; he doesn't accumulate state.
- **Sophia drives cognition.** She decides what's important, what to record, what to act on. Hermes provides raw signal; Sophia interprets.

## What Persona Entries Are

Persona entries are **episodic memories** — raw observations of interactions. They are NOT evaluations. They don't carry sentiment labels or text summaries at creation time.

Over time, CWM-E reflects on accumulated persona entries to build a self-model. The interpretation comes later; the entry itself is just a structural record.

**Reflection entries** are higher-order observations — patterns Sophia notices across persona entries. They have the same structural format as persona entries, but reference entries rather than raw interactions. This can recurse (reflections on reflections).

## What Sophia Actually Knows

When an interaction flows through Sophia (via Hermes's proposal), she has access to:

### From Hermes (in the proposal):
- **Content embeddings** — the interaction's representation in embedding space
- **Affect signal** — a valence/arousal reading of the user's language (Hermes reads this from the text; Sophia cannot)

### From her own processing:
- **Milvus similarity distances** — how close the incoming embeddings are to existing graph content
- **Graph topology changes** around the insertion point:
  - Local density (how many nodes/edges nearby)
  - Conflict count (contradictions with existing assertions)
  - Confidence delta (new assertion vs. existing nodes it touches)
  - Bridge score (does it connect previously separate components)
  - Path disruption (does it break or create causal chains)

These structural signals are genuinely non-linguistic and computable. They are Sophia's native vocabulary for characterizing interactions.

## Sentiment: The Resolution

Sophia cannot independently derive sentiment from embeddings — the emotional signal in embeddings was put there by language, and extracting it reliably requires language understanding. Projection onto pre-defined axes (e.g., "negative" to "positive") was explored but rejected: there's no principled basis for confidence that such an axis cleanly isolates emotion from everything else in embedding space.

### What Sophia CAN do: Graph-based affect signals

The structure/shape of the graph relative to the placement of a new element carries meaningful information:

| Graph condition | What it indicates |
|---|---|
| Contradicts high-confidence node | Conflict / challenge |
| Lands in sparse region | Uncharted territory |
| Bridges two disconnected subgraphs | Unexpected connection |
| Reinforces existing dense cluster | Confirmation / routine |
| Creates a causal cycle where none existed | Paradox / incoherence |
| High-confidence node gets second contradiction | Pattern of dispute |

These are all derivable from basic graph metrics around the insertion point. No embeddings, no language, no Hermes. Just structure.

### Hermes provides affect signal in proposals

For user-facing interactions where the emotional signal is in the language (e.g., user becomes agitated about something routine), Hermes includes an affect reading in the proposal. This is architecturally correct — the component that understands language reads the emotion. Sophia uses it as one input among many but doesn't originate it.

### Emotion nodes emerge organically

1. Sophia tries to classify an interaction against existing emotion nodes in the graph
2. Early on, there are none — no matches
3. Finding nothing, Sophia knows something is unclassified — she has structural signals that indicate something notable, but no emotional category
4. She creates a new emotion node from the structural data she has
5. She asks Hermes: "what do you call this?" — Hermes provides a label and its embedding
6. That node now exists in the graph for future matching

The emotional vocabulary grows organically rather than being a hardcoded enum. Hermes builds the map legend; Sophia reads the map.

### Persona/reflection entries as queryable context

When new interactions come in, Sophia can query existing persona and reflection entries near the relevant region — "how do I usually react when I see this sort of thing?" This gives Sophia learned intuition: her own accumulated structural experience influencing how she processes new events.

## The Emotional State Matrix

The emotional state is not a property on individual entries. It is a **matrix that filters everything Sophia perceives and does**.

### Structure
- **Dimensions:** N x N where N = embedding dimension (e.g., 1536 x 1536)
- **Value range:** -1 to 1
- **Initialization:** Zeros (blank slate)
- **Application:** `output = (M + I) * embedding` — zeros = identity behavior, no filtering
- **Bounding:** Values kept in range via activation function (tanh, etc.)

### Interpretation of values
| Value | Meaning | Effective multiplier |
|---|---|---|
| -1 | Full suppression | 0x |
| -0.5 | Dampen | 0.5x |
| 0 | Neutral (pass through) | 1x |
| 0.5 | Heighten | 1.5x |
| 1 | Full amplification | 2x |

The -1 to 1 range was chosen because:
- 0 as neutral is intuitive (zeros = blank slate)
- Sign directly indicates direction (negative = dampen, positive = heighten)
- Maps cleanly to standard activation functions (tanh, etc.) which naturally bound values
- Saturation curve is meaningful: early experiences shape the filter easily; well-established biases resist further change

### What the matrix does

The matrix transforms embeddings before they enter Sophia's processing. It warps embedding space — amplifying certain dimensions, suppressing others, potentially mixing them via off-diagonal values. Everything downstream (Milvus search, graph placement, pattern matching) operates on the filtered version.

The matrix doesn't need to be human-interpretable. Individual dimensions of modern embeddings don't correspond to nameable concepts. But the space is consistent within a single embedding model, and modifications have consistent effects on future processing. The test is not "can I read it" but "does Sophia's behavior change appropriately over time."

**Critical requirement:** Always use the same embedding model. If the model changes, the entire matrix becomes meaningless.

### How updates work

The matrix is updated by significant events, with magnitude tiered by significance:

| Event | Update magnitude |
|---|---|
| Raw interaction (no persona entry) | No update |
| Persona entry created | Small nudge |
| Reflection entry created | Larger push |
| Memory promotion (ephemeral to short-term) | Another nudge |
| Memory promotion (short-term to long-term) | Larger push |
| Higher-order reflection | Larger still |

### Update mechanism: Single-column update (working hypothesis)

The proposed approach is surgical: update a single column of the matrix per persona entry, rather than modifying the entire matrix.

- Identify the relevant column for the entry (selection method TBD — see open questions)
- Nudge that column using the entry's embedding: `column_j = activation_fn(column_j + alpha * entry_embedding)`
- alpha is small for persona entries, larger for reflections and promotions
- Activation function (tanh) enforces -1 to 1 bounds

**What this means geometrically:** Modifying column j changes what dimension j of the input contributes to the output. Starting from zero (pass-through), a modified column means that dimension now evokes the pattern of the persona entry across the whole output. That's association — experience in that part of the space colors everything that touches it.

**Properties:**
- 99% of the matrix stays at zero — most perception is unfiltered
- Columns touched repeatedly drift further from zero — accumulated experience
- Columns never touched remain zero — no emotional bias in that region
- Competing updates (contradictory experiences on same column) average out
- Saturation via tanh means early experiences shape easily, established biases resist change

### Decay toward neutral

Unreinforced changes should naturally fade. A decay mechanism pulls the matrix back toward zeros over time:

`M = (1 - lambda) * M + update`

This means:
- Biases that aren't reinforced through reflection entries and memory promotions naturally decay
- The matrix has a "resting state" (zeros) it returns to without continued stimulation
- Reinforcement through the reflection/promotion cycle is necessary to maintain a bias

## The Full Cycle

1. Interaction embedding arrives from Hermes (with affect signal)
2. Sophia filters it through the emotional state matrix: `output = (M + I) * embedding`
3. Filtered embedding hits the graph — Milvus search, node placement
4. Sophia computes structural metrics around the insertion (density, conflict, bridging, etc.)
5. Sophia queries existing persona/reflection entries near this region — "what's my history here?"
6. Combination of current metrics + historical context determines: worth recording?
7. If yes, create persona entry (embedding + structural fingerprint)
8. Entry updates one column of the matrix (small nudge)
9. Activation function enforces bounds
10. Periodically, reflection process reviews recent persona entries, finds patterns, creates reflection entries (bigger matrix push)
11. Memory tier promotions provide additional matrix pushes
12. Next interaction gets filtered through the slightly updated matrix

The matrix is the emotional state. The persona entries are what train it. No sentiment labels anywhere in Sophia's native processing.

## Open Questions

### 1. Column selection
Which column of the matrix does a persona entry update? "Most activated dimension" (highest absolute value in the embedding) was proposed but questioned. The selection criterion needs more thought:
- Highest absolute value may just reflect commonly-strong dimensions in the model, not what's distinctive about this entry
- Could use "most different from running average" to find distinctive dimensions
- Could update multiple columns proportionally to activation
- Could be determined by the structural metrics rather than the embedding

### 2. What exactly gets written to the column
We said "the entry's embedding" but embedding values aren't bounded to -1 to 1. Need to determine:
- Is it the raw embedding scaled by alpha?
- Is it something derived from both the embedding and the structural metrics?
- How do the structural signals (the graph-topology affect vocabulary) influence the update?

### 3. Update math rigor
- Does the single-column update preserve useful properties of the matrix?
- What prevents pathological states (e.g., a column that suppresses everything)?
- Is the (M + I) application the right formulation, or should the identity be handled differently?

### 4. Threshold for recording
What makes an interaction "worth recording" as a persona entry? Candidates:
- Absolute thresholds on structural metrics (high arousal, strong conflict)
- Relative thresholds against historical context (unusual for this region)
- The first mechanism (absolute) bootstraps the second (relative)

### 5. Reflection process mechanics
- What triggers reflection? Periodic schedule? Accumulation threshold?
- How does reflection select which persona entries to examine?
- What determines whether a pattern across entries deserves a reflection entry?

### 6. Embedding dimension relevance
Can individual values of an embedding vector indicate relative importance to overall meaning? Higher absolute values suggest more activation, but this is a rough heuristic. Needs more investigation for the column selection problem.

### 7. Matrix stability
- What decay rate (lambda) prevents drift while allowing learning?
- Should there be a maximum deviation from zero?
- How to handle the matrix if the embedding model is ever changed?

## Resolved Questions

### Threshold for recording (formerly Open Question #4)

**Resolution:** High absolute value = meaningful.

In the -1 to 1 range, values near 0 are neutral/uninteresting. Values near -1 or +1 are maximally significant regardless of direction. The sign indicates *which way*; the magnitude indicates *how much it matters*.

The threshold for "worth recording as a persona entry" is: does the interaction produce any structural metrics with high absolute values? If the conflict score, bridge score, density delta, or Hermes's affect signal are all near zero — routine, nothing to record. If any of them spike toward the extremes — that's a recording trigger.

This also connects to the matrix update mechanism: the magnitude of the structural signal can directly scale the alpha (update strength). Stronger signals push the matrix harder. A mild novelty barely nudges it. A major contradiction shoves it. The recording threshold and the update magnitude are the same signal — just with a floor applied.

## Experiment Framework

### Principles
- **Generate data early.** Implementations can be naive. The point is to establish a baseline against which improvement can be measured.
- **Everything is swappable.** Components sit behind clean interfaces. Any piece — column selection, update rule, recording threshold, structural metrics, reflection logic — can be replaced without touching anything else.
- **The data teaches you what to look for.** Evaluation criteria will emerge from observing the output of multiple configurations, not from theoretical design.

### Experiment structure
- **Configuration:** Which component implementations are plugged in, matrix initialization parameters, any other tunables
- **Start condition:** Matrix initialized with gaussian noise (mean=0, configurable std, e.g. 0.01). Random seed recorded for reproducibility. Configurable — could be zeros, a saved matrix state, or any custom initialization. Graph state also configurable (empty, seeded, snapshot from previous run).
- **Input:** A corpus of interactions run through the system. For the first experiments, content doesn't matter — even synthetic/random input is fine. The corpus is fixed per experiment for reproducibility.
- **End condition:** Input exhausted. The experiment is over when you run out of input.
- **Output/artifacts:** Matrix state (before and after), persona entries created, reflection entries created, structural metrics logged, any other observables.

### Evaluation
Evaluation criteria are NOT pre-defined. The first experiments exist to generate data so that useful evaluation criteria can be discovered. After observing output from multiple configurations using the same input, patterns will emerge — too many entries, too few, matrix collapse, degenerate states, etc. Those observations become the evaluation metrics for subsequent experiments.

### Iteration loop
1. Define a configuration (swap one component or tunable)
2. Run the fixed input corpus through it
3. Examine output artifacts
4. Compare against baseline and previous configurations
5. Observations inform both evaluation criteria and next configuration to try

Automating this loop (configuration sweep + comparison) is a long-term goal. Systematically exploring the configuration space would allow discovery of effective component combinations without manual experimentation.

## Experiment Framework

The experiment infrastructure for this work is defined in a separate document: `2026-02-26-experiment-framework-design.md`. The persona entry / emotional matrix work is the first use case, but the framework is domain-agnostic.

## Implementation Handoff

### What to build first: The dumbest possible experiment

**Goal:** Test whether filtering embeddings through a matrix shaped by emotionally-charged experiences causes neutral inputs to drift toward that emotional character.

### Prerequisites
- Neo4j and Milvus wiped clean (done 2026-02-26)
- OpenAI embedding model (text-embedding-3-small, 1536 dimensions)
- Embedding model is configurable — everything derives from `embedding_dim` in config
- If embedding model changes, matrix must be rebuilt from scratch

### Experiment definition

**Arrange:**
- Initialize a 1536x1536 matrix with gaussian noise (mean=0, small std, seeded for reproducibility)
- Generate embeddings for emotionally distinct text via OpenAI:
  - A set of "angry" phrases
  - A set of "curious" phrases
  - A set of neutral phrases
- All embeddings via the same model

**Act:**
- Use the emotional embeddings to update the matrix (simulating persona entries)
- Naive column selection: highest absolute value in the embedding vector (starting point — we expect to improve this)
- Update rule: `column_j = tanh(column_j + alpha * entry_embedding)`
- Alpha is small (e.g., 0.01)
- Then pass neutral embeddings through the updated matrix: `output = (M + I) * embedding`

**Assert:**
- Measure cosine similarity between filtered neutral embeddings and the emotional embeddings that shaped the matrix
- Compare against baseline: cosine similarity of the SAME neutral embeddings passed through the UNMODIFIED (initial gaussian) matrix
- Success = filtered outputs are measurably closer to the emotional embeddings than unfiltered outputs

### Architecture

**Contracts live in logos (foundry):**
- `AgentDefinition` base class — minimal contract: `process(input_data) -> output`
- Richer subclasses: `StatefulAgent`, `GraphAgent`, `ReflectiveAgent`
- Experiment runner: arrange/act/assert lifecycle
- Experiment definition format (config-driven)
- Factory function pattern for creating agents from config

**Experiment implementations live in sophia:**
- Specific agents: matrix initializer, embedding generator, matrix updater, similarity measurer
- Experiment configs for specific runs
- Results/artifacts storage

### Agent definitions needed for first experiment

1. **EmbeddingAgent** — takes text, returns embedding vector (wraps OpenAI API call)
2. **MatrixInitAgent** (StatefulAgent) — creates and holds the matrix, exposes state/snapshot/reset
3. **MatrixFilterAgent** — takes embedding + matrix, returns filtered embedding: `(M + I) * embedding`
4. **MatrixUpdateAgent** — takes embedding + matrix, updates one column, returns updated matrix
5. **SimilarityAgent** — takes two embeddings, returns cosine similarity score

Each is trivial. The value is in the interfaces, not the implementations.

### What this experiment will tell us
- Does the matrix-as-emotional-filter concept work at all?
- Does the naive column selection (highest absolute value) produce meaningful results?
- Does the update rule (additive + tanh) converge, diverge, or collapse?
- How sensitive is the result to alpha (update magnitude)?
- How sensitive is the result to the initial gaussian std?

### What it will NOT tell us
- Whether graph-based structural metrics are useful (no graph in this experiment)
- Whether the reflection/promotion tiering matters (single-pass, no reflection)
- Whether the recording threshold works (we're manually choosing what updates the matrix)
- How this behaves at scale (small input corpus)

### Open questions carried forward
- Column selection: "highest absolute value" is the naive starting point. May need to become "most distinctive relative to running average" or something else entirely. The experiment will generate data to inform this.
- Update math: additive + tanh is the naive starting point. Multiplicative composition or other approaches may be better. Again, data will inform.
- Embedding dimension values as importance proxy: rough heuristic, needs empirical validation — this experiment provides that.

### Next experiments (after first results)
1. Vary alpha — sweep from 0.001 to 0.1, compare results
2. Vary initial std — sweep, compare
3. Try different column selection strategies — random, top-k, proportional
4. Introduce real graph structure — use Neo4j, compute structural metrics, let those influence updates
5. Add reflection — multiple passes, let the matrix compound
6. Add memory tier promotion — ephemeral/short/long with different update magnitudes
