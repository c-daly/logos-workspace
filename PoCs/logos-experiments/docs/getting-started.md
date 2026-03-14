# Getting Started

## Step 1: Unzip and install

```bash
tar xzf logos-experiments.tar.gz
cd logos-experiments
pip install -e ".[dev]"
```

Verify:
```bash
harness-new --help
```

You should see the argparse help text. If you get "command not found", the pip install didn't put scripts on your PATH — try `python -m harness.new --help` instead.

## Step 2: Run the tests

```bash
pytest tests/ -v
```

Expected: 124 passed, 4 skipped. The 4 skips are the live Claude Code tests (they need `--run-live`). If anything fails, fix it before proceeding — the tests validate the harness itself.

## Step 3: Run the doc_search e2e manually

This proves the ticket→eval→solution loop works on your machine.

```bash
# See the ticket
cat experiments/doc_search/goal.yaml

# Run the eval (no solution yet — should fail)
harness-eval --tests experiments/doc_search/eval/test_search.py
```

You should see `pass_rate=0.0` and `FAIL`. That's correct — there's no solution yet.

## Step 4: Let Claude Code solve doc_search

This is the real test — does the agent actually follow the harness workflow?

```bash
cd logos-experiments
claude
```

In the Claude Code session:
```
Work on the doc_search experiment. Read goal.yaml and the eval,
build a solution, run the eval, and write a journal entry.
```

Watch what happens. After it finishes:

```bash
# Did it create a solution?
cat experiments/doc_search/workspace/solution.py

# Does the solution pass?
harness-eval --tests experiments/doc_search/eval/test_search.py

# Did it write a journal entry?
harness-journal doc_search summary
```

If all three check out, the harness works. If the agent didn't journal or didn't run the eval on its own, you know which parts of the CLAUDE.md need tuning.

## Step 5: Live test (optional but recommended)

If you want the pytest suite to verify this automatically:

```bash
pytest tests/test_live.py -v --run-live
```

This launches `claude --print` as a subprocess, gives it the doc_search ticket, and checks that a solution exists, passes the eval, and a journal entry was written. Takes 2-5 minutes.

## Step 6: Set up for V-JEPA → CLIP

Now the real experiment. First, verify the eval works:

```bash
# Check what data exists (none yet)
python experiments/vjepa_clip_alignment/eval/test_alignment.py \
    --projector /dev/null --discover-only

# Test the eval pipeline with synthetic data
# (you need a dummy projector for this — create a minimal one)
cat > /tmp/dummy_projector.py << 'EOF'
import numpy as np
def load(checkpoint_path=None):
    return None
def project(state, vjepa_embeddings):
    d_out = 512
    rng = np.random.RandomState(0)
    W = rng.randn(vjepa_embeddings.shape[1], d_out).astype(np.float32) * 0.01
    return vjepa_embeddings @ W
EOF

python experiments/vjepa_clip_alignment/eval/test_alignment.py \
    --projector /tmp/dummy_projector.py --synthetic --verbose
```

You should see `INTERFACE OK` with meaningless metrics (random projector on random data). This proves the eval pipeline runs.

## Step 7: Read the constraints

```bash
cat experiments/vjepa_clip_alignment/constraints.yaml
```

This has five known failure modes from your prior work. The agent will read these before starting.

## Step 8: Enable agent teams (optional)

If you want the agent to use parallel workstreams:

```bash
# In your shell or Claude Code settings.json
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```

## Step 9: Set up RunPod MCP (if using remote GPU)

```bash
claude mcp add runpod --scope user \
    -e RUNPOD_API_KEY=your_key_here \
    -- npx -y @runpod/mcp-server@latest
```

Verify with `/mcp` inside Claude Code.

## Step 10: Launch the V-JEPA experiment

```bash
cd logos-experiments
claude
```

In the Claude Code session:
```
Work on the vjepa_clip_alignment experiment.
```

The agent should:
1. Read CLAUDE.md (operating instructions)
2. Read goal.yaml (the ticket — map V-JEPA into CLIP, cosine sim > 0.70)
3. Read constraints.yaml (five known failure modes)
4. Read the eval to understand what it expects
5. Figure out how to get data (this is the bootstrapping phase)
6. Try approaches, run eval, journal results
7. Iterate until eval passes or escalate

This is where you observe and learn. The first run will tell you what the CLAUDE.md gets right and what needs adjusting.

## What to watch for

**Good signs:**
- Agent reads goal.yaml before doing anything else
- Agent runs the eval in synthetic mode to test its solution interface
- Agent writes journal entries after attempts
- Agent reads constraints.yaml and avoids known failure modes
- Agent tries simple approaches before complex ones

**Signs the CLAUDE.md needs tuning:**
- Agent jumps straight to code without reading the ticket
- Agent doesn't run the eval
- Agent doesn't journal
- Agent ignores the constraints and hits the same bugs
- Agent gets stuck and doesn't escalate

**Signs the eval needs tuning:**
- Agent can't figure out what the eval expects
- Error messages are unhelpful
- Agent produces a valid solution that the eval rejects due to a bug in the eval

## Troubleshooting

**"command not found" for harness-* commands:**
```bash
# Use module syntax instead
python -m harness.new my_experiment --goal "..."
python -m harness.journal my_experiment summary
python -m harness.pytest_eval --tests path/to/tests
```

**Claude Code auth issues:**
```bash
# Verify auth works
claude --print "say hello"

# If using Max subscription, interactive login handles refresh:
claude login
```

**Tests fail on import:**
```bash
# Make sure you installed in dev mode
pip install -e ".[dev]"

# Verify harness is importable
python -c "from harness import find_experiments_dir; print('OK')"
```
