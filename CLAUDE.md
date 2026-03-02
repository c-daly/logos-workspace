# CLAUDE.md - Project Guidelines for Claude Code

## Ecosystem Overview

LOGOS is a cognitive architecture composed of **five tightly coupled repositories**:

| Repo | Purpose | AGENTS.md |
|------|---------|-----------|
| **logos** | Foundry—contracts, ontology, SDKs, shared tooling | `logos/AGENTS.md` |
| **sophia** | Non-linguistic cognitive core (Orchestrator, CWM, Planner) | `sophia/AGENTS.md` |
| **hermes** | Stateless language & embedding utility (STT, TTS, NLP) | `hermes/AGENTS.md` |
| **talos** | Hardware abstraction layer for sensors/actuators | `talos/AGENTS.md` |
| **apollo** | Thin client UI and command layer | `apollo/AGENTS.md` |

**Each repo has its own AGENTS.md** with repo-specific guidance. Read it before working in that repo.

---

## Token Efficiency: Use Subagents

Spawn subagents (via Task tool) for work that doesn't need full conversation context. Each subagent has its own context window—only a summary returns to the main conversation.

### Use Subagents For
1. **Code exploration** - "Find all files that use X", "How does Y work"
2. **Cross-repo searches** - Searching across multiple repos
3. **Research tasks** - Looking up docs, finding patterns
4. **Large refactors** - Mechanical changes across many files
5. **Test execution** - Running suites and analyzing results
6. **Documentation generation** - Docstrings, READMEs

### Do NOT Use Subagents For
- Quick single-file edits (do directly)
- Tasks requiring conversation history
- Interactive debugging with user
- Tasks with <3 steps

### Subagent Types
- `Explore` - Fast codebase exploration, file searches
- `general-purpose` - Complex multi-step tasks, code changes
- `Plan` - Architecture planning, implementation strategy

                                                                                                            │
│ ## Sub-agent Usage                                                                                           │
│                                                                                                              │
│ Use specialized agents proactively for better results:                                                       │
│                                                                                                              │
│ ### Exploration (use liberally)                                                                              │
│ - Before modifying unfamiliar code, spawn an Explore agent to understand context                             │
│ - For questions like "where is X handled?" or "how does Y work?", use Explore with appropriate thoroughness  │
│ - When unsure about codebase patterns, explore first rather than guessing                                    │
│                                                                                                              │
│ ### Planning (use for non-trivial changes)                                                                   │
│ - Before implementing features touching 3+ files, use Plan agent to design approach                          │
│ - For refactoring tasks, plan first to identify all affected areas                                           │
│ - Get user approval on plans before executing                                                                │
│                                                                                                              │
│ ### Parallel Agents                                                                                          │
│ - When multiple independent investigations are needed, launch them simultaneously                            │
│ - Example: researching both frontend and backend changes in parallel                                         │
│ - Use background agents for long-running tasks while continuing other work                                   │
│                                                                                                              │
│ ### Research Tasks                                                                                           │
│ - Use general-purpose agent for complex searches requiring multiple iterations                               │
│ - Delegate documentation lookups and API research to agents                                                  │
│ - Keep main context focused on implementation                                                                │
│                                                                                                              │
│ ## Workflow Patterns                                                                                         │
│                                                                                                              │
│ ### Before Coding                                                                                            │
│ 1. Explore relevant areas of codebase                                                                        │
│ 2. Plan implementation for complex changes                                                                   │
│ 3. Verify understanding with user if ambiguous                                                               │
│                                                                                                              │
│ ### During Implementation                                                                                    │
│ - Track progress with TodoWrite for multi-step tasks                                                         │
│ - Run tests/builds in background while continuing work                                                       │
│ - Spawn agents to investigate errors rather than guessing                                                    │
│                                                                                                              │
│ ### Code Review                                                                                              │
│ - After significant changes, use Explore to verify no regressions in related code                            │
│ - Check for similar patterns elsewhere that might need updates                                               │
│                                                    
---

## Cross-Repo Workflow

### Contract changes flow downstream
```
logos (contracts) → sophia, hermes, talos, apollo
```
If you need to change an API, update the contract in logos first.

### Port allocation
| Repo | API | Neo4j | Milvus |
|------|-----|-------|--------|
| hermes | 17000 | 7474/7687 | 19530 |
| apollo | 27000 | 7474/7687 | 19530 |
| logos | 37000 | 7474/7687 | 19530 |
| sophia | 47000 | 7474/7687 | 19530 |
| talos | 57000 | 7474/7687 | 19530 |

Infrastructure (Neo4j, Milvus) runs on default ports — shared across all repos.

### Shared config
Use `logos_config` package for environment, ports, settings across all repos.

---

## Current Experiments

### V-JEPA2 Fine-tuning
- **Notebook**: `logos/experiments/notebooks/jepa_finetune_lora_v2.ipynb`
- **Purpose**: Align V-JEPA2 video embeddings to CLIP semantic space
- **Memory target**: 8GB VRAM (batch_size=1, float16, gradient checkpointing)

---

  ## Paper Tracking

-  LOGOS has 13 candidate academic papers (see `LOGOS_Implementation_Spec.md` Appendix C). Paper logs live in the Obsidian vault at `10-projects/LOGOS/papers/`. 
-  When implementation work produces results, measurements, or observations relevant to a paper, prompt the user: "This looks relevant to paper C.X — want to log it?" The user should
-  not have to remember; you should notice and ask.
---

## Common Commands

```bash
# Clear notebook outputs (fixes VS Code caching issues)
jupyter nbconvert --clear-output --inplace NOTEBOOK.ipynb

# Or via Python
python -c "import json; nb=json.load(open('NB.ipynb')); [c.update({'outputs':[],'execution_count':None}) for c in nb['cells'] if c['cell_type']=='code']; json.dump(nb,open('NB.ipynb','w'),indent=1)"

# Lint (all repos use ruff + mypy)
poetry run ruff check --fix .
poetry run ruff format .
poetry run mypy src/

# Run tests
poetry run pytest tests/
```

---

## Code Standards (all repos)

- **Type hints**: Required for public functions
- **Docstrings**: Required for complex functions
- **Small functions**: Prefer composable over monolithic
- **Backward compat**: Maintain unless explicitly breaking
- **Security**: Never log secrets/PII, sanitize inputs

See `logos/docs/TESTING_STANDARDS.md` and `logos/docs/GIT_PROJECT_STANDARDS.md` for full standards.
