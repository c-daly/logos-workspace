# Cognitive Loop — Implementation Handoff

## What Happened

We brainstormed, designed, wrote, reviewed, and revised the implementation plan for closing the cognitive loop (Apollo → Hermes → Sophia → Hermes → Apollo).

### Artifacts

| File | What |
|------|------|
| `docs/plans/2026-02-18-cognitive-loop-design.md` | Design doc — architecture, principles, ontology model, per-turn flow |
| `docs/plans/2026-02-18-cognitive-loop-implementation.md` | Implementation plan — 7 tasks, TDD, complete code |
| `docs/plans/cognitive-loop-spec-review.md` | External spec review (17 issues) |
| `docs/plans/cognitive-loop-spec-review-reaction.md` | Our reaction — accepted 10, rejected 6, clarified 1 |
| `~/Downloads/cognitive-loop-spec-review-revised.md` | Reviewer's revised assessment after our reaction |

### Plan Status

The implementation plan has been revised with all accepted review feedback:

- `add_edge()` uses MERGE for idempotency
- Descriptive edge names (`Paris_LOCATED_IN_France`)
- Milvus Edge collection with explicit index creation
- Neo4j indexes (uuid uniqueness, type, name, relation)
- Task 4 (Sophia Alignment) fully specified — no hand-waves
- `SPACY_TO_ONTOLOGY` mapping in Hermes ProposalBuilder
- `ENTITY_MATCH_THRESHOLD` for search-before-create in ProposalProcessor
- Partial ingestion explicitly accepted, pruning handles residuals
- Differentiated error logging in `_get_sophia_context()`
- Automated integration assertion script

## What's Next

Execute the plan. Start a new session and run:

```
/skill superpowers:executing-plans docs/plans/2026-02-18-cognitive-loop-implementation.md
```

### Task dependency graph

```
Task 1 (Edge reification — logos) → Task 2 (Query rewrite) → Task 3 (Seeder + planner)
                                  → Task 4 (Sophia alignment) → Task 6 (Proposal processor) → Task 7 (Loop closing)
                                  → Task 5 (Hermes proposal builder) ─────────────────────────→ Task 7
```

Task 1 first. Tasks 2/4/5 can parallelize after Task 1. Task 7 is last.

### Key reminders for the implementer

- Python 3.12 only (`poetry env use /usr/local/bin/python3.12`)
- Feature branch, never main
- Neo4j + Milvus via `docker compose -f logos/infra/docker-compose.hcg.dev.yml up -d`
- Read the Standards section at the top of the plan before writing code
