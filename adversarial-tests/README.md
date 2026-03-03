# Adversarial Test Rewrite System

Three-agent adversarial test suite redesign driven by a ralph loop.

## Usage

From any LOGOS repo root:

    /ralph-loop "$(cat /Users/cdaly/projects/LOGOS/adversarial-tests/prompts/orchestrator.md)" --max-iterations 20 --completion-promise "TESTS COMPLETE"

Before running, source a repo config:

    source /Users/cdaly/projects/LOGOS/adversarial-tests/config/hermes.env

## Design

See docs/plans/2026-02-21-adversarial-test-rewrite-design.md for the full design document.

## Agents

- **Writer** (prompts/writer.md): Designs and implements optimal test suite
- **Adversary** (prompts/adversary.md): Finds weaknesses, produces risk map
- **Judge** (prompts/judge.md): Validates critique, makes authoritative ruling

## Recommended Repo Order

1. hermes (smallest, good pilot)
2. talos (small, high threshold)
3. apollo Python backend
4. sophia (largest, most churn)
5. logos (shared foundry)
6. apollo webapp (vitest, separate)
