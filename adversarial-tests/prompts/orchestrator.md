# Adversarial Test Rewrite Orchestrator

You are orchestrating an adversarial test suite redesign. Each iteration, you dispatch three subagents (Writer + Adversary in parallel, then Judge) and check exit criteria.

## Configuration

Before this loop was started, the following environment was configured. Read these values from the shell environment or from the config sourced before invocation:
- REPO: The repository name
- REPO_ROOT: Absolute path to the repo
- COVERAGE_TARGET: Minimum coverage percentage
- MUTATION_TARGET: Minimum mutation score percentage
- TEST_COMMAND: Command to run tests
- MUTATION_COMMAND: Command to run mutation testing
- MUTATION_RESULTS_COMMAND: Command to get mutation results
- FRAMEWORK: Test framework (pytest / vitest)

If you cannot determine these, read the config files at /Users/cdaly/projects/LOGOS/adversarial-tests/config/ and find the one matching this repository.

## Prompt Files

The agent prompts are at:
- Writer: /Users/cdaly/projects/LOGOS/adversarial-tests/prompts/writer.md
- Adversary: /Users/cdaly/projects/LOGOS/adversarial-tests/prompts/adversary.md
- Judge: /Users/cdaly/projects/LOGOS/adversarial-tests/prompts/judge.md

Read these files and substitute {REPO}, {REPO_ROOT}, {FRAMEWORK}, {TEST_COMMAND}, {MUTATION_COMMAND}, {MUTATION_RESULTS_COMMAND}, {COVERAGE_TARGET}, {MUTATION_TARGET} with the actual values.

## Each Iteration

### Step 1: Read current state

Read these files at REPO_ROOT (they may not exist in iteration 1):
- JUDGMENT.md
- RESULTS.md
- CRITIQUE.md
- REBUTTAL.md
- EVALUATION.md

Check for **consensus overrides**: if BOTH REBUTTAL.md and EVALUATION.md exist and flag the SAME Judge ruling as incorrect, write CONSENSUS_OVERRIDES.md listing those overrides.

Check `git log --oneline -5` for recent changes.

Determine the current iteration number from the ralph loop state or from counting commits.

### Step 2: Spawn Writer + Adversary (parallel)

Use the Task tool to spawn TWO general-purpose subagents simultaneously:

**Writer subagent:**
- Read the Writer prompt from /Users/cdaly/projects/LOGOS/adversarial-tests/prompts/writer.md
- Substitute all {PLACEHOLDERS} with actual config values
- Include the current JUDGMENT.md contents (if exists) in the prompt
- Include CONSENSUS_OVERRIDES.md contents (if exists) in the prompt
- Tell it the current iteration number
- Set mode to "bypassPermissions" so it can run tests and mutations freely

**Adversary subagent:**
- Read the Adversary prompt from /Users/cdaly/projects/LOGOS/adversarial-tests/prompts/adversary.md
- Substitute all {PLACEHOLDERS} with actual config values
- Include the current JUDGMENT.md contents (if exists) in the prompt
- Tell it the current iteration number
- Set mode to "bypassPermissions"

Wait for BOTH to complete before proceeding.

### Step 3: Spawn Judge

After both Writer and Adversary complete:

Use the Task tool to spawn ONE general-purpose subagent:

**Judge subagent:**
- Read the Judge prompt from /Users/cdaly/projects/LOGOS/adversarial-tests/prompts/judge.md
- Substitute all {PLACEHOLDERS} with actual config values
- Include the contents of: CRITIQUE.md, RESULTS.md, TEST_PLAN.md, RISK_MAP.md
- Include REBUTTAL.md and EVALUATION.md contents if they exist
- Include previous JUDGMENT.md if it exists (for precedent)
- Include CONSENSUS_OVERRIDES.md if it exists
- Tell it the current iteration number
- Set mode to "bypassPermissions"

Wait for completion.

### Step 4: Evaluate exit criteria

Read RESULTS.md. Extract:
1. Coverage percentage
2. Mutation score percentage
3. Writer confidence (N%)

Read CRITIQUE.md. Extract:
4. Adversary confidence (N%)

Read JUDGMENT.md. Extract:
5. Judge verdict (PASS / FAIL)
6. Judge's calibrated confidence (N%)

Check ALL of:
- [ ] Judge verdict = PASS
- [ ] Coverage >= COVERAGE_TARGET
- [ ] Mutation score >= MUTATION_TARGET
- [ ] Confidence convergence: max(writer, adversary, judge confidence) - min(writer, adversary, judge confidence) < 10
- [ ] No unaddressed consensus overrides in CONSENSUS_OVERRIDES.md

**If ALL criteria met:**

Summarize the final state:
- Total iterations
- Final coverage
- Final mutation score
- Final confidence scores
- Key findings that were addressed

Then output: <promise>TESTS COMPLETE</promise>

**If NOT all criteria met:**

Summarize what's still needed:
- Which criteria failed
- Key priorities from JUDGMENT.md
- Confidence gap analysis

The loop will continue with the next iteration.
