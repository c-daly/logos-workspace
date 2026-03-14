"""
Live end-to-end test — runs a real Claude Code instance against an experiment.

These tests launch `claude --print` as a subprocess, give it a ticket,
and verify it actually follows the harness workflow: reads the goal,
runs the eval, writes journal entries, and produces a passing solution.

Requires:
    - Claude Code CLI installed and authenticated (`claude --version`)
    - Marked with @pytest.mark.live — skipped in normal test runs

Run:
    pytest tests/test_live.py -v                    # skips if no claude CLI
    pytest tests/test_live.py -v --run-live          # actually runs them
    pytest tests/test_live.py -v --run-live -k basic # run just one
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent


# ============================================================================
# Markers and fixtures
# ============================================================================

def has_claude_cli() -> bool:
    """Check if claude CLI is available."""
    try:
        r = subprocess.run(["claude", "--version"], capture_output=True, timeout=10)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def claude_print(prompt: str, cwd: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run claude --print with a prompt in a given directory."""
    return subprocess.run(
        ["claude", "--print", prompt],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
        env={**os.environ, "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"},
    )


live = pytest.mark.skipif(
    not has_claude_cli(),
    reason="Claude CLI not available"
)


def skip_without_flag(request):
    """Skip live tests unless --run-live is passed."""
    if not request.config.getoption("--run-live", default=False):
        pytest.skip("Live tests require --run-live flag")


# ============================================================================
# Test data: the search engine problem (self-contained, fast to solve)
# ============================================================================

SEARCH_TESTS = textwrap.dedent('''\
    """Test suite for a document search engine."""
    import sys
    import importlib.util
    import pytest
    from pathlib import Path

    SOLUTION_PATH = Path(__file__).parent.parent / "workspace" / "solution.py"

    @pytest.fixture
    def engine():
        if not SOLUTION_PATH.exists():
            pytest.skip(f"No solution at {SOLUTION_PATH}")
        spec = importlib.util.spec_from_file_location("solution", SOLUTION_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        state = mod.load()
        return mod, state

    class TestBasicSearch:
        def test_returns_list(self, engine):
            mod, state = engine
            results = mod.search(state, "python")
            assert isinstance(results, list)

        def test_result_structure(self, engine):
            mod, state = engine
            results = mod.search(state, "python")
            if results:
                r = results[0]
                assert "id" in r
                assert "score" in r
                assert "title" in r

        def test_scores_descending(self, engine):
            mod, state = engine
            results = mod.search(state, "python")
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

        def test_top_k_limit(self, engine):
            mod, state = engine
            results = mod.search(state, "the", top_k=3)
            assert len(results) <= 3

        def test_empty_query(self, engine):
            mod, state = engine
            results = mod.search(state, "")
            assert results == []

        def test_no_match(self, engine):
            mod, state = engine
            results = mod.search(state, "xyzzyplugh")
            assert results == []

    class TestRelevance:
        def test_keyword_match(self, engine):
            mod, state = engine
            results = mod.search(state, "database")
            titles = [r["title"].lower() for r in results]
            assert any("database" in t or "sql" in t for t in titles)

        def test_case_insensitive(self, engine):
            mod, state = engine
            r1 = mod.search(state, "python")
            r2 = mod.search(state, "PYTHON")
            ids1 = {r["id"] for r in r1}
            ids2 = {r["id"] for r in r2}
            assert ids1 == ids2

    class TestEdgeCases:
        def test_special_characters(self, engine):
            mod, state = engine
            mod.search(state, "hello! @#$%")

        def test_scores_positive(self, engine):
            mod, state = engine
            results = mod.search(state, "python")
            for r in results:
                assert r["score"] >= 0
''')


SEARCH_TICKET = textwrap.dedent("""\
    objective: |
      Build a document search engine that indexes a small corpus of 
      programming topic documents and returns ranked results for 
      text queries. The engine must handle edge cases like empty queries,
      special characters, and case insensitivity.

    success_criteria:
      - metric: pass_rate
        threshold: 1.0
        primary: true

    eval: eval/test_search.py

    notes: |
      The eval is a pytest suite. Your solution must be a Python module
      at workspace/solution.py that defines:
        - load(checkpoint_path=None) -> state
        - search(state, query: str, top_k: int = 5) -> list[dict]
      Each result dict needs: {"id": str, "score": float, "title": str}
      The corpus should contain 5-10 documents about programming topics.
""")


def create_search_experiment(base_dir: Path) -> Path:
    """Create a complete search engine experiment."""
    exp = base_dir / "experiments" / "doc_search"
    for d in ["eval", "workspace", "journal", "checkpoints"]:
        (exp / d).mkdir(parents=True)

    (exp / "goal.yaml").write_text(SEARCH_TICKET)
    (exp / "eval" / "test_search.py").write_text(SEARCH_TESTS)
    (exp / "status.yaml").write_text(textwrap.dedent("""\
        experiment: doc_search
        current_attempt: 0
        status: not_started
        total_attempts: 0
        last_updated: null
    """))

    # Copy CLAUDE.md and harness
    shutil.copy2(ROOT / "CLAUDE.md", base_dir / "CLAUDE.md")
    shutil.copytree(ROOT / "harness", base_dir / "harness", dirs_exist_ok=True)

    return exp


# ============================================================================
# Live tests
# ============================================================================

@live
class TestLiveBasic:
    """Basic: Can Claude Code read the experiment and produce a solution?"""

    @pytest.fixture
    def experiment(self, tmp_path, request):
        skip_without_flag(request)
        return create_search_experiment(tmp_path)

    def test_reads_goal_and_creates_solution(self, experiment):
        """Give Claude the experiment and ask it to solve the ticket."""
        result = claude_print(
            "Read experiments/doc_search/goal.yaml and the eval at "
            "experiments/doc_search/eval/test_search.py. "
            "Create a solution at experiments/doc_search/workspace/solution.py "
            "that passes all the tests. "
            "Run the eval to verify: python -m pytest experiments/doc_search/eval/ -v",
            cwd=str(experiment.parent.parent),
            timeout=180,
        )

        # The agent should have created a solution file
        solution = experiment / "workspace" / "solution.py"
        assert solution.exists(), (
            f"Agent didn't create solution.py.\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-1000:]}"
        )

        # Verify the solution actually passes
        eval_result = subprocess.run(
            [sys.executable, "-m", "pytest",
             str(experiment / "eval" / "test_search.py"), "-v"],
            capture_output=True, text=True,
            cwd=str(experiment.parent.parent),
        )

        assert eval_result.returncode == 0, (
            f"Solution doesn't pass eval.\n"
            f"Test output: {eval_result.stdout[-2000:]}"
        )


@live
class TestLiveWithJournal:
    """Can Claude Code follow the full workflow including journaling?"""

    @pytest.fixture
    def experiment(self, tmp_path, request):
        skip_without_flag(request)
        return create_search_experiment(tmp_path)

    def test_full_workflow(self, experiment):
        """Ask the agent to follow the CLAUDE.md workflow explicitly."""
        result = claude_print(
            "You are working on the experiment at experiments/doc_search/. "
            "Follow the workflow in CLAUDE.md: "
            "1. Read goal.yaml to understand the objective. "
            "2. Read the eval to understand what success looks like. "
            "3. Build a solution at experiments/doc_search/workspace/solution.py. "
            "4. Run the eval: python -m pytest experiments/doc_search/eval/ -v "
            "5. Write a journal entry in experiments/doc_search/journal/ "
            "   documenting what you built and whether the eval passed. "
            "   Use the format from CLAUDE.md.",
            cwd=str(experiment.parent.parent),
            timeout=240,
        )

        # Solution exists
        solution = experiment / "workspace" / "solution.py"
        assert solution.exists(), "Agent didn't create solution"

        # Journal entry exists
        journal_entries = list((experiment / "journal").glob("*.md"))
        journal_entries = [e for e in journal_entries if e.stem != ".gitkeep"]
        assert len(journal_entries) >= 1, (
            f"Agent didn't write a journal entry.\n"
            f"Journal dir contents: {list((experiment / 'journal').iterdir())}\n"
            f"stdout tail: {result.stdout[-1500:]}"
        )

        # Journal has substance
        entry_text = journal_entries[0].read_text()
        assert len(entry_text) > 50, "Journal entry is too short to be useful"

        # Solution passes eval
        eval_result = subprocess.run(
            [sys.executable, "-m", "pytest",
             str(experiment / "eval" / "test_search.py"), "-v", "-q"],
            capture_output=True, text=True,
        )
        assert eval_result.returncode == 0, (
            f"Solution doesn't pass eval: {eval_result.stdout[-1000:]}"
        )


@live
class TestLiveEvalAdapter:
    """Can Claude Code use the pytest eval adapter?"""

    @pytest.fixture
    def experiment(self, tmp_path, request):
        skip_without_flag(request)
        return create_search_experiment(tmp_path)

    def test_uses_pytest_eval(self, experiment):
        """Ask the agent to use the harness eval adapter."""
        result = claude_print(
            "Read experiments/doc_search/goal.yaml. "
            "Create a solution at experiments/doc_search/workspace/solution.py "
            "that passes the test suite. Then evaluate it using: "
            "python -m harness.pytest_eval "
            "--tests experiments/doc_search/eval/test_search.py --verbose "
            "Report the metrics.",
            cwd=str(experiment.parent.parent),
            timeout=180,
        )

        # Solution should exist
        assert (experiment / "workspace" / "solution.py").exists()

        # Run the eval adapter ourselves to verify
        eval_result = subprocess.run(
            [sys.executable, "-m", "harness.pytest_eval",
             "--tests", str(experiment / "eval" / "test_search.py")],
            capture_output=True, text=True,
            cwd=str(experiment.parent.parent),
        )

        assert "[EVAL] PASS" in eval_result.stdout, (
            f"Solution doesn't pass: {eval_result.stdout}"
        )


@live
class TestLiveSelfDiagnosis:
    """The 'point it at its own tests' capability."""

    @pytest.fixture
    def experiment(self, tmp_path, request):
        skip_without_flag(request)
        exp = tmp_path / "experiments" / "fix_harness"
        for d in ["eval", "workspace", "journal", "checkpoints"]:
            (exp / d).mkdir(parents=True)

        # The ticket: make the harness's own tests pass
        (exp / "goal.yaml").write_text(textwrap.dedent(f"""\
            objective: |
              The experiment harness test suite has failures.
              Fix the code in the harness/ directory to make all tests pass.

            success_criteria:
              - metric: pass_rate
                threshold: 1.0
                primary: true

            eval: "{ROOT / 'tests'}"

            notes: |
              Run: python -m pytest {ROOT / 'tests'} -v
              Fix any failures in the harness/ modules.
        """))

        (exp / "status.yaml").write_text("status: not_started\n")

        shutil.copy2(ROOT / "CLAUDE.md", tmp_path / "CLAUDE.md")
        shutil.copytree(ROOT / "harness", tmp_path / "harness", dirs_exist_ok=True)

        return exp

    def test_self_diagnosis_ticket_is_valid(self, experiment):
        """Just verify the self-referential ticket parses correctly."""
        goal = yaml.safe_load((experiment / "goal.yaml").read_text())
        assert "harness" in goal["objective"].lower()
        assert "pass_rate" in goal["success_criteria"][0]["metric"]
