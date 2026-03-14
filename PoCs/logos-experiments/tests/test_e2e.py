"""
End-to-end test: Full experiment lifecycle.

Domain: Document search engine (completely unrelated to ML/embeddings).

Simulates the agent's journey:
1. Ingest a ticket
2. Read the eval (test suite the solution must pass)
3. Submit a naive solution → eval fails
4. Record failure in journal
5. Submit a correct solution → eval passes
6. Record success in journal
7. Verify final state

This proves: ticket ingestion, eval discovery, pass/fail detection,
journal accumulation, status tracking, and the pytest eval adapter.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent


# ============================================================================
# The Search Engine Test Suite (this IS the eval)
# ============================================================================

SEARCH_TESTS = textwrap.dedent('''\
    """
    Test suite for the document search engine.
    
    The solution module must define:
        def load(checkpoint_path=None) -> Any
        def search(state, query: str, top_k: int = 5) -> list[dict]
    
    Each result dict must have: {"id": str, "score": float, "title": str}
    Results must be sorted by score descending.
    """
    import sys
    import importlib.util
    import pytest
    from pathlib import Path


    # Load the solution module
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


    # === Core functionality ===

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

        def test_empty_query_returns_empty(self, engine):
            mod, state = engine
            results = mod.search(state, "")
            assert results == []

        def test_no_match_returns_empty(self, engine):
            mod, state = engine
            results = mod.search(state, "xyzzyplugh")
            assert results == []


    # === Relevance ===

    class TestRelevance:
        def test_exact_title_match_ranks_first(self, engine):
            """A query matching a document title exactly should rank it first."""
            mod, state = engine
            results = mod.search(state, "Quick Sort Algorithm")
            assert len(results) > 0
            assert "quick sort" in results[0]["title"].lower()

        def test_keyword_match(self, engine):
            """Documents containing the query term should appear."""
            mod, state = engine
            results = mod.search(state, "database")
            titles = [r["title"].lower() for r in results]
            assert any("database" in t or "sql" in t for t in titles)

        def test_partial_match(self, engine):
            """Partial word matches should work."""
            mod, state = engine
            results = mod.search(state, "sort")
            titles = [r["title"].lower() for r in results]
            assert any("sort" in t for t in titles)

        def test_case_insensitive(self, engine):
            """Search should be case insensitive."""
            mod, state = engine
            r1 = mod.search(state, "python")
            r2 = mod.search(state, "PYTHON")
            r3 = mod.search(state, "Python")
            ids1 = {r["id"] for r in r1}
            ids2 = {r["id"] for r in r2}
            ids3 = {r["id"] for r in r3}
            assert ids1 == ids2 == ids3

        def test_multi_word_query(self, engine):
            """Multi-word queries should match documents containing those words."""
            mod, state = engine
            results = mod.search(state, "binary search tree")
            assert len(results) > 0


    # === Edge cases ===

    class TestEdgeCases:
        def test_special_characters_dont_crash(self, engine):
            mod, state = engine
            # Should not raise
            mod.search(state, "hello! @#$% world")

        def test_very_long_query(self, engine):
            mod, state = engine
            mod.search(state, "word " * 1000)

        def test_numeric_query(self, engine):
            mod, state = engine
            results = mod.search(state, "42")
            assert isinstance(results, list)

        def test_scores_are_positive(self, engine):
            mod, state = engine
            results = mod.search(state, "python")
            for r in results:
                assert r["score"] >= 0
''')


# ============================================================================
# Solutions (simulating agent attempts)
# ============================================================================

# Attempt 1: Naive — has several bugs
NAIVE_SOLUTION = textwrap.dedent('''\
    """Naive search engine — has bugs."""
    
    DOCUMENTS = [
        {"id": "1", "title": "Introduction to Python Programming",
         "body": "Python is a versatile programming language used for web development, data science, and automation."},
        {"id": "2", "title": "Quick Sort Algorithm",
         "body": "Quick sort is a divide-and-conquer sorting algorithm with average O(n log n) complexity."},
        {"id": "3", "title": "Database Design with SQL",
         "body": "SQL databases use structured query language for managing relational data."},
        {"id": "4", "title": "Binary Search Tree Implementation",
         "body": "A binary search tree is a data structure where each node has at most two children."},
        {"id": "5", "title": "Machine Learning with Python",
         "body": "Machine learning algorithms learn patterns from data to make predictions."},
        {"id": "6", "title": "Web Development with JavaScript",
         "body": "JavaScript is the language of the web, enabling interactive user interfaces."},
        {"id": "7", "title": "Understanding Big O Notation",
         "body": "Big O notation describes the upper bound of algorithm complexity, helping compare efficiency."},
        {"id": "8", "title": "Graph Algorithms and Applications",
         "body": "Graphs model relationships between entities. Common algorithms include BFS, DFS, and Dijkstra."},
    ]

    def load(checkpoint_path=None):
        return DOCUMENTS

    def search(state, query, top_k=5):
        # BUG: doesn't handle empty query
        # BUG: case sensitive matching
        # BUG: doesn't sort by score
        results = []
        for doc in state:
            if query in doc["title"] or query in doc["body"]:
                results.append({
                    "id": doc["id"],
                    "score": 1.0,
                    "title": doc["title"],
                })
        return results[:top_k]
''')

# Attempt 2: Fixed — passes all tests
CORRECT_SOLUTION = textwrap.dedent('''\
    """Document search engine — TF-IDF style scoring."""
    
    import re
    import math
    from collections import Counter

    DOCUMENTS = [
        {"id": "1", "title": "Introduction to Python Programming",
         "body": "Python is a versatile programming language used for web development, data science, and automation."},
        {"id": "2", "title": "Quick Sort Algorithm",
         "body": "Quick sort is a divide-and-conquer sorting algorithm with average O(n log n) complexity."},
        {"id": "3", "title": "Database Design with SQL",
         "body": "SQL databases use structured query language for managing relational data."},
        {"id": "4", "title": "Binary Search Tree Implementation",
         "body": "A binary search tree is a data structure where each node has at most two children."},
        {"id": "5", "title": "Machine Learning with Python",
         "body": "Machine learning algorithms learn patterns from data to make predictions."},
        {"id": "6", "title": "Web Development with JavaScript",
         "body": "JavaScript is the language of the web, enabling interactive user interfaces."},
        {"id": "7", "title": "Understanding Big O Notation",
         "body": "Big O notation describes the upper bound of algorithm complexity, helping compare efficiency."},
        {"id": "8", "title": "Graph Algorithms and Applications",
         "body": "Graphs model relationships between entities. Common algorithms include BFS, DFS, and Dijkstra."},
    ]

    def _tokenize(text):
        return re.findall(r\'[a-z0-9]+\', text.lower())

    def load(checkpoint_path=None):
        # Build inverted index
        docs = DOCUMENTS
        index = {}  # term -> {doc_id: count}
        doc_lengths = {}
        
        for doc in docs:
            tokens = _tokenize(doc["title"]) + _tokenize(doc["body"])
            doc_lengths[doc["id"]] = len(tokens)
            # Title terms get 3x weight
            title_tokens = _tokenize(doc["title"])
            counts = Counter(tokens)
            for t in title_tokens:
                counts[t] += 2  # extra weight
            
            for term, count in counts.items():
                if term not in index:
                    index[term] = {}
                index[term][doc["id"]] = count
        
        return {
            "docs": {d["id"]: d for d in docs},
            "index": index,
            "doc_lengths": doc_lengths,
            "n_docs": len(docs),
        }

    def search(state, query, top_k=5):
        if not query or not query.strip():
            return []
        
        terms = _tokenize(query)
        if not terms:
            return []
        
        scores = {}
        for term in terms:
            postings = state["index"].get(term, {})
            if not postings:
                continue
            # IDF
            idf = math.log(state["n_docs"] / (1 + len(postings)))
            for doc_id, tf in postings.items():
                tf_norm = tf / (state["doc_lengths"].get(doc_id, 1))
                scores[doc_id] = scores.get(doc_id, 0) + tf_norm * idf
        
        if not scores:
            return []
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in ranked:
            doc = state["docs"][doc_id]
            results.append({
                "id": doc_id,
                "score": round(score, 4),
                "title": doc["title"],
            })
        
        return results
''')


# ============================================================================
# E2E Test
# ============================================================================

class TestEndToEnd:
    """Full lifecycle: ticket → attempt → fail → journal → fix → pass → journal."""

    @pytest.fixture
    def experiment(self, tmp_path):
        """Create a fresh experiment in a temp directory."""
        exp = tmp_path / "experiments" / "doc_search"
        for d in ["eval", "workspace", "journal", "checkpoints"]:
            (exp / d).mkdir(parents=True)

        # Write the ticket
        (exp / "goal.yaml").write_text(textwrap.dedent("""\
            objective: |
              Build a document search engine that indexes a small corpus
              and returns ranked results for text queries.

            success_criteria:
              - metric: pass_rate
                threshold: 1.0
                primary: true
              - metric: tests_total
                threshold: 15
                description: Must have at least 15 meaningful tests

            eval: eval/test_search.py

            notes: |
              The eval is a pytest suite. The solution module must define
              load() and search() functions. See eval for details.
        """))

        (exp / "status.yaml").write_text(textwrap.dedent("""\
            experiment: doc_search
            current_attempt: 0
            status: not_started
            total_attempts: 0
            last_updated: null
        """))

        # Write the test suite as the eval
        (exp / "eval" / "test_search.py").write_text(SEARCH_TESTS)

        return exp

    def test_ticket_ingestible(self, experiment):
        """Verify the ticket is valid YAML with required fields."""
        goal = yaml.safe_load((experiment / "goal.yaml").read_text())
        assert "objective" in goal
        assert "success_criteria" in goal
        assert "eval" in goal
        assert "search engine" in goal["objective"].lower()

    def test_eval_discoverable(self, experiment):
        """Verify the eval script exists where the ticket says."""
        goal = yaml.safe_load((experiment / "goal.yaml").read_text())
        eval_path = experiment / goal["eval"]
        assert eval_path.exists()

    def test_naive_attempt_fails(self, experiment):
        """Submit the buggy solution. Eval should fail."""
        # Write naive solution
        (experiment / "workspace" / "solution.py").write_text(NAIVE_SOLUTION)

        # Run eval via pytest adapter
        result = subprocess.run(
            [sys.executable, "-m", "harness.pytest_eval",
             "--tests", str(experiment / "eval" / "test_search.py"),
             "--verbose"],
            capture_output=True, text=True,
            cwd=str(ROOT),
        )

        assert result.returncode == 1
        assert "[EVAL] FAIL" in result.stdout
        assert "tests_failed=" in result.stdout

        # Parse metrics
        metrics = {}
        for line in result.stdout.splitlines():
            if line.startswith("[METRIC]"):
                key, val = line[9:].split("=", 1)
                metrics[key] = float(val)

        assert metrics["tests_failed"] > 0
        assert metrics["pass_rate"] < 1.0

    def test_correct_attempt_passes(self, experiment):
        """Submit the correct solution. Eval should pass."""
        (experiment / "workspace" / "solution.py").write_text(CORRECT_SOLUTION)

        result = subprocess.run(
            [sys.executable, "-m", "harness.pytest_eval",
             "--tests", str(experiment / "eval" / "test_search.py")],
            capture_output=True, text=True,
            cwd=str(ROOT),
        )

        assert result.returncode == 0
        assert "[EVAL] PASS" in result.stdout

        metrics = {}
        for line in result.stdout.splitlines():
            if line.startswith("[METRIC]"):
                key, val = line[9:].split("=", 1)
                metrics[key] = float(val)

        assert metrics["pass_rate"] == 1.0
        assert metrics["tests_total"] >= 15

    def test_full_lifecycle(self, experiment):
        """
        Simulate the full agent loop:
        1. Read ticket
        2. Submit naive solution → FAIL
        3. Journal the failure
        4. Submit correct solution → PASS
        5. Journal the success
        6. Verify final state
        """
        sys.path.insert(0, str(ROOT))
        from harness.journal import Journal

        j = Journal.__new__(Journal)
        j.experiment_name = "doc_search"
        j.journal_dir = experiment / "journal"

        # --- Step 1: Read ticket ---
        goal = yaml.safe_load((experiment / "goal.yaml").read_text())
        assert "search engine" in goal["objective"].lower()

        # --- Step 2: Attempt 1 (naive) ---
        (experiment / "workspace" / "solution.py").write_text(NAIVE_SOLUTION)

        r1 = subprocess.run(
            [sys.executable, "-m", "harness.pytest_eval",
             "--tests", str(experiment / "eval" / "test_search.py"),
             "--output-json", str(experiment / "workspace" / "eval_result.json")],
            capture_output=True, text=True,
            cwd=str(ROOT),
        )
        result1 = json.loads((experiment / "workspace" / "eval_result.json").read_text())

        # --- Step 3: Journal the failure ---
        j.add_entry(
            title="Naive substring search",
            hypothesis="Simple string matching on title and body",
            changes="Created workspace/solution.py with basic search",
            results=f"pass_rate={result1['pass_rate']}, "
                    f"{result1['tests_failed']} tests failing",
            diagnosis="Case-sensitive matching fails case insensitivity tests. "
                      "No scoring differentiation — all results score 1.0. "
                      "Empty query not handled.",
            next_direction="Implement proper tokenization, TF-IDF scoring, "
                          "and case-insensitive matching",
        )

        assert len(j.list_entries()) == 1
        assert "Naive" in j.list_entries()[0].read_text()

        # --- Step 4: Attempt 2 (correct) ---
        (experiment / "workspace" / "solution.py").write_text(CORRECT_SOLUTION)

        r2 = subprocess.run(
            [sys.executable, "-m", "harness.pytest_eval",
             "--tests", str(experiment / "eval" / "test_search.py"),
             "--output-json", str(experiment / "workspace" / "eval_result.json")],
            capture_output=True, text=True,
            cwd=str(ROOT),
        )
        result2 = json.loads((experiment / "workspace" / "eval_result.json").read_text())

        # --- Step 5: Journal the success ---
        j.add_entry(
            title="TF-IDF search engine",
            hypothesis="Proper tokenization + TF-IDF scoring + title boosting",
            changes="Rewrote workspace/solution.py with inverted index",
            results=f"pass_rate={result2['pass_rate']}, "
                    f"all {result2['tests_total']} tests passing",
            diagnosis="All tests pass. TF-IDF with title boosting provides "
                      "good relevance ranking.",
            next_direction="Done — goal achieved",
        )

        # --- Step 6: Verify final state ---

        # Journal has both entries
        entries = j.list_entries()
        assert len(entries) == 2
        assert "001_" in entries[0].name
        assert "002_" in entries[1].name

        # Failure is in failures summary
        failures = j.failures_summary()
        assert "Case-sensitive" in failures or "fail" in failures.lower()

        # Success summary has both
        summary = j.summary()
        assert "Naive" in summary
        assert "TF-IDF" in summary

        # Eval results are correct
        assert result1["pass_rate"] < 1.0
        assert result1["tests_failed"] > 0
        assert result2["pass_rate"] == 1.0
        assert result2["tests_failed"] == 0

        # Status update would happen here in real flow
        status = yaml.safe_load((experiment / "status.yaml").read_text())
        assert status["status"] == "not_started"  # Agent would update this


class TestPytestEvalAdapter:
    """Test the generic pytest eval adapter itself."""

    @pytest.fixture
    def passing_tests(self, tmp_path):
        (tmp_path / "test_pass.py").write_text(textwrap.dedent("""\
            def test_one(): assert True
            def test_two(): assert 1 + 1 == 2
            def test_three(): assert "hello".upper() == "HELLO"
        """))
        return tmp_path / "test_pass.py"

    @pytest.fixture
    def failing_tests(self, tmp_path):
        (tmp_path / "test_fail.py").write_text(textwrap.dedent("""\
            def test_pass(): assert True
            def test_fail(): assert False
            def test_also_fail(): assert 1 == 2
        """))
        return tmp_path / "test_fail.py"

    def test_all_passing(self, passing_tests):
        result = subprocess.run(
            [sys.executable, "-m", "harness.pytest_eval",
             "--tests", str(passing_tests)],
            capture_output=True, text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0
        assert "[EVAL] PASS" in result.stdout
        assert "pass_rate=1.0" in result.stdout
        assert "tests_total=3" in result.stdout

    def test_some_failing(self, failing_tests):
        result = subprocess.run(
            [sys.executable, "-m", "harness.pytest_eval",
             "--tests", str(failing_tests)],
            capture_output=True, text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 1
        assert "[EVAL] FAIL" in result.stdout
        assert "tests_failed=2" in result.stdout

    def test_threshold(self, failing_tests):
        """With a lower threshold, partial passes can still PASS."""
        result = subprocess.run(
            [sys.executable, "-m", "harness.pytest_eval",
             "--tests", str(failing_tests),
             "--threshold", "0.3"],
            capture_output=True, text=True,
            cwd=str(ROOT),
        )
        # 1/3 passing = 0.333, threshold 0.3 → should pass
        assert result.returncode == 0
        assert "[EVAL] PASS" in result.stdout

    def test_json_output(self, passing_tests, tmp_path):
        out = tmp_path / "result.json"
        subprocess.run(
            [sys.executable, "-m", "harness.pytest_eval",
             "--tests", str(passing_tests),
             "--output-json", str(out)],
            capture_output=True, text=True,
            cwd=str(ROOT),
        )
        data = json.loads(out.read_text())
        assert data["tests_passed"] == 3
        assert data["pass_rate"] == 1.0

    def test_nonexistent_tests(self):
        result = subprocess.run(
            [sys.executable, "-m", "harness.pytest_eval",
             "--tests", "/nonexistent/tests.py"],
            capture_output=True, text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 1
