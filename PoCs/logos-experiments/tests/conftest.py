import sys
from pathlib import Path

# Add project root to path so `import harness` works
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_addoption(parser):
    """Add --run-live flag for live Claude Code tests."""
    parser.addoption("--run-live", action="store_true", default=False,
                     help="Run live Claude Code tests (requires auth)")
