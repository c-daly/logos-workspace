"""
Journal Manager — Append-only experiment log for agent memory.

Usage:
    python -m harness.journal vjepa_clip_alignment summary
    python -m harness.journal vjepa_clip_alignment add
    python -m harness.journal vjepa_clip_alignment failures
"""

import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Optional


from harness import find_experiments_dir

EXPERIMENTS_DIR = find_experiments_dir()

ENTRY_TEMPLATE = """# Attempt {number:03d}: {title}
**Date:** {date}

**Hypothesis:** {hypothesis}

**Changes:** {changes}

**Results:** {results}

**Diagnosis:** {diagnosis}

**Next direction:** {next_direction}

**Training time:** {training_time}
"""


class Journal:
    """Manages the experiment journal — append-only log of attempts."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.journal_dir = EXPERIMENTS_DIR / experiment_name / "journal"
        self.journal_dir.mkdir(parents=True, exist_ok=True)

    def list_entries(self) -> list[Path]:
        """List all journal entries in order."""
        entries = sorted(self.journal_dir.glob("*.md"))
        return [e for e in entries if e.stem != ".gitkeep"]

    def next_number(self) -> int:
        """Get the next attempt number."""
        entries = self.list_entries()
        if not entries:
            return 1
        # Extract number from filename like "001_baseline.md"
        numbers = []
        for e in entries:
            m = re.match(r"(\d+)", e.stem)
            if m:
                numbers.append(int(m.group(1)))
        return max(numbers, default=0) + 1

    def add_entry(
        self,
        title: str,
        hypothesis: str,
        changes: str,
        results: str,
        diagnosis: str,
        next_direction: str,
        training_time: str = "unknown",
    ) -> Path:
        """Add a new journal entry."""
        number = self.next_number()
        slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
        filename = f"{number:03d}_{slug}.md"
        
        content = ENTRY_TEMPLATE.format(
            number=number,
            title=title,
            date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            hypothesis=hypothesis,
            changes=changes,
            results=results,
            diagnosis=diagnosis,
            next_direction=next_direction,
            training_time=training_time,
        )

        path = self.journal_dir / filename
        path.write_text(content)
        return path

    def add_raw(self, title: str, content: str) -> Path:
        """Add a raw markdown journal entry."""
        number = self.next_number()
        slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
        filename = f"{number:03d}_{slug}.md"
        path = self.journal_dir / filename
        path.write_text(content)
        return path

    def summary(self) -> str:
        """Generate a summary of all experiments."""
        entries = self.list_entries()
        if not entries:
            return "No experiments recorded yet."

        lines = [f"# Experiment Journal: {self.experiment_name}",
                 f"Total attempts: {len(entries)}", ""]

        for entry_path in entries:
            content = entry_path.read_text()
            # Extract title
            title_match = re.search(r'^# (.+)', content, re.MULTILINE)
            title = title_match.group(1) if title_match else entry_path.stem

            # Extract results summary
            results_match = re.search(
                r'\*\*Results:\*\*\s*(.+?)(?=\n\*\*|\Z)', content, re.DOTALL
            )
            results = results_match.group(1).strip()[:200] if results_match else "N/A"

            # Extract diagnosis
            diag_match = re.search(
                r'\*\*Diagnosis:\*\*\s*(.+?)(?=\n\*\*|\Z)', content, re.DOTALL
            )
            diagnosis = diag_match.group(1).strip()[:200] if diag_match else "N/A"

            lines.append(f"## {title}")
            lines.append(f"**Results:** {results}")
            lines.append(f"**Diagnosis:** {diagnosis}")
            lines.append("")

        return "\n".join(lines)

    def failures_summary(self) -> str:
        """Extract just the failure modes — what NOT to repeat."""
        entries = self.list_entries()
        if not entries:
            return "No experiments recorded yet."

        lines = ["# Known Failure Modes (from journal)", ""]

        for entry_path in entries:
            content = entry_path.read_text()
            title_match = re.search(r'^# (.+)', content, re.MULTILINE)
            title = title_match.group(1) if title_match else entry_path.stem

            diag_match = re.search(
                r'\*\*Diagnosis:\*\*\s*(.+?)(?=\n\*\*|\Z)', content, re.DOTALL
            )
            if diag_match:
                diagnosis = diag_match.group(1).strip()
                # Only include if it looks like a failure
                if any(kw in diagnosis.lower() for kw in
                       ['fail', 'nan', 'crash', 'bug', 'error', 'wrong', 'diverge',
                        'explod', 'didn\'t work', 'did not work', 'poor', 'low']):
                    lines.append(f"- **{title}:** {diagnosis[:300]}")

        if len(lines) == 2:
            lines.append("No clear failures documented yet.")

        return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m harness.journal <experiment_name> <command>")
        print("Commands: summary, failures, add")
        sys.exit(1)

    experiment_name = sys.argv[1]
    command = sys.argv[2]
    journal = Journal(experiment_name)

    if command == "summary":
        print(journal.summary())

    elif command == "failures":
        print(journal.failures_summary())

    elif command == "add":
        # Interactive entry
        print(f"Adding journal entry for: {experiment_name}")
        print(f"This will be attempt #{journal.next_number():03d}")
        print()
        title = input("Title (brief): ")
        hypothesis = input("Hypothesis: ")
        changes = input("Changes made: ")
        results = input("Results: ")
        diagnosis = input("Diagnosis: ")
        next_dir = input("Next direction: ")
        train_time = input("Training time (e.g. '45 min'): ")

        path = journal.add_entry(
            title=title,
            hypothesis=hypothesis,
            changes=changes,
            results=results,
            diagnosis=diagnosis,
            next_direction=next_dir,
            training_time=train_time,
        )
        print(f"\nSaved: {path}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
