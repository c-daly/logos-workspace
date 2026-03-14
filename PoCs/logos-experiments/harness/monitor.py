"""
Training Monitor — Wraps a training subprocess and watches for failure signals.

Usage:
    python -m harness.monitor -- python workspace/train.py --lr 1e-4
    
    Or programmatically:
        monitor = TrainingMonitor(timeout_hours=4, nan_patience=10)
        result = monitor.run(["python", "workspace/train.py"])
"""

import subprocess
import sys
import re
import time
import signal
import json
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class RunResult:
    """Result of a monitored training run."""
    exit_code: int
    duration_seconds: float
    killed_reason: Optional[str]  # None if completed normally
    nan_detected: bool
    metrics_captured: dict = field(default_factory=dict)
    stdout_tail: str = ""  # Last N lines
    stderr_tail: str = ""

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.nan_detected and self.killed_reason is None

    def to_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration_seconds, 1),
            "duration_human": str(timedelta(seconds=int(self.duration_seconds))),
            "killed_reason": self.killed_reason,
            "nan_detected": self.nan_detected,
            "metrics": self.metrics_captured,
            "succeeded": self.succeeded,
        }


class TrainingMonitor:
    """Monitors a training subprocess for failures and captures metrics."""

    # Patterns that indicate NaN/Inf in output
    NAN_PATTERNS = [
        re.compile(r'\bnan\b', re.IGNORECASE),
        re.compile(r'\binf\b', re.IGNORECASE),
        re.compile(r'loss[:\s]*nan', re.IGNORECASE),
        re.compile(r'loss[:\s]*inf', re.IGNORECASE),
    ]

    # Pattern to capture metrics logged as key=value or key: value
    METRIC_PATTERN = re.compile(
        r'(?:^|\s)(\w+(?:_\w+)*)\s*[=:]\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\b'
    )

    # Structured metric line: [METRIC] key=value
    STRUCTURED_METRIC = re.compile(
        r'\[METRIC\]\s+(\w+)\s*=\s*(.+)'
    )

    def __init__(
        self,
        timeout_hours: float = 4.0,
        nan_patience: int = 10,
        log_dir: Optional[Path] = None,
        tail_lines: int = 100,
    ):
        self.timeout_seconds = timeout_hours * 3600
        self.nan_patience = nan_patience
        self.log_dir = Path(log_dir) if log_dir else None
        self.tail_lines = tail_lines

    def run(self, command: list[str], cwd: Optional[str] = None) -> RunResult:
        """Run a command with monitoring. Returns RunResult."""
        
        start_time = time.time()
        nan_count = 0
        killed_reason = None
        metrics = {}
        stdout_lines = []
        stderr_lines = []
        
        # Set up log files if log_dir specified
        stdout_log = None
        stderr_log = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stdout_log = open(self.log_dir / f"stdout_{timestamp}.log", "w")
            stderr_log = open(self.log_dir / f"stderr_{timestamp}.log", "w")

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                cwd=cwd,
            )

            import selectors
            sel = selectors.DefaultSelector()
            sel.register(proc.stdout, selectors.EVENT_READ)
            sel.register(proc.stderr, selectors.EVENT_READ)

            while proc.poll() is None:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.timeout_seconds:
                    killed_reason = f"timeout ({self.timeout_seconds/3600:.1f}h)"
                    proc.send_signal(signal.SIGTERM)
                    time.sleep(5)
                    if proc.poll() is None:
                        proc.kill()
                    break

                # Read available output
                for key, _ in sel.select(timeout=1.0):
                    line = key.fileobj.readline()
                    if not line:
                        continue

                    is_stderr = key.fileobj == proc.stderr

                    if is_stderr:
                        stderr_lines.append(line.rstrip())
                        if stderr_log:
                            stderr_log.write(line)
                        # Still print stderr for visibility
                        print(line, end="", file=sys.stderr)
                    else:
                        stdout_lines.append(line.rstrip())
                        if stdout_log:
                            stdout_log.write(line)
                        # Print stdout for visibility
                        print(line, end="")

                        # Check for NaN
                        if self._check_nan(line):
                            nan_count += 1
                            print(f"\n⚠️  NaN/Inf detected ({nan_count}/{self.nan_patience})",
                                  file=sys.stderr)
                            if nan_count >= self.nan_patience:
                                killed_reason = f"nan_detected ({nan_count} consecutive)"
                                proc.send_signal(signal.SIGTERM)
                                time.sleep(5)
                                if proc.poll() is None:
                                    proc.kill()
                                break
                        else:
                            nan_count = 0  # Reset on clean line

                        # Capture metrics
                        self._capture_metrics(line, metrics)

            sel.close()

            # Drain any remaining buffered output after process exits
            for stream, is_stderr in [(proc.stdout, False), (proc.stderr, True)]:
                for line in stream:
                    if not line:
                        continue
                    if is_stderr:
                        stderr_lines.append(line.rstrip())
                        if stderr_log:
                            stderr_log.write(line)
                        print(line, end="", file=sys.stderr)
                    else:
                        stdout_lines.append(line.rstrip())
                        if stdout_log:
                            stdout_log.write(line)
                        print(line, end="")
                        if self._check_nan(line):
                            nan_count += 1
                            if nan_count >= self.nan_patience and killed_reason is None:
                                killed_reason = f"nan_detected ({nan_count} consecutive)"
                        else:
                            nan_count = 0
                        self._capture_metrics(line, metrics)

            # Get exit code
            exit_code = proc.returncode if proc.returncode is not None else -1
            duration = time.time() - start_time

            return RunResult(
                exit_code=exit_code,
                duration_seconds=duration,
                killed_reason=killed_reason,
                nan_detected=nan_count > 0,
                metrics_captured=metrics,
                stdout_tail="\n".join(stdout_lines[-self.tail_lines:]),
                stderr_tail="\n".join(stderr_lines[-self.tail_lines:]),
            )

        finally:
            if stdout_log:
                stdout_log.close()
            if stderr_log:
                stderr_log.close()

    def _check_nan(self, line: str) -> bool:
        """Check if a line indicates NaN/Inf in training."""
        # Only check lines that look like they contain metrics
        if any(kw in line.lower() for kw in ['loss', 'metric', 'step', 'epoch', 'batch']):
            return any(p.search(line) for p in self.NAN_PATTERNS)
        return False

    def _capture_metrics(self, line: str, metrics: dict):
        """Capture metrics from structured or semi-structured output."""
        # Prefer structured format: [METRIC] key=value
        m = self.STRUCTURED_METRIC.search(line)
        if m:
            key, value = m.group(1), m.group(2).strip()
            try:
                value = float(value)
            except ValueError:
                pass
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)
            return

        # Fall back to heuristic capture for common metric names
        for m in self.METRIC_PATTERN.finditer(line):
            key, value = m.group(1), m.group(2)
            if key.lower() in {'loss', 'lr', 'learning_rate', 'cosine_sim',
                               'accuracy', 'grad_norm', 'epoch', 'step'}:
                try:
                    value = float(value)
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
                except ValueError:
                    pass


def main():
    """CLI entry point: python -m harness.monitor -- <command>"""
    import argparse
    parser = argparse.ArgumentParser(description="Training Monitor")
    parser.add_argument("--timeout", type=float, default=4.0, help="Timeout in hours")
    parser.add_argument("--nan-patience", type=int, default=10, help="NaN lines before kill")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for log files")
    parser.add_argument("command", nargs="+", help="Command to run")
    args = parser.parse_args()

    monitor = TrainingMonitor(
        timeout_hours=args.timeout,
        nan_patience=args.nan_patience,
        log_dir=Path(args.log_dir) if args.log_dir else None,
    )

    result = monitor.run(args.command)
    
    print("\n" + "=" * 60)
    print("MONITOR RESULT:")
    print(json.dumps(result.to_dict(), indent=2))
    print("=" * 60)

    sys.exit(0 if result.succeeded else 1)


if __name__ == "__main__":
    main()
