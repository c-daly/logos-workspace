"""Experiment coordinator — manages the search loop over architectures.

Provides an ExperimentLog for tracking results across rounds, functions
for generating initial and follow-up experiment configurations, and
convergence detection.
"""

from __future__ import annotations

import json
import random
import uuid

from config import (
    ArchitectureConfig,
    DataConfig,
    ExperimentConfig,
    LossConfig,
    TrainingConfig,
)


class ExperimentLog:
    """Append-only log of experiment results with convergence tracking."""

    def __init__(self) -> None:
        self._results: list[dict] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_result(self, result: dict) -> None:
        """Append an experiment result dict."""
        self._results.append(result)

    def num_experiments(self) -> int:
        """Return the number of logged experiments."""
        return len(self._results)

    def best_result(self) -> dict | None:
        """Return the result with the lowest ``val_loss``, or *None*."""
        if not self._results:
            return None
        return min(self._results, key=lambda r: r["val_loss"])

    def is_converged(self, patience_rounds: int = 3) -> bool:
        """True if the last *patience_rounds* results haven't beaten the best.

        Requires at least ``patience_rounds + 1`` results *after* the best
        result before declaring convergence, so that we don't trigger too
        early in the search.
        """
        if not self._results:
            return False
        best = self.best_result()
        assert best is not None
        best_loss = best["val_loss"]

        # Find index of best result (first occurrence of min val_loss)
        best_idx = next(
            i for i, r in enumerate(self._results) if r["val_loss"] == best_loss
        )
        results_after_best = len(self._results) - best_idx - 1
        if results_after_best < patience_rounds + 1:
            return False

        tail = self._results[-patience_rounds:]
        return all(r["val_loss"] > best_loss for r in tail)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the log to JSON."""
        with open(path, "w") as f:
            json.dump({"experiments": self._results}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentLog":
        """Deserialize an ExperimentLog from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        log = cls()
        for entry in data["experiments"]:
            log.add_result(entry)
        return log

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable table of all experiments with the best highlighted."""
        if not self._results:
            return "(no experiments logged)"

        best = self.best_result()
        best_id = best["experiment_id"] if best else None

        lines: list[str] = []
        header = f"{'ID':<20} {'val_loss':>10} {'val_cos':>10} {'best':>5}"
        lines.append(header)
        lines.append("-" * len(header))

        for r in self._results:
            eid = r.get("experiment_id", "?")
            vloss = r.get("val_loss", float("nan"))
            vcos = r.get("val_cosine_sim", float("nan"))
            marker = "  *" if eid == best_id else ""
            lines.append(f"{eid:<20} {vloss:>10.4f} {vcos:>10.4f} {marker:>5}")

        return "\n".join(lines)


# ======================================================================
# Config generation
# ======================================================================


def generate_round1_configs() -> list[ExperimentConfig]:
    """Return 3 initial configs: linear, MLP, residual."""
    linear = ExperimentConfig(
        experiment_id="exp_001_linear",
        architecture=ArchitectureConfig(type="linear"),
    )
    mlp = ExperimentConfig(
        experiment_id="exp_002_mlp",
        architecture=ArchitectureConfig(type="mlp", hidden_dim=512),
    )
    residual = ExperimentConfig(
        experiment_id="exp_003_residual",
        architecture=ArchitectureConfig(
            type="residual", hidden_dim=512, num_blocks=4
        ),
    )
    return [linear, mlp, residual]


def _short_uid() -> str:
    """8-char hex suffix for experiment IDs."""
    return uuid.uuid4().hex[:8]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def generate_next_configs(
    log: ExperimentLog,
    num_configs: int = 3,
    seed: int | None = None,
) -> list[ExperimentConfig]:
    """Generate *num_configs* mutated variants of the best config so far.

    Each config gets a unique experiment_id and random perturbations to
    architecture, training, and loss hyper-parameters.
    """
    rng = random.Random(seed)
    best = log.best_result()
    if best is None:
        raise ValueError("Cannot generate next configs from an empty log")

    base_dict = best["config"]
    configs: list[ExperimentConfig] = []

    arch_types: list[str] = ["linear", "mlp", "residual"]
    loss_types: list[str] = ["mse", "cosine", "contrastive", "combined"]

    for _ in range(num_configs):
        # Deep-copy the base config dict
        d = json.loads(json.dumps(base_dict))

        # --- Architecture mutations ---
        if rng.random() < 0.3:
            d["architecture"]["type"] = rng.choice(arch_types)
        if rng.random() < 0.5:
            factor = rng.choice([0.5, 0.75, 1.0, 1.5, 2.0])
            d["architecture"]["hidden_dim"] = max(
                64, int(d["architecture"]["hidden_dim"] * factor)
            )
        if rng.random() < 0.4:
            d["architecture"]["num_blocks"] = rng.randint(1, 8)
        if rng.random() < 0.4:
            d["architecture"]["dropout"] = round(
                _clamp(d["architecture"]["dropout"] + rng.gauss(0, 0.05), 0.0, 0.5),
                3,
            )

        # --- Loss mutations ---
        if rng.random() < 0.3:
            d["loss"]["type"] = rng.choice(loss_types)
        if rng.random() < 0.4:
            d["loss"]["temperature"] = round(
                _clamp(d["loss"]["temperature"] * rng.uniform(0.5, 2.0), 0.01, 0.5),
                4,
            )
        if rng.random() < 0.4:
            d["loss"]["contrastive_weight"] = round(
                _clamp(rng.uniform(0.3, 0.9), 0.0, 1.0), 2
            )
            d["loss"]["cosine_weight"] = round(
                1.0 - d["loss"]["contrastive_weight"], 2
            )

        # --- Training mutations ---
        if rng.random() < 0.5:
            d["training"]["lr"] = round(
                _clamp(
                    d["training"]["lr"] * rng.uniform(0.3, 3.0), 1e-6, 1e-2
                ),
                7,
            )
        if rng.random() < 0.3:
            d["training"]["batch_size"] = rng.choice([32, 64, 128, 256])

        # Assign a unique experiment ID
        d["experiment_id"] = f"exp_{_short_uid()}"

        configs.append(ExperimentConfig.from_dict(d))

    return configs
