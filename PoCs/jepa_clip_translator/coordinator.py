"""Experiment coordinator: config generation and search loop management."""

from __future__ import annotations

import json
import os
import random
import re
import uuid
from dataclasses import dataclass, field

from config import (
    ArchitectureConfig,
    DataConfig,
    ExperimentConfig,
    LossConfig,
    TrainingConfig,
)

SYSTEM_PROMPT = """You are an ML experiment optimizer for a JEPA-to-CLIP embedding translator (1024-dim → 768-dim shared embedding space).

The task: learn a translator that maps V-JEPA temporal video embeddings into a shared space alongside CLIP image and text embeddings. This is NOT about making JEPA look like CLIP — it's about finding a shared geometry that works for all modalities.

You will receive results sorted by val_cosine_sim. Your job is to propose configs that improve on the best result. When the search is plateauing (top results clustered near the same score), be bold — propose genuinely different ideas, not just parameter tweaks.

IMPORTANT — known failure mode:
- Contrastive loss causes training collapse in this setup. Do NOT use contrastive. Stick to mse and cosine only.

Things worth trying when stuck:
- Deeper or wider MLPs, or residual nets with bottleneck layers
- Aggressive LR warmup + cosine decay with long cooldown
- Multi-target losses: cosine toward both clip_image and clip_text simultaneously
- Warmup on MSE (geometry anchor), then switch to cosine (angular alignment)
- Very small LR + long training (200-500 epochs) for linear/shallow models
- Different MSE:cosine weight ratios — pure MSE, pure cosine, or mixed
- Layer norm, dropout, or both — or neither
- Different activations (silu often outperforms gelu on projection tasks)
- Per-target cosine: try clip_text_mean or clip_text_first as primary target instead of clip_image

Schema (all fields optional — only include what you're changing from defaults):
- architecture: type (linear|mlp|residual), hidden_dim (64-2048), num_blocks (1-12), num_layers (1-6), dropout (0.0-0.5), activation (gelu|relu|silu), use_layer_norm (bool)
- loss.terms: [{function (mse|cosine|contrastive), target (clip_image|clip_text_mean|clip_text_first), weight, temperature (0.01-1.0), label_smoothing (0.0-0.2)}]
- loss.warmup_terms: loss mix for first warmup_epochs epochs; loss.warmup_epochs: int
- training: optimizer (adamw|adam|sgd), lr (1e-6 to 5e-2), lr_min, lr_schedule (cosine|step|none), warmup_epochs (0-30), cooldown_epochs (0-50), cooldown_lr, weight_decay (0.0-0.3), batch_size (64|128|256|512|1024), max_epochs (50-500), early_stop_patience (5-30), grad_clip (0.1-10.0)
- data: noise_std (0.0-0.1), embedding_dropout (0.0-0.3), num_tokens (int or null) — for token-level JEPA: number of temporal tokens to use (null = all); for legacy mean-pooled JEPA: number of CLIP frames to average (null = all)

Respond with:

## Analysis
What's working? What's plateauing? What haven't we tried?

## Configs
```json
[...array of ExperimentConfig dicts...]
```"""


# ---------------------------------------------------------------------------
# LLM provider config
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Configures which LLM backend the coordinator uses.

    Defaults are read from environment variables:
      COORDINATOR_LLM_PROVIDER  (default: "openai")
      COORDINATOR_LLM_MODEL     (default: "" → auto per provider)
    """
    provider: str = field(default_factory=lambda: os.environ.get("COORDINATOR_LLM_PROVIDER", "openai"))
    model: str = field(default_factory=lambda: os.environ.get("COORDINATOR_LLM_MODEL", ""))

    def __post_init__(self) -> None:
        if not self.model:
            self.model = "gpt-4o" if self.provider == "openai" else "claude-opus-4-6"

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


def _call_llm(results: list[dict], num_configs: int, llm_config: LLMConfig) -> list[ExperimentConfig]:
    """Call the configured LLM to generate next experiment configs. Returns [] on failure."""
    compact = []
    for r in results:
        c = {k: v for k, v in r.items() if k not in ("best_state", "history")}
        h = r.get("history", [])
        if h:
            c["history_summary"] = {
                "first": h[0],
                "best": h[r.get("best_epoch", 1) - 1] if r.get("best_epoch") else h[0],
                "last": h[-1],
            }
        compact.append(c)

    sorted_results = sorted(compact, key=lambda r: -r.get("val_cosine_sim", 0))
    best_cos = sorted_results[0].get("val_cosine_sim", 0) if sorted_results else 0
    top5 = sorted_results[:5]
    plateau = len(sorted_results) >= 5 and (best_cos - sorted_results[4].get("val_cosine_sim", 0)) < 0.005

    prompt = (
        f"Current best val_cosine_sim: {best_cos:.4f} ({len(compact)} experiments run so far)\n"
        + ("Status: PLATEAUING — top 5 results are within 0.005 of each other. Try something genuinely different.\n" if plateau else "")
        + f"\nTop 5 results:\n{json.dumps(top5, indent=2)}"
        + f"\n\nAll results:\n{json.dumps(sorted_results, indent=2)}"
        + f"\n\nPropose {num_configs} new ExperimentConfig dicts."
    )

    print(f"  Calling {llm_config} ...")

    if llm_config.provider == "openai":
        text = _call_openai(llm_config.model, prompt)
    elif llm_config.provider == "anthropic":
        text = _call_anthropic(llm_config.model, prompt)
    else:
        print(f"  Unknown provider: {llm_config.provider!r} (supported: openai, anthropic)")
        return []

    if text is None:
        return []

    return _parse_configs(text)


def _call_openai(model: str, prompt: str) -> str | None:
    try:
        from openai import OpenAI, AuthenticationError, APIError
    except ImportError:
        print("  openai package not installed — run: pip install openai")
        return None

    try:
        client = OpenAI()  # reads OPENAI_API_KEY from environment
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except AuthenticationError:
        print("  OpenAI auth error — check OPENAI_API_KEY")
        return None
    except APIError as e:
        print(f"  OpenAI API error: {e}")
        return None
    except Exception as e:
        print(f"  OpenAI call failed: {e}")
        return None


def _call_anthropic(model: str, prompt: str) -> str | None:
    try:
        import anthropic
    except ImportError:
        print("  anthropic package not installed — run: pip install anthropic")
        return None

    try:
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
        with client.messages.stream(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            return stream.get_final_message().content[0].text
    except anthropic.AuthenticationError:
        print("  Anthropic auth error — check ANTHROPIC_API_KEY")
        return None
    except anthropic.APIError as e:
        print(f"  Anthropic API error: {e}")
        return None
    except Exception as e:
        print(f"  Anthropic call failed: {e}")
        return None


def _parse_configs(text: str) -> list[ExperimentConfig]:
    """Extract and parse the JSON config array from the LLM response."""
    # Print analysis section (everything before the first code block)
    if "```" in text:
        analysis = text.split("```")[0].strip()
        if analysis:
            print(f"\n{'─' * 70}")
            print("LLM ANALYSIS:")
            print(f"{'─' * 70}")
            print(analysis)
            print(f"{'─' * 70}\n")

    # Extract JSON block
    json_str = None
    if "```json" in text:
        json_str = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        json_str = text.split("```", 1)[1].split("```", 1)[0].strip()
    else:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            json_str = m.group()

    if not json_str:
        print("  Could not extract JSON from LLM response.")
        return []

    try:
        configs_json = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return []

    configs = []
    for d in configs_json:
        if "experiment_id" not in d:
            d["experiment_id"] = f"exp_llm_{uuid.uuid4().hex[:6]}"
        configs.append(ExperimentConfig.from_dict(d))
    return configs


# ---------------------------------------------------------------------------
# Round 1 baseline configs
# ---------------------------------------------------------------------------

def generate_round1_configs() -> list[ExperimentConfig]:
    """Five initial configs covering architectures, loss strategies, and batch sizes."""
    return [
        ExperimentConfig(
            experiment_id="exp_001_linear_mse",
            architecture=ArchitectureConfig(type="linear"),
            training=TrainingConfig(batch_size=128, max_epochs=200, early_stop_patience=10),
            loss=LossConfig(terms=[{"function": "mse", "target": "clip_image", "weight": 1.0,
                                    "temperature": 0.07, "label_smoothing": 0.0}]),
        ),
        ExperimentConfig(
            experiment_id="exp_002_mlp_cosine",
            architecture=ArchitectureConfig(type="mlp", hidden_dim=512, num_layers=2),
            training=TrainingConfig(batch_size=256, max_epochs=200, early_stop_patience=10),
            loss=LossConfig(terms=[
                {"function": "cosine", "target": "clip_image", "weight": 1.0, "temperature": 0.07, "label_smoothing": 0.0},
            ]),
        ),
        ExperimentConfig(
            experiment_id="exp_003_mlp_multitarget",
            architecture=ArchitectureConfig(type="mlp", hidden_dim=512, num_layers=3),
            training=TrainingConfig(batch_size=256, max_epochs=200, early_stop_patience=10, warmup_epochs=10),
            loss=LossConfig(terms=[
                {"function": "cosine", "target": "clip_image", "weight": 0.6, "temperature": 0.07, "label_smoothing": 0.0},
                {"function": "cosine", "target": "clip_text_mean", "weight": 0.4, "temperature": 0.07, "label_smoothing": 0.0},
            ]),
        ),
        ExperimentConfig(
            experiment_id="exp_004_residual_mse_warmup",
            architecture=ArchitectureConfig(type="residual", hidden_dim=512, num_blocks=4),
            training=TrainingConfig(batch_size=256, max_epochs=200, early_stop_patience=10, warmup_epochs=20),
            loss=LossConfig(
                terms=[{"function": "cosine", "target": "clip_image", "weight": 1.0, "temperature": 0.07, "label_smoothing": 0.0}],
                warmup_terms=[{"function": "mse", "target": "clip_image", "weight": 1.0, "temperature": 0.07, "label_smoothing": 0.0}],
                warmup_epochs=20,
            ),
        ),
        ExperimentConfig(
            experiment_id="exp_005_residual_silu",
            architecture=ArchitectureConfig(type="residual", hidden_dim=768, num_blocks=6, dropout=0.1, activation="silu"),
            training=TrainingConfig(batch_size=256, max_epochs=300, early_stop_patience=15,
                                    lr=3e-4, warmup_epochs=15, cooldown_epochs=30, lr_schedule="cosine"),
            loss=LossConfig(terms=[
                {"function": "cosine", "target": "clip_image", "weight": 0.7, "temperature": 0.07, "label_smoothing": 0.0},
                {"function": "cosine", "target": "clip_text_mean", "weight": 0.3, "temperature": 0.07, "label_smoothing": 0.0},
            ]),
            data=DataConfig(noise_std=0.01, embedding_dropout=0.05),
        ),
    ]


# ---------------------------------------------------------------------------
# Random mutation fallback
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def generate_next_configs_random(
    all_results: list[dict], num_configs: int = 2, seed: int | None = None
) -> list[ExperimentConfig]:
    """Mutate the best config to produce variants."""
    rng = random.Random(seed)
    best = min(all_results, key=lambda r: r["val_loss"])
    base = json.loads(json.dumps(best["config"]))

    configs = []
    for _ in range(num_configs):
        d = json.loads(json.dumps(base))

        if rng.random() < 0.3:
            d["architecture"]["type"] = rng.choice(["linear", "mlp", "residual"])
        if rng.random() < 0.5:
            d["architecture"]["hidden_dim"] = max(64, int(
                d["architecture"]["hidden_dim"] * rng.choice([0.5, 0.75, 1.0, 1.5, 2.0])
            ))
        if rng.random() < 0.4:
            d["architecture"]["num_blocks"] = rng.randint(1, 8)
        if rng.random() < 0.3:
            d["architecture"]["num_layers"] = rng.randint(1, 4)
        if rng.random() < 0.4:
            d["architecture"]["dropout"] = round(
                _clamp(d["architecture"]["dropout"] + rng.gauss(0, 0.05), 0.0, 0.5), 3
            )
        if rng.random() < 0.2:
            d["architecture"]["activation"] = rng.choice(["gelu", "relu", "silu"])

        terms = d.get("loss", {}).get("terms", [])
        for term in terms:
            if rng.random() < 0.3:
                term["function"] = rng.choice(["mse", "cosine", "contrastive"])
            if rng.random() < 0.3:
                term["weight"] = round(_clamp(term.get("weight", 1.0) * rng.uniform(0.5, 2.0), 0.1, 5.0), 2)
            if term.get("function") == "contrastive" and rng.random() < 0.3:
                term["temperature"] = round(_clamp(term.get("temperature", 0.07) * rng.uniform(0.5, 2.0), 0.01, 0.5), 4)
        d["loss"]["terms"] = terms

        if rng.random() < 0.5:
            d["training"]["lr"] = round(_clamp(d["training"]["lr"] * rng.uniform(0.3, 3.0), 1e-6, 1e-2), 7)
        if rng.random() < 0.3:
            d["training"]["batch_size"] = rng.choice([64, 128, 256, 512])
        if rng.random() < 0.2:
            d["training"]["optimizer"] = rng.choice(["adamw", "adam", "sgd"])
        if rng.random() < 0.2:
            d["training"]["lr_schedule"] = rng.choice(["cosine", "step", "none"])

        d["experiment_id"] = f"exp_{uuid.uuid4().hex[:8]}"
        configs.append(ExperimentConfig.from_dict(d))

    print(f"  Random mutator proposed {len(configs)} configs from: {best['experiment_id']}")
    return configs


# ---------------------------------------------------------------------------
# Main entry point for config generation
# ---------------------------------------------------------------------------

def generate_next_configs(
    all_results: list[dict],
    num_configs: int = 2,
    llm_config: LLMConfig | None = None,
) -> list[ExperimentConfig]:
    """Try LLM first, fall back to random mutation."""
    if llm_config is None:
        llm_config = LLMConfig()
    configs = _call_llm(all_results, num_configs, llm_config)
    if configs:
        print(f"  LLM ({llm_config}) proposed {len(configs)} configs.")
        return configs
    print("  LLM unavailable, falling back to random mutation.")
    return generate_next_configs_random(all_results, num_configs)
