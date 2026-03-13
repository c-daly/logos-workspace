"""Experiment coordinator: config generation and search loop management."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

from config import (
    ArchitectureConfig,
    DataConfig,
    ExperimentConfig,
    LossConfig,
    TrainingConfig,
)

SYSTEM_PROMPT = """You are an ML experiment optimizer for a JEPA-to-CLIP embedding translator (1024-dim → 768-dim shared embedding space).

The task: learn a translator that maps V-JEPA temporal video embeddings into a shared space alongside CLIP image and text embeddings. This is NOT about making JEPA look like CLIP — it’s about finding a shared geometry that works for all modalities.

You will receive results sorted by txt_R@1. Each result includes imgR@1, imgR@5 (image retrieval) and txtR@1, txtR@5 (text retrieval). The primary objective is to maximize txt_R@1 and txt_R@5. Image retrieval is already solved (imgR@5 ~0.93). **To improve text retrieval, increase the weight on infonce loss** -- infonce is the only loss term that directly trains retrieval behavior. mse/cosine losses improve geometry but not retrieval rank. When the search is plateauing, be bold -- propose genuinely different ideas, not just parameter tweaks.

IMPORTANT -- loss function guidance:
- Vanilla "contrastive" causes training collapse. Do NOT use it.
- Use "infonce" (false-negative masked InfoNCE) for contrastive objectives -- it is safe and effective.
- Mixed targets work well: 0.7xmse->clip_text_mean + 0.3xcosine->clip_image improved text retrieval significantly.
- Good combination: 0.5xmse->clip_image + 0.3xcosine->clip_text_mean + 0.2xinfonce->clip_image
- For text retrieval specifically: infonce->clip_text_mean is the most direct lever. Try weighting it heavily.

Things worth trying when stuck:
- Deeper or wider MLPs, or residual nets with bottleneck layers
- Aggressive LR warmup + cosine decay with long cooldown
- Multi-target losses: infonce toward both clip_image and clip_text_mean simultaneously
- Warmup on MSE (geometry anchor), then switch to infonce (retrieval alignment)
- Very small LR + long training (200-500 epochs) for linear/shallow models
- Different MSE:cosine:infonce weight ratios
- Layer norm, dropout, or both — or neither
- Different activations (silu often outperforms gelu on projection tasks)
- Per-target infonce: try clip_text_mean or clip_text_first as primary target instead of clip_image

Schema (all fields optional — only include what you’re changing from defaults):
- architecture: type (linear|mlp|residual|transformer), hidden_dim (64-2048), num_blocks (1-12), num_layers (1-6), num_heads (1-16, must divide hidden_dim evenly), dropout (0.0-0.5), activation (gelu|relu|silu), use_layer_norm (bool)
  - transformer: self-attention over T temporal tokens before projecting — the ONLY architecture that models cross-token temporal context. num_blocks = transformer encoder layers, num_heads = attention heads.
  - stages (optional list): build a pipeline of heterogeneous stages instead of a single type. Each stage is a dict with its own type/hidden_dim/etc. Dimensions chain automatically — each stage’s hidden_dim is its output width, except the last stage which always outputs to clip_dim (768). Use this to combine architectures, e.g. residual feature extraction followed by an MLP bottleneck. Example: [{"type": "residual", "hidden_dim": 1024, "num_blocks": 6}, {"type": "mlp", "hidden_dim": 512, "num_layers": 2}]. When stages is set, top-level type/hidden_dim/etc. are ignored.
- loss.terms: [{function (mse|cosine|infonce), target (clip_image|clip_text_mean|clip_text_first), weight, temperature (0.01-1.0), label_smoothing (0.0-0.2)}]
- loss.warmup_terms: loss mix for first warmup_epochs epochs; loss.warmup_epochs: int
- training: optimizer (adamw|adam|sgd), lr (1e-6 to 5e-2), lr_min, lr_schedule (cosine|step|none), warmup_epochs (0-30), cooldown_epochs (0-50), cooldown_lr, weight_decay (0.0-0.3), batch_size (64|128|256), max_epochs (50-500), early_stop_patience (5-30), grad_clip (0.1-10.0)
- data: noise_std (0.0-0.1), embedding_dropout (0.0-0.3), num_tokens (int or null) — for token-level JEPA: number of temporal tokens to use (null = all); for legacy mean-pooled JEPA: number of CLIP frames to average (null = all)

CRITICAL: batch_size must be ≤ 256. Token-level training multiplies effective batch by K=32 — larger values cause GPU OOM on a 24GB card.

Example pipeline config (stages override top-level type):
```json
{
  "experiment_id": "exp_residual_then_mlp",
  "architecture": {
    "stages": [
      {"type": "residual", "hidden_dim": 1024, "num_blocks": 6, "activation": "silu", "dropout": 0.1, "use_layer_norm": true},
      {"type": "mlp", "hidden_dim": 512, "num_layers": 2, "activation": "gelu", "dropout": 0.05, "use_layer_norm": true}
    ]
  },
  "training": {"lr": 3e-4, "warmup_epochs": 15, "cooldown_epochs": 30, "max_epochs": 300}
}
```

Respond with:

## Analysis
Optionally start with `TARGET: <metric>` (one of: txt_R@1, txt_R@5, R@5) to recommend what to optimize this round — default is txt_R@1.
What’s working? What’s plateauing? What haven’t we tried?

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


def _call_llm(results: list[dict], num_configs: int, llm_config: LLMConfig, llm_log_path: str | None = None) -> tuple[list[ExperimentConfig], str | None]:
    """Call the configured LLM to generate next experiment configs. Returns ([], None) on failure."""
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

    # Sort by txt_R@1 if available, else R@5, else R@1
    def _sort_key(r):
        if r.get("txt_R@1", 0.0) > 0:
            return -r["txt_R@1"]
        return -r.get("R@5", r.get("R@1", 0))
    sorted_results = sorted(compact, key=_sort_key)
    best_txt_r1 = sorted_results[0].get("txt_R@1", 0.0) if sorted_results else 0.0
    best_txt_r5 = sorted_results[0].get("txt_R@5", 0.0) if sorted_results else 0.0
    best_img_r5 = max((r.get("R@5", 0.0) for r in sorted_results), default=0.0)
    top5 = sorted_results[:5]
    plateau = len(sorted_results) >= 5 and (best_txt_r1 - sorted_results[4].get("txt_R@1", best_txt_r1)) < 0.01

    prompt = (
        f"Current best txt_R@1={best_txt_r1:.4f}  txt_R@5={best_txt_r5:.4f}  imgR@5={best_img_r5:.4f} ({len(compact)} experiments run so far)\n"
        + ("NOTE: older results ranked by imgR@5 (txt_R@1 not available).\n" if best_txt_r1 == 0.0 else "")
        + ("Status: PLATEAUING on txt_R@1 — top 5 are within 0.01 of each other. Try something genuinely different.\n" if plateau else "")
        + f"\nTop 5 results:\n{json.dumps(top5, indent=2)}"
        + f"\n\nAll results:\n{json.dumps(sorted_results, indent=2)}"
        + f"\n\nPropose {num_configs} new ExperimentConfig dicts."
    )

    logger.info("  Calling %s ...", llm_config)

    if llm_config.provider == "openai":
        text = _call_openai(llm_config.model, prompt)
    elif llm_config.provider == "anthropic":
        text = _call_anthropic(llm_config.model, prompt)
    else:
        print(f"  Unknown provider: {llm_config.provider!r} (supported: openai, anthropic)")
        return [], None

    if text is None:
        return [], None

    return _parse_configs(text, llm_log_path=llm_log_path)


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


_VALID_TARGET_METRICS = {"txt_R@1", "txt_R@5", "R@5", "R@1", "val_cosine_sim"}


def _parse_configs(text: str, llm_log_path: str | None = None) -> tuple[list[ExperimentConfig], str | None]:
    """Extract and parse the JSON config array from the LLM response.

    Returns (configs, recommended_target_metric).
    recommended_target_metric is None if the coordinator didn't specify one.
    """
    # Write full response to dedicated LLM log file
    if llm_log_path:
        try:
            from datetime import datetime as _dt
            with open(llm_log_path, "a") as _lf:
                _lf.write(f"\n{'=' * 70}\n{_dt.now().strftime('%H:%M:%S')} LLM RESPONSE\n{'=' * 70}\n")
                _lf.write(text)
                _lf.write(f"\n{'=' * 70}\n")
        except Exception:
            pass

    # Log analysis section (everything before the first code block)
    if "```" in text:
        analysis = text.split("```")[0].strip()
    else:
        m = re.search(r"\[", text)
        analysis = text[:m.start()].strip() if m else text.strip()
    if analysis:
        logger.info("\n%s\nLLM ANALYSIS:\n%s\n%s", "-" * 70, analysis, "-" * 70)

    # Extract recommended target metric from analysis (e.g. "TARGET: txt_R@5")
    recommended_target: str | None = None
    m = re.search(r"^TARGET:\s*([\w@]+)", analysis, re.MULTILINE | re.IGNORECASE)
    if m:
        candidate = m.group(1)
        if candidate in _VALID_TARGET_METRICS:
            recommended_target = candidate
            logger.info("  Coordinator recommends target metric: %s", recommended_target)
        else:
            logger.info("  Coordinator suggested unknown metric %r — ignoring.", candidate)

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
        return [], recommended_target

    try:
        configs_json = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return [], recommended_target

    configs = []
    for d in configs_json:
        if "experiment_id" not in d:
            d["experiment_id"] = f"exp_llm_{uuid.uuid4().hex[:6]}"
        configs.append(ExperimentConfig.from_dict(d))
    return configs, recommended_target


# ---------------------------------------------------------------------------
# Round 1 baseline configs
# ---------------------------------------------------------------------------

def generate_round1_configs() -> list[ExperimentConfig]:
    """Five initial configs: transformer first (boosted), then pointwise baselines and a pipeline."""
    return [
        # Transformer first — higher LR, longer warmup, more patience than round 0 attempt.
        ExperimentConfig(
            experiment_id="exp_001_transformer",
            architecture=ArchitectureConfig(type="transformer", hidden_dim=512, num_blocks=4, num_heads=8, dropout=0.05),
            training=TrainingConfig(batch_size=512, max_epochs=300, early_stop_patience=20,
                                    lr=5e-4, warmup_epochs=20, cooldown_epochs=40, lr_schedule="cosine"),
            loss=LossConfig(terms=[
                {"function": "cosine", "target": "clip_image", "weight": 0.7, "temperature": 0.07, "label_smoothing": 0.0},
                {"function": "cosine", "target": "clip_text_mean", "weight": 0.3, "temperature": 0.07, "label_smoothing": 0.0},
            ]),
        ),
        ExperimentConfig(
            experiment_id="exp_002_linear_mse",
            architecture=ArchitectureConfig(type="linear"),
            training=TrainingConfig(batch_size=128, max_epochs=200, early_stop_patience=10),
            loss=LossConfig(terms=[{"function": "mse", "target": "clip_image", "weight": 1.0,
                                    "temperature": 0.07, "label_smoothing": 0.0}]),
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
            experiment_id="exp_004_residual_silu",
            architecture=ArchitectureConfig(type="residual", hidden_dim=768, num_blocks=6, dropout=0.1, activation="silu"),
            training=TrainingConfig(batch_size=256, max_epochs=300, early_stop_patience=15,
                                    lr=3e-4, warmup_epochs=15, cooldown_epochs=30, lr_schedule="cosine"),
            loss=LossConfig(terms=[
                {"function": "cosine", "target": "clip_image", "weight": 0.7, "temperature": 0.07, "label_smoothing": 0.0},
                {"function": "cosine", "target": "clip_text_mean", "weight": 0.3, "temperature": 0.07, "label_smoothing": 0.0},
            ]),
            data=DataConfig(noise_std=0.01, embedding_dropout=0.05),
        ),
        # Pipeline: residual for feature extraction, MLP to compress to output dim.
        ExperimentConfig(
            experiment_id="exp_005_residual_mlp_pipeline",
            architecture=ArchitectureConfig(stages=[
                {"type": "residual", "hidden_dim": 1024, "num_blocks": 6, "dropout": 0.1, "activation": "silu", "use_layer_norm": True},
                {"type": "mlp", "hidden_dim": 512, "num_layers": 2, "dropout": 0.05, "activation": "gelu", "use_layer_norm": True},
            ]),
            training=TrainingConfig(batch_size=256, max_epochs=300, early_stop_patience=15,
                                    lr=3e-4, warmup_epochs=15, cooldown_epochs=30, lr_schedule="cosine"),
            loss=LossConfig(terms=[
                {"function": "cosine", "target": "clip_image", "weight": 0.7, "temperature": 0.07, "label_smoothing": 0.0},
                {"function": "cosine", "target": "clip_text_mean", "weight": 0.3, "temperature": 0.07, "label_smoothing": 0.0},
            ]),
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
                term["function"] = rng.choice(["mse", "cosine", "infonce"])
            if rng.random() < 0.3:
                term["weight"] = round(_clamp(term.get("weight", 1.0) * rng.uniform(0.5, 2.0), 0.1, 5.0), 2)
            if term.get("function") == "infonce" and rng.random() < 0.3:
                term["temperature"] = round(_clamp(term.get("temperature", 0.07) * rng.uniform(0.5, 2.0), 0.01, 0.5), 4)
        d["loss"]["terms"] = terms

        if rng.random() < 0.5:
            d["training"]["lr"] = round(_clamp(d["training"]["lr"] * rng.uniform(0.3, 3.0), 1e-6, 1e-2), 7)
        if rng.random() < 0.3:
            d["training"]["batch_size"] = rng.choice([64, 128, 256])
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
    llm_log_path: str | None = None,
) -> tuple[list[ExperimentConfig], str | None]:
    """Try LLM first, fall back to random mutation.

    Returns (configs, recommended_target_metric).
    recommended_target_metric is None if the coordinator didn't specify one.
    """
    if llm_config is None:
        llm_config = LLMConfig()
    configs, recommended_target = _call_llm(all_results, num_configs, llm_config, llm_log_path=llm_log_path)
    if configs:
        print(f"  LLM ({llm_config}) proposed {len(configs)} configs.")
        return configs, recommended_target
    print("  LLM unavailable, falling back to random mutation.")
    return generate_next_configs_random(all_results, num_configs), None
