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

SYSTEM_PROMPT = """You are an ML experiment optimizer for a JEPA-to-CLIP embedding translator (V-JEPA 1024-dim token embeddings -> 768-dim CLIP space).

PRIMARY OBJECTIVE: maximize TEXT retrieval (txt_R@1, txt_R@5). Image retrieval is already solved (imgR@5 ~0.93). The critical gap is text: current best txt_R@1=0.11, txt_R@5=0.31.

KEY FINDINGS so far:
- InfoNCE (contrastive) loss is ESSENTIAL for retrieval — it is the single biggest lever. Use it heavily.
- Pure MSE/cosine losses produce high cosine similarity but near-zero retrieval. Do NOT rely on them alone.
- Multi-target losses that include clip_text_mean alongside clip_image significantly help text retrieval.
- Transformer architectures with InfoNCE achieve the best results.

TO IMPROVE TEXT RETRIEVAL specifically:
- Add InfoNCE loss terms targeting `clip_text_mean` or `clip_text_first` (not just clip_image).
- Use bidirectional alignment: loss against both image AND text CLIP embeddings.
- Try heavier weights on text-target loss terms (e.g. 0.5+ on cosine->clip_text_mean).
- Warmup with MSE then switch to InfoNCE works well.
- Lower InfoNCE temperature (0.03-0.07) tends to sharpen retrieval.

Schema: architecture: {type, hidden_dim, num_blocks, num_layers, num_heads, dropout, activation, use_layer_norm}
loss.terms: [{function: mse|cosine|infonce, target: clip_image|clip_text_mean|clip_text_first, weight, temperature, label_smoothing}]
training: {optimizer, lr, lr_min, lr_schedule, warmup_epochs, cooldown_epochs, weight_decay, batch_size, max_epochs, early_stop_patience, grad_clip}
data: {noise_std, embedding_dropout, num_tokens}"""


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

    sorted_results = sorted(compact, key=lambda r: -r.get("txt_R@1", r.get("R@1", 0)))
    best_txt_r1 = sorted_results[0].get("txt_R@1", 0.0) if sorted_results else 0.0
    best_txt_r5 = sorted_results[0].get("txt_R@5", 0.0) if sorted_results else 0.0
    best_img_r5 = max((r.get("R@5", 0.0) for r in sorted_results), default=0.0)
    top5 = sorted_results[:5]
    plateau = len(sorted_results) >= 5 and (best_txt_r1 - sorted_results[4].get("txt_R@1", best_txt_r1)) < 0.01

    prompt = (
        f"Results so far ({len(compact)} experiments). Ranked by txt_R@1 (primary target).\n"
        f"Best: txt_R@1={best_txt_r1:.3f}  txt_R@5={best_txt_r5:.3f}  imgR@5={best_img_r5:.3f}\n"
        f"Goal: push txt_R@1 higher. Text retrieval is far below image retrieval — focus on text-target losses and InfoNCE.\n"
        + ("Status: PLATEAUING on txt_R@1 — try a genuinely different approach.\n" if plateau else "")
        + f"\nTop 5 results:\n{json.dumps(top5, indent=2)}"
        + f"\n\nAll results:\n{json.dumps(sorted_results, indent=2)}"
        + f"\n\nPropose {num_configs} new ExperimentConfig dicts targeting improved txt_R@1."
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
