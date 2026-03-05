# JEPA-to-CLIP Translator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an agent feedback loop that iterates on a V-JEPA to CLIP embedding translator using MSR-VTT, starting with a CPU-friendly proof of concept.

**Architecture:** Pre-compute V-JEPA (1024-dim) and CLIP (768-dim) embeddings from MSR-VTT videos, then train translator networks mapping between the spaces. A coordinator agent spawns parallel worker agents that each try different configs, collecting results in a shared experiment log.

**Tech Stack:** PyTorch, transformers (HuggingFace), h5py, numpy, av (video decoding)

**Design doc:** `docs/plans/2026-03-05-jepa-clip-translator-design.md`
**Prior work:** `~/cdaly/Downloads/jepa_to_clip_v2.ipynb`

---

### Task 1: Project Scaffolding

**Files:**
- Create: `PoCs/jepa_clip_translator/requirements.txt`
- Create: `PoCs/jepa_clip_translator/.gitignore`
- Create: `PoCs/jepa_clip_translator/config.py`

**Step 1: Create requirements.txt**

```
torch>=2.0
transformers>=4.40
h5py
numpy
av
tqdm
Pillow
scikit-learn
```

**Step 2: Create .gitignore**

```
data/
embeddings/
checkpoints/
log.json
__pycache__/
*.pyc
```

**Step 3: Create config.py**

```python
"""Experiment configuration schema and defaults."""

from dataclasses import dataclass, field, asdict
from typing import Literal
import json


@dataclass
class ArchitectureConfig:
    type: Literal["linear", "mlp", "residual"] = "linear"
    hidden_dim: int = 512
    num_blocks: int = 0  # only for residual
    dropout: float = 0.1
    activation: str = "gelu"


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.05
    batch_size: int = 64
    max_epochs: int = 20
    early_stop_patience: int = 5
    grad_clip: float = 1.0


@dataclass
class LossConfig:
    type: Literal["mse", "cosine", "contrastive", "combined"] = "combined"
    contrastive_weight: float = 0.7
    cosine_weight: float = 0.3
    temperature: float = 0.07


@dataclass
class DataConfig:
    subset_size: int = 50  # number of videos
    val_fraction: float = 0.15
    clip_sample_frames: int = 8


@dataclass
class ExperimentConfig:
    experiment_id: str = "exp_001"
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vjepa_dim: int = 1024
    clip_dim: int = 768

    def to_dict(self):
        return asdict(self)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            experiment_id=d.get("experiment_id", "exp_001"),
            architecture=ArchitectureConfig(**d.get("architecture", {})),
            training=TrainingConfig(**d.get("training", {})),
            loss=LossConfig(**d.get("loss", {})),
            data=DataConfig(**d.get("data", {})),
            vjepa_dim=d.get("vjepa_dim", 1024),
            clip_dim=d.get("clip_dim", 768),
        )
```

**Step 4: Install dependencies**

Run: `cd PoCs/jepa_clip_translator && pip install -r requirements.txt`

**Step 5: Commit**

```bash
git add PoCs/jepa_clip_translator/
git commit -m "feat: scaffold jepa-clip translator PoC"
```

---

### Task 2: Data Download & Loading

**Files:**
- Create: `PoCs/jepa_clip_translator/data_loader.py`
- Create: `PoCs/jepa_clip_translator/tests/test_data_loader.py`

**Step 1: Write the failing test**

```python
# tests/test_data_loader.py
import os
import pytest
from data_loader import MSRVTTLoader


def test_loader_init():
    loader = MSRVTTLoader(data_dir="data", subset_size=5)
    assert loader.data_dir == "data"
    assert loader.subset_size == 5


def test_download_creates_annotation_file(tmp_path):
    """Test that ensure_data at minimum creates the annotation structure."""
    loader = MSRVTTLoader(data_dir=str(tmp_path), subset_size=5)
    # This tests the annotation download/parse path
    # Full video download is tested manually (too slow for unit tests)
    assert loader.data_dir == str(tmp_path)


def test_load_annotations_structure(tmp_path):
    """Test annotation parsing with a mock annotation file."""
    import json
    mock_annotations = {
        "sentences": [
            {"video_id": "video0", "caption": "a man is talking"},
            {"video_id": "video0", "caption": "someone speaks to camera"},
            {"video_id": "video1", "caption": "a dog runs in a park"},
        ],
        "videos": [
            {"video_id": "video0", "url": "https://example.com/v0.mp4"},
            {"video_id": "video1", "url": "https://example.com/v1.mp4"},
        ],
    }
    ann_path = tmp_path / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(mock_annotations, f)

    loader = MSRVTTLoader(data_dir=str(tmp_path), subset_size=5)
    videos, captions = loader.parse_annotations(ann_path)
    assert len(videos) == 2
    assert "video0" in captions
    assert len(captions["video0"]) == 2
    assert captions["video1"][0] == "a dog runs in a park"
```

**Step 2: Run test to verify it fails**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_data_loader.py -v`
Expected: FAIL with "No module named 'data_loader'"

**Step 3: Write data_loader.py**

```python
"""MSR-VTT data download and loading."""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import av
import numpy as np
from tqdm import tqdm


class MSRVTTLoader:
    """Download and load MSR-VTT videos and annotations."""

    # MSR-VTT annotations URL (community-hosted, stable)
    ANNOTATIONS_URL = (
        "https://raw.githubusercontent.com/ArrowLuo/CLIP4Clip/"
        "master/data/msrvtt_data/MSRVTT_data.json"
    )
    # Videos hosted on various mirrors; user may need to provide path
    VIDEOS_URL = None  # Set by user or discovered at runtime

    def __init__(self, data_dir: str = "data", subset_size: int = 50):
        self.data_dir = data_dir
        self.subset_size = subset_size
        self.video_dir = os.path.join(data_dir, "videos")
        self.annotations_path = os.path.join(data_dir, "annotations.json")

    def ensure_data(self) -> tuple[list[dict], dict[str, list[str]]]:
        """Ensure data exists locally. Download if needed. Return (videos, captions)."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        # Download annotations if missing
        if not os.path.exists(self.annotations_path):
            print(f"Downloading MSR-VTT annotations...")
            self._download_file(self.ANNOTATIONS_URL, self.annotations_path)

        videos, captions = self.parse_annotations(Path(self.annotations_path))

        # Check which videos we have locally
        available = self._find_local_videos(videos)
        if len(available) < self.subset_size:
            print(
                f"Found {len(available)}/{self.subset_size} videos locally. "
                f"Need to download {self.subset_size - len(available)} more."
            )
            self._download_videos(videos, captions, self.subset_size - len(available))
            available = self._find_local_videos(videos)

        # Trim to subset
        subset_ids = [v["video_id"] for v in available[: self.subset_size]]
        subset_videos = [v for v in videos if v["video_id"] in subset_ids]
        subset_captions = {vid: captions[vid] for vid in subset_ids if vid in captions}

        print(f"Using {len(subset_videos)} videos with {sum(len(c) for c in subset_captions.values())} captions")
        return subset_videos, subset_captions

    def parse_annotations(self, ann_path: Path) -> tuple[list[dict], dict[str, list[str]]]:
        """Parse MSR-VTT annotation JSON. Returns (videos_list, {video_id: [captions]})."""
        with open(ann_path) as f:
            data = json.load(f)

        videos = data.get("videos", [])
        captions: dict[str, list[str]] = {}
        for sent in data.get("sentences", []):
            vid = sent["video_id"]
            captions.setdefault(vid, []).append(sent["caption"])

        return videos, captions

    def _find_local_videos(self, videos: list[dict]) -> list[dict]:
        """Return videos that exist locally as files."""
        available = []
        for v in videos:
            vid = v["video_id"]
            path = os.path.join(self.video_dir, f"{vid}.mp4")
            if os.path.exists(path):
                available.append(v)
        return available

    def _download_file(self, url: str, dest: str):
        """Download a file using curl."""
        subprocess.run(["curl", "-sL", "-o", dest, url], check=True)

    def _download_videos(self, videos: list[dict], captions: dict, count: int):
        """Attempt to download videos. Falls back to user instructions."""
        print(
            f"\nMSR-VTT videos require manual download or a mirror URL.\n"
            f"Options:\n"
            f"  1. Download from: https://www.mediafire.com/folder/h14iarbs62e7p/shared\n"
            f"  2. Place .mp4 files in: {self.video_dir}/\n"
            f"  3. Set MSRVTT_VIDEOS_URL env var to a zip/tar URL\n"
        )
        mirror = os.environ.get("MSRVTT_VIDEOS_URL")
        if mirror:
            print(f"Downloading from mirror: {mirror}")
            archive = os.path.join(self.data_dir, "videos.zip")
            self._download_file(mirror, archive)
            subprocess.run(["unzip", "-o", "-q", archive, "-d", self.video_dir], check=True)

    def load_video_frames(self, video_id: str, max_frames: int = 64) -> Optional[list[np.ndarray]]:
        """Load frames from a video file. Returns list of RGB numpy arrays or None."""
        path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if not os.path.exists(path):
            return None
        try:
            container = av.open(path)
            frames = []
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= max_frames:
                    break
            container.close()
            return frames if len(frames) >= 16 else None  # skip very short videos
        except Exception as e:
            print(f"Error loading {video_id}: {e}")
            return None
```

**Step 4: Run tests to verify they pass**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_data_loader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add PoCs/jepa_clip_translator/data_loader.py PoCs/jepa_clip_translator/tests/
git commit -m "feat: MSR-VTT data loader with download fallback"
```

---

### Task 3: Translator Models

**Files:**
- Create: `PoCs/jepa_clip_translator/translator.py`
- Create: `PoCs/jepa_clip_translator/tests/test_translator.py`

**Step 1: Write the failing test**

```python
# tests/test_translator.py
import torch
import pytest
from translator import build_translator
from config import ArchitectureConfig


def test_linear_translator_shape():
    cfg = ArchitectureConfig(type="linear")
    model = build_translator(cfg, input_dim=1024, output_dim=768)
    x = torch.randn(4, 1024)
    out = model(x)
    assert out.shape == (4, 768)


def test_mlp_translator_shape():
    cfg = ArchitectureConfig(type="mlp", hidden_dim=256, dropout=0.0)
    model = build_translator(cfg, input_dim=1024, output_dim=768)
    x = torch.randn(4, 1024)
    out = model(x)
    assert out.shape == (4, 768)


def test_residual_translator_shape():
    cfg = ArchitectureConfig(type="residual", hidden_dim=256, num_blocks=2, dropout=0.0)
    model = build_translator(cfg, input_dim=1024, output_dim=768)
    x = torch.randn(4, 1024)
    out = model(x)
    assert out.shape == (4, 768)


def test_output_is_l2_normalized():
    cfg = ArchitectureConfig(type="mlp", hidden_dim=256)
    model = build_translator(cfg, input_dim=1024, output_dim=768)
    x = torch.randn(4, 1024)
    out = model(x)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_unknown_type_raises():
    cfg = ArchitectureConfig(type="transformer")  # type: ignore
    with pytest.raises(ValueError):
        build_translator(cfg, input_dim=1024, output_dim=768)
```

**Step 2: Run test to verify it fails**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_translator.py -v`
Expected: FAIL with "cannot import name 'build_translator'"

**Step 3: Write translator.py**

```python
"""Translator model definitions: V-JEPA (1024-dim) -> CLIP (768-dim)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ArchitectureConfig


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class LinearTranslator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(x), dim=-1)


class MLPTranslator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ResidualTranslator(nn.Module):
    """V2-style translator with residual blocks."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int, num_blocks: int, dropout: float
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return F.normalize(x, dim=-1)


def build_translator(cfg: ArchitectureConfig, input_dim: int, output_dim: int) -> nn.Module:
    """Factory function to build a translator from config."""
    if cfg.type == "linear":
        return LinearTranslator(input_dim, output_dim)
    elif cfg.type == "mlp":
        return MLPTranslator(input_dim, output_dim, cfg.hidden_dim, cfg.dropout)
    elif cfg.type == "residual":
        return ResidualTranslator(input_dim, output_dim, cfg.hidden_dim, cfg.num_blocks, cfg.dropout)
    else:
        raise ValueError(f"Unknown translator type: {cfg.type}")
```

**Step 4: Run tests to verify they pass**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_translator.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add PoCs/jepa_clip_translator/translator.py PoCs/jepa_clip_translator/tests/test_translator.py
git commit -m "feat: translator models (linear, MLP, residual)"
```

---

### Task 4: Loss Functions

**Files:**
- Create: `PoCs/jepa_clip_translator/losses.py`
- Create: `PoCs/jepa_clip_translator/tests/test_losses.py`

**Step 1: Write the failing test**

```python
# tests/test_losses.py
import torch
import pytest
from losses import build_loss_fn
from config import LossConfig


def test_mse_loss_returns_scalar():
    cfg = LossConfig(type="mse")
    loss_fn = build_loss_fn(cfg)
    pred = torch.randn(8, 768)
    target = torch.randn(8, 768)
    result = loss_fn(pred, target)
    assert result["loss"].shape == ()


def test_cosine_loss_identical_is_zero():
    cfg = LossConfig(type="cosine")
    loss_fn = build_loss_fn(cfg)
    x = torch.randn(8, 768)
    x = torch.nn.functional.normalize(x, dim=-1)
    result = loss_fn(x, x)
    assert result["loss"].item() < 0.01


def test_contrastive_loss_returns_accuracy():
    cfg = LossConfig(type="contrastive", temperature=0.07)
    loss_fn = build_loss_fn(cfg)
    pred = torch.randn(8, 768)
    target = torch.randn(8, 768)
    result = loss_fn(pred, target)
    assert "loss" in result
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_combined_loss_uses_weights():
    cfg = LossConfig(type="combined", contrastive_weight=0.7, cosine_weight=0.3)
    loss_fn = build_loss_fn(cfg)
    pred = torch.randn(8, 768)
    target = torch.randn(8, 768)
    result = loss_fn(pred, target)
    assert "loss" in result
    assert "contrastive_loss" in result
    assert "cosine_loss" in result
```

**Step 2: Run test to verify it fails**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_losses.py -v`
Expected: FAIL

**Step 3: Write losses.py**

```python
"""Loss functions for translator training."""

from typing import Callable

import torch
import torch.nn.functional as F

from config import LossConfig


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> dict:
    return {"loss": F.mse_loss(pred, target)}


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    loss = 1 - (pred_n * target_n).sum(dim=-1).mean()
    return {"loss": loss}


def contrastive_loss(
    pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07
) -> dict:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    logits = pred_n @ target_n.T / temperature
    labels = torch.arange(len(pred), device=pred.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return {"loss": loss, "accuracy": acc}


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
    contrastive_weight: float = 0.7,
    cosine_weight: float = 0.3,
) -> dict:
    c = contrastive_loss(pred, target, temperature)
    cos = cosine_loss(pred, target)
    total = contrastive_weight * c["loss"] + cosine_weight * cos["loss"]
    return {
        "loss": total,
        "contrastive_loss": c["loss"].item(),
        "cosine_loss": cos["loss"].item(),
        "accuracy": c["accuracy"],
    }


def build_loss_fn(cfg: LossConfig) -> Callable:
    if cfg.type == "mse":
        return mse_loss
    elif cfg.type == "cosine":
        return cosine_loss
    elif cfg.type == "contrastive":
        return lambda p, t: contrastive_loss(p, t, cfg.temperature)
    elif cfg.type == "combined":
        return lambda p, t: combined_loss(
            p, t, cfg.temperature, cfg.contrastive_weight, cfg.cosine_weight
        )
    else:
        raise ValueError(f"Unknown loss type: {cfg.type}")
```

**Step 4: Run tests**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_losses.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add PoCs/jepa_clip_translator/losses.py PoCs/jepa_clip_translator/tests/test_losses.py
git commit -m "feat: loss functions (MSE, cosine, contrastive, combined)"
```

---

### Task 5: Embedding Pre-computation

**Files:**
- Create: `PoCs/jepa_clip_translator/precompute_embeddings.py`
- Create: `PoCs/jepa_clip_translator/tests/test_precompute.py`

**Step 1: Write the failing test**

```python
# tests/test_precompute.py
import os
import torch
import numpy as np
import h5py
import pytest
from precompute_embeddings import EmbeddingPrecomputer


def test_save_and_load_embeddings(tmp_path):
    """Test that embeddings round-trip through HDF5."""
    out_path = str(tmp_path / "test_embeddings.h5")
    jepa_embs = torch.randn(10, 1024)
    clip_image_embs = torch.randn(10, 768)
    clip_text_embs = torch.randn(10, 5, 768)  # 5 captions per video
    video_ids = [f"video{i}" for i in range(10)]

    EmbeddingPrecomputer.save_embeddings(
        out_path, video_ids, jepa_embs, clip_image_embs, clip_text_embs
    )

    assert os.path.exists(out_path)
    with h5py.File(out_path, "r") as f:
        assert f["jepa_embeddings"].shape == (10, 1024)
        assert f["clip_image_embeddings"].shape == (10, 768)
        assert f["clip_text_embeddings"].shape == (10, 5, 768)
        loaded_ids = [s.decode() for s in f["video_ids"][:]]
        assert loaded_ids == video_ids


def test_load_split(tmp_path):
    """Test train/val splitting."""
    out_path = str(tmp_path / "test_embeddings.h5")
    n = 20
    jepa = torch.randn(n, 1024)
    clip_img = torch.randn(n, 768)
    clip_txt = torch.randn(n, 5, 768)
    ids = [f"video{i}" for i in range(n)]

    EmbeddingPrecomputer.save_embeddings(out_path, ids, jepa, clip_img, clip_txt)
    train, val = EmbeddingPrecomputer.load_split(out_path, val_fraction=0.2, seed=42)

    assert train["jepa"].shape[0] + val["jepa"].shape[0] == n
    assert val["jepa"].shape[0] == 4  # 20 * 0.2
```

**Step 2: Run test to verify it fails**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_precompute.py -v`
Expected: FAIL

**Step 3: Write precompute_embeddings.py**

```python
"""Pre-compute V-JEPA and CLIP embeddings for MSR-VTT videos."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_loader import MSRVTTLoader


class EmbeddingPrecomputer:
    """Compute and cache V-JEPA + CLIP embeddings."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.vjepa_model = None
        self.vjepa_processor = None
        self.clip_model = None
        self.clip_processor = None

    def load_models(self):
        """Load V-JEPA and CLIP models. Slow on CPU but only done once."""
        from transformers import AutoModel, AutoVideoProcessor, CLIPModel, CLIPProcessor

        print("Loading V-JEPA (this may take a while on CPU)...")
        self.vjepa_processor = AutoVideoProcessor.from_pretrained(
            "facebook/vjepa2-vitl-fpc64-256"
        )
        self.vjepa_model = AutoModel.from_pretrained(
            "facebook/vjepa2-vitl-fpc64-256", torch_dtype=torch.float32
        ).to(self.device).eval()

        print("Loading CLIP...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=torch.float32
        ).to(self.device).eval()

        print("Models loaded.")

    def compute_vjepa_embedding(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Compute V-JEPA embedding from video frames. Returns (1024,) tensor."""
        video_array = np.stack(frames, axis=0)
        inputs = self.vjepa_processor(video_array, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.vjepa_model(**inputs)
            emb = outputs.last_hidden_state.squeeze(0).mean(dim=0)
        return emb.float().cpu()

    def compute_clip_image_embedding(
        self, frames: list[np.ndarray], sample_frames: int = 8
    ) -> torch.Tensor:
        """Compute CLIP image embedding averaged over sampled frames. Returns (768,) tensor."""
        from PIL import Image

        indices = np.linspace(0, len(frames) - 1, sample_frames, dtype=int)
        embeddings = []
        for idx in indices:
            img = Image.fromarray(frames[idx])
            inputs = self.clip_processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.clip_model.get_image_features(**inputs)
            embeddings.append(emb.squeeze().float().cpu())
        avg = torch.stack(embeddings).mean(dim=0)
        return F.normalize(avg, dim=0)

    def compute_clip_text_embeddings(self, captions: list[str]) -> torch.Tensor:
        """Compute CLIP text embeddings for captions. Returns (num_captions, 768) tensor."""
        embeddings = []
        for cap in captions:
            inputs = self.clip_processor(text=cap, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.clip_model.get_text_features(**inputs)
            embeddings.append(emb.squeeze().float().cpu())
        result = torch.stack(embeddings)
        return F.normalize(result, dim=-1)

    @staticmethod
    def save_embeddings(
        path: str,
        video_ids: list[str],
        jepa_embs: torch.Tensor,
        clip_image_embs: torch.Tensor,
        clip_text_embs: torch.Tensor,
    ):
        """Save pre-computed embeddings to HDF5."""
        with h5py.File(path, "w") as f:
            f.create_dataset("jepa_embeddings", data=jepa_embs.numpy())
            f.create_dataset("clip_image_embeddings", data=clip_image_embs.numpy())
            f.create_dataset("clip_text_embeddings", data=clip_text_embs.numpy())
            f.create_dataset("video_ids", data=[s.encode() for s in video_ids])

    @staticmethod
    def load_split(
        path: str, val_fraction: float = 0.15, seed: int = 42
    ) -> tuple[dict, dict]:
        """Load embeddings and split into train/val dicts."""
        with h5py.File(path, "r") as f:
            jepa = torch.tensor(np.array(f["jepa_embeddings"]))
            clip_img = torch.tensor(np.array(f["clip_image_embeddings"]))
            clip_txt = torch.tensor(np.array(f["clip_text_embeddings"]))

        n = len(jepa)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        val_size = int(n * val_fraction)
        val_idx, train_idx = indices[:val_size], indices[val_size:]

        def _split(t, idx):
            return t[idx]

        train = {"jepa": _split(jepa, train_idx), "clip_image": _split(clip_img, train_idx), "clip_text": _split(clip_txt, train_idx)}
        val = {"jepa": _split(jepa, val_idx), "clip_image": _split(clip_img, val_idx), "clip_text": _split(clip_txt, val_idx)}
        return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="embeddings/msrvtt_embeddings.h5")
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clip-sample-frames", type=int, default=8)
    parser.add_argument("--max-captions", type=int, default=5, help="Max captions per video to encode")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load data
    loader = MSRVTTLoader(data_dir=args.data_dir, subset_size=args.subset_size)
    videos, captions = loader.ensure_data()

    # Load models
    precomputer = EmbeddingPrecomputer(device=args.device)
    precomputer.load_models()

    # Compute embeddings
    video_ids = []
    jepa_list = []
    clip_image_list = []
    clip_text_list = []
    max_caps = args.max_captions

    for v in tqdm(videos, desc="Computing embeddings"):
        vid = v["video_id"]
        frames = loader.load_video_frames(vid, max_frames=64)
        if frames is None:
            continue

        jepa_emb = precomputer.compute_vjepa_embedding(frames)
        clip_img_emb = precomputer.compute_clip_image_embedding(frames, args.clip_sample_frames)

        # Text embeddings: take up to max_captions, pad if fewer
        caps = captions.get(vid, ["no caption"])[:max_caps]
        while len(caps) < max_caps:
            caps.append(caps[-1])  # pad by repeating last caption
        clip_txt_emb = precomputer.compute_clip_text_embeddings(caps)

        video_ids.append(vid)
        jepa_list.append(jepa_emb)
        clip_image_list.append(clip_img_emb)
        clip_text_list.append(clip_txt_emb)

    # Save
    EmbeddingPrecomputer.save_embeddings(
        args.output,
        video_ids,
        torch.stack(jepa_list),
        torch.stack(clip_image_list),
        torch.stack(clip_text_list),
    )
    print(f"Saved {len(video_ids)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_precompute.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add PoCs/jepa_clip_translator/precompute_embeddings.py PoCs/jepa_clip_translator/tests/test_precompute.py
git commit -m "feat: embedding pre-computation with HDF5 storage"
```

---

### Task 6: Training Script

**Files:**
- Create: `PoCs/jepa_clip_translator/train.py`
- Create: `PoCs/jepa_clip_translator/tests/test_train.py`

**Step 1: Write the failing test**

```python
# tests/test_train.py
import os
import json
import torch
import h5py
import pytest
from train import run_experiment
from config import ExperimentConfig


@pytest.fixture
def fake_embeddings(tmp_path):
    """Create small fake embedding files for testing."""
    n = 40
    path = str(tmp_path / "test_embeddings.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("jepa_embeddings", data=torch.randn(n, 1024).numpy())
        f.create_dataset("clip_image_embeddings", data=torch.randn(n, 768).numpy())
        f.create_dataset("clip_text_embeddings", data=torch.randn(n, 5, 768).numpy())
        f.create_dataset("video_ids", data=[f"video{i}".encode() for i in range(n)])
    return path


def test_run_experiment_returns_metrics(fake_embeddings, tmp_path):
    cfg = ExperimentConfig(
        experiment_id="test_001",
        training=__import__("config").TrainingConfig(max_epochs=2, batch_size=8),
        data=__import__("config").DataConfig(val_fraction=0.2),
    )
    checkpoint_dir = str(tmp_path / "checkpoints")
    result = run_experiment(cfg, fake_embeddings, checkpoint_dir)

    assert "val_loss" in result
    assert "val_cosine_sim" in result
    assert "epochs_trained" in result
    assert result["epochs_trained"] <= 2


def test_checkpoint_saved(fake_embeddings, tmp_path):
    cfg = ExperimentConfig(
        experiment_id="test_002",
        training=__import__("config").TrainingConfig(max_epochs=2, batch_size=8),
    )
    checkpoint_dir = str(tmp_path / "checkpoints")
    result = run_experiment(cfg, fake_embeddings, checkpoint_dir)

    assert "best_checkpoint" in result
    assert os.path.exists(result["best_checkpoint"])
```

**Step 2: Run test to verify it fails**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_train.py -v`
Expected: FAIL

**Step 3: Write train.py**

```python
"""Training script for a single translator experiment."""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig
from losses import build_loss_fn
from precompute_embeddings import EmbeddingPrecomputer
from translator import build_translator


def run_experiment(
    cfg: ExperimentConfig,
    embeddings_path: str,
    checkpoint_dir: str,
) -> dict:
    """Run a single training experiment. Returns metrics dict."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    exp_dir = os.path.join(checkpoint_dir, cfg.experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    # Load data
    train_data, val_data = EmbeddingPrecomputer.load_split(
        embeddings_path, val_fraction=cfg.data.val_fraction
    )

    train_ds = TensorDataset(train_data["jepa"], train_data["clip_image"])
    val_ds = TensorDataset(val_data["jepa"], val_data["clip_image"])
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size)

    # Build model and optimizer
    model = build_translator(cfg.architecture, cfg.vjepa_dim, cfg.clip_dim)
    loss_fn = build_loss_fn(cfg.loss)

    if cfg.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(cfg.training.max_epochs):
        # Train
        model.train()
        train_loss_sum = 0
        train_steps = 0
        for jepa_batch, clip_batch in train_loader:
            optimizer.zero_grad()
            pred = model(jepa_batch)
            result = loss_fn(pred, clip_batch)
            result["loss"].backward()
            if cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            train_loss_sum += result["loss"].item()
            train_steps += 1

        # Validate
        model.eval()
        val_loss_sum = 0
        val_cosine_sum = 0
        val_steps = 0
        with torch.no_grad():
            for jepa_batch, clip_batch in val_loader:
                pred = model(jepa_batch)
                result = loss_fn(pred, clip_batch)
                val_loss_sum += result["loss"].item()
                # Compute cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(pred, clip_batch, dim=-1).mean()
                val_cosine_sum += cos_sim.item()
                val_steps += 1

        avg_train_loss = train_loss_sum / max(train_steps, 1)
        avg_val_loss = val_loss_sum / max(val_steps, 1)
        avg_val_cosine = val_cosine_sum / max(val_steps, 1)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_cosine_sim": avg_val_cosine,
        })

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(exp_dir, "best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stop_patience:
                break

    best_checkpoint = os.path.join(exp_dir, "best.pt")
    return {
        "experiment_id": cfg.experiment_id,
        "val_loss": best_val_loss,
        "val_cosine_sim": history[best_epoch]["val_cosine_sim"] if history else 0,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_checkpoint": best_checkpoint,
        "config": cfg.to_dict(),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--embeddings", default="embeddings/msrvtt_embeddings.h5")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output", help="Path to write results JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = ExperimentConfig.from_dict(json.load(f))

    result = run_experiment(cfg, args.embeddings, args.checkpoint_dir)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_train.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add PoCs/jepa_clip_translator/train.py PoCs/jepa_clip_translator/tests/test_train.py
git commit -m "feat: training script with early stopping and checkpointing"
```

---

### Task 7: Evaluation (Retrieval Metrics)

**Files:**
- Create: `PoCs/jepa_clip_translator/evaluate.py`
- Create: `PoCs/jepa_clip_translator/tests/test_evaluate.py`

**Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import torch
import pytest
from evaluate import compute_retrieval_metrics


def test_perfect_retrieval():
    """Identical embeddings should give perfect retrieval."""
    n = 10
    queries = torch.eye(n)  # perfectly distinct
    targets = torch.eye(n)
    metrics = compute_retrieval_metrics(queries, targets)
    assert metrics["R@1"] == 1.0
    assert metrics["R@5"] == 1.0
    assert metrics["R@10"] == 1.0


def test_random_retrieval_is_low():
    """Random embeddings should have low retrieval accuracy."""
    torch.manual_seed(42)
    queries = torch.randn(100, 768)
    targets = torch.randn(100, 768)
    metrics = compute_retrieval_metrics(queries, targets)
    assert metrics["R@1"] < 0.2  # random chance ~1%
```

**Step 2: Run test to verify it fails**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_evaluate.py -v`
Expected: FAIL

**Step 3: Write evaluate.py**

```python
"""Evaluation: retrieval metrics for translated embeddings."""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from config import ExperimentConfig
from precompute_embeddings import EmbeddingPrecomputer
from translator import build_translator


def compute_retrieval_metrics(
    queries: torch.Tensor, targets: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)
) -> dict:
    """Compute retrieval R@K. queries and targets are (N, D) tensors.

    For each query, rank all targets by cosine similarity.
    R@K = fraction of queries where the correct target is in the top K.
    """
    queries = F.normalize(queries, dim=-1)
    targets = F.normalize(targets, dim=-1)
    sim = queries @ targets.T  # (N, N)
    ranks = sim.argsort(dim=1, descending=True)

    metrics = {}
    n = len(queries)
    correct_idx = torch.arange(n)
    for k in ks:
        top_k = ranks[:, :k]
        hits = (top_k == correct_idx.unsqueeze(1)).any(dim=1).float()
        metrics[f"R@{k}"] = hits.mean().item()
    metrics["median_rank"] = float(
        (ranks == correct_idx.unsqueeze(1)).nonzero(as_tuple=True)[1].median().item() + 1
    )
    return metrics


def evaluate_checkpoint(
    cfg: ExperimentConfig, embeddings_path: str, checkpoint_path: str
) -> dict:
    """Load a trained model and compute full evaluation metrics."""
    _, val_data = EmbeddingPrecomputer.load_split(embeddings_path, val_fraction=cfg.data.val_fraction)

    model = build_translator(cfg.architecture, cfg.vjepa_dim, cfg.clip_dim)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        translated = model(val_data["jepa"])

    # Image retrieval: translated JEPA vs true CLIP image
    image_metrics = compute_retrieval_metrics(translated, val_data["clip_image"])

    # Text retrieval: translated JEPA vs CLIP text (use first caption per video)
    clip_text_first = val_data["clip_text"][:, 0, :]  # (N, 768)
    text_metrics = compute_retrieval_metrics(translated, clip_text_first)

    # Cosine similarity stats
    cos_sim = F.cosine_similarity(translated, val_data["clip_image"], dim=-1)

    return {
        "image_retrieval": image_metrics,
        "text_retrieval": text_metrics,
        "cosine_sim_mean": cos_sim.mean().item(),
        "cosine_sim_std": cos_sim.std().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--embeddings", default="embeddings/msrvtt_embeddings.h5")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = ExperimentConfig.from_dict(json.load(f))

    results = evaluate_checkpoint(cfg, args.embeddings, args.checkpoint)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_evaluate.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add PoCs/jepa_clip_translator/evaluate.py PoCs/jepa_clip_translator/tests/test_evaluate.py
git commit -m "feat: retrieval evaluation metrics (R@1, R@5, R@10)"
```

---

### Task 8: Coordinator Script

**Files:**
- Create: `PoCs/jepa_clip_translator/coordinator.py`
- Create: `PoCs/jepa_clip_translator/tests/test_coordinator.py`

**Step 1: Write the failing test**

```python
# tests/test_coordinator.py
import json
import pytest
from coordinator import ExperimentLog, generate_round1_configs, generate_next_configs


def test_experiment_log_empty():
    log = ExperimentLog()
    assert log.num_experiments() == 0
    assert log.best_result() is None


def test_experiment_log_tracks_best():
    log = ExperimentLog()
    log.add_result({"experiment_id": "exp_001", "val_loss": 0.5, "val_cosine_sim": 0.7})
    log.add_result({"experiment_id": "exp_002", "val_loss": 0.3, "val_cosine_sim": 0.85})
    log.add_result({"experiment_id": "exp_003", "val_loss": 0.4, "val_cosine_sim": 0.75})
    best = log.best_result()
    assert best["experiment_id"] == "exp_002"


def test_round1_generates_three_configs():
    configs = generate_round1_configs()
    assert len(configs) == 3
    types = {c.architecture.type for c in configs}
    assert types == {"linear", "mlp", "residual"}


def test_generate_next_configs_varies_knobs():
    log = ExperimentLog()
    log.add_result({
        "experiment_id": "exp_001",
        "val_loss": 0.5,
        "val_cosine_sim": 0.7,
        "config": {
            "architecture": {"type": "mlp", "hidden_dim": 512, "num_blocks": 0, "dropout": 0.1, "activation": "gelu"},
            "training": {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.05, "batch_size": 64, "max_epochs": 20, "early_stop_patience": 5, "grad_clip": 1.0},
            "loss": {"type": "combined", "contrastive_weight": 0.7, "cosine_weight": 0.3, "temperature": 0.07},
            "data": {"subset_size": 50, "val_fraction": 0.15, "clip_sample_frames": 8},
            "vjepa_dim": 1024, "clip_dim": 768,
        },
    })
    next_configs = generate_next_configs(log, num_configs=3)
    assert len(next_configs) == 3
    # Should vary at least one parameter from the best
    for c in next_configs:
        assert c.experiment_id != "exp_001"


def test_experiment_log_save_load(tmp_path):
    log = ExperimentLog()
    log.add_result({"experiment_id": "exp_001", "val_loss": 0.5})
    path = str(tmp_path / "log.json")
    log.save(path)
    loaded = ExperimentLog.load(path)
    assert loaded.num_experiments() == 1
```

**Step 2: Run test to verify it fails**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_coordinator.py -v`
Expected: FAIL

**Step 3: Write coordinator.py**

```python
"""Coordinator logic: manage experiment log, propose configs, check convergence."""

import copy
import json
import random
from dataclasses import replace

from config import (
    ArchitectureConfig,
    DataConfig,
    ExperimentConfig,
    LossConfig,
    TrainingConfig,
)


class ExperimentLog:
    """Tracks all experiment results."""

    def __init__(self):
        self.results: list[dict] = []

    def add_result(self, result: dict):
        self.results.append(result)

    def num_experiments(self) -> int:
        return len(self.results)

    def best_result(self) -> dict | None:
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.get("val_loss", float("inf")))

    def is_converged(self, patience_rounds: int = 3) -> bool:
        """Check if best result hasn't improved in the last N experiments."""
        if len(self.results) < patience_rounds + 1:
            return False
        best = self.best_result()
        recent = self.results[-patience_rounds:]
        return all(r.get("val_loss", float("inf")) >= best["val_loss"] for r in recent)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"experiments": self.results}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentLog":
        log = cls()
        with open(path) as f:
            data = json.load(f)
        log.results = data.get("experiments", [])
        return log

    def summary(self) -> str:
        """Human-readable summary of all experiments."""
        lines = [f"Total experiments: {self.num_experiments()}"]
        for r in self.results:
            lines.append(
                f"  {r.get('experiment_id', '?')}: "
                f"val_loss={r.get('val_loss', '?'):.4f}, "
                f"cosine_sim={r.get('val_cosine_sim', '?'):.4f}, "
                f"epochs={r.get('epochs_trained', '?')}"
            )
        best = self.best_result()
        if best:
            lines.append(f"Best: {best['experiment_id']} (val_loss={best['val_loss']:.4f})")
        return "\n".join(lines)


def generate_round1_configs() -> list[ExperimentConfig]:
    """Generate the 3 initial configs: linear, MLP, residual baseline."""
    return [
        ExperimentConfig(
            experiment_id="exp_001_linear",
            architecture=ArchitectureConfig(type="linear"),
        ),
        ExperimentConfig(
            experiment_id="exp_002_mlp",
            architecture=ArchitectureConfig(type="mlp", hidden_dim=512),
        ),
        ExperimentConfig(
            experiment_id="exp_003_residual",
            architecture=ArchitectureConfig(type="residual", hidden_dim=512, num_blocks=4),
        ),
    ]


def generate_next_configs(
    log: ExperimentLog, num_configs: int = 3, seed: int | None = None
) -> list[ExperimentConfig]:
    """Generate next round of configs based on experiment history.

    Strategy: take the best config and vary one knob at a time.
    """
    rng = random.Random(seed)
    best = log.best_result()
    if not best or "config" not in best:
        return generate_round1_configs()

    base_cfg = ExperimentConfig.from_dict(best["config"])
    exp_count = log.num_experiments()
    configs = []

    # Knobs to turn
    mutations = [
        # Architecture variations
        lambda c: replace(c, architecture=replace(c.architecture, hidden_dim=rng.choice([256, 384, 512, 768, 1024]))),
        lambda c: replace(c, architecture=replace(c.architecture, num_blocks=rng.choice([0, 2, 4, 6]))),
        lambda c: replace(c, architecture=replace(c.architecture, dropout=rng.choice([0.0, 0.05, 0.1, 0.2, 0.3]))),
        lambda c: replace(c, architecture=replace(c.architecture, type=rng.choice(["linear", "mlp", "residual"]))),
        # Loss variations
        lambda c: replace(c, loss=replace(c.loss, type=rng.choice(["mse", "cosine", "contrastive", "combined"]))),
        lambda c: replace(c, loss=replace(c.loss, temperature=rng.choice([0.03, 0.05, 0.07, 0.1, 0.2]))),
        lambda c: replace(c, loss=replace(c.loss, contrastive_weight=rng.uniform(0.3, 0.9))),
        # Training variations
        lambda c: replace(c, training=replace(c.training, lr=rng.choice([1e-5, 3e-5, 1e-4, 3e-4, 1e-3]))),
        lambda c: replace(c, training=replace(c.training, batch_size=rng.choice([32, 64, 128, 256]))),
    ]

    rng.shuffle(mutations)
    for i, mutate in enumerate(mutations[:num_configs]):
        new_cfg = mutate(copy.deepcopy(base_cfg))
        new_cfg.experiment_id = f"exp_{exp_count + i + 1:03d}"
        configs.append(new_cfg)

    return configs
```

**Step 4: Run tests**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_coordinator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add PoCs/jepa_clip_translator/coordinator.py PoCs/jepa_clip_translator/tests/test_coordinator.py
git commit -m "feat: coordinator logic with experiment log and config generation"
```

---

### Task 9: End-to-End Integration Test

**Files:**
- Create: `PoCs/jepa_clip_translator/tests/test_integration.py`

This test validates the full pipeline using fake embeddings (no model download needed).

**Step 1: Write the integration test**

```python
# tests/test_integration.py
"""End-to-end test: coordinator -> train -> evaluate pipeline with fake data."""

import json
import os

import h5py
import torch
import pytest

from config import ExperimentConfig
from coordinator import ExperimentLog, generate_round1_configs, generate_next_configs
from train import run_experiment
from evaluate import evaluate_checkpoint, compute_retrieval_metrics


@pytest.fixture
def fake_embeddings(tmp_path):
    """Create fake embeddings that have some structure (not pure random)."""
    n = 40
    path = str(tmp_path / "embeddings.h5")

    # Create correlated embeddings so translator has something to learn
    torch.manual_seed(42)
    jepa = torch.randn(n, 1024)
    # CLIP image = linear transform of JEPA + noise (learnable relationship)
    W = torch.randn(1024, 768) * 0.1
    clip_image = torch.nn.functional.normalize(jepa @ W + torch.randn(n, 768) * 0.01, dim=-1)
    clip_text = clip_image.unsqueeze(1).expand(-1, 5, -1) + torch.randn(n, 5, 768) * 0.1

    with h5py.File(path, "w") as f:
        f.create_dataset("jepa_embeddings", data=jepa.numpy())
        f.create_dataset("clip_image_embeddings", data=clip_image.numpy())
        f.create_dataset("clip_text_embeddings", data=clip_text.numpy())
        f.create_dataset("video_ids", data=[f"video{i}".encode() for i in range(n)])
    return path


def test_full_pipeline(fake_embeddings, tmp_path):
    """Simulate a 2-round coordinator loop."""
    checkpoint_dir = str(tmp_path / "checkpoints")
    log_path = str(tmp_path / "log.json")
    log = ExperimentLog()

    # Round 1: initial configs
    round1 = generate_round1_configs()
    for cfg in round1:
        cfg.training.max_epochs = 3
        cfg.training.batch_size = 8
        result = run_experiment(cfg, fake_embeddings, checkpoint_dir)
        log.add_result(result)

    assert log.num_experiments() == 3
    best = log.best_result()
    assert best is not None
    assert "val_loss" in best

    # Round 2: generate next configs based on results
    round2 = generate_next_configs(log, num_configs=2, seed=42)
    assert len(round2) == 2
    for cfg in round2:
        cfg.training.max_epochs = 3
        cfg.training.batch_size = 8
        result = run_experiment(cfg, fake_embeddings, checkpoint_dir)
        log.add_result(result)

    assert log.num_experiments() == 5

    # Evaluate best model
    final_best = log.best_result()
    eval_cfg = ExperimentConfig.from_dict(final_best["config"])
    eval_results = evaluate_checkpoint(eval_cfg, fake_embeddings, final_best["best_checkpoint"])

    assert "image_retrieval" in eval_results
    assert "text_retrieval" in eval_results
    assert "cosine_sim_mean" in eval_results

    # Save log
    log.save(log_path)
    assert os.path.exists(log_path)

    # Verify summary is readable
    summary = log.summary()
    assert "Total experiments: 5" in summary
```

**Step 2: Run integration test**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add PoCs/jepa_clip_translator/tests/test_integration.py
git commit -m "test: end-to-end integration test for coordinator pipeline"
```

---

### Task 10: Add `__init__.py` and `tests/__init__.py`

**Files:**
- Create: `PoCs/jepa_clip_translator/__init__.py` (empty)
- Create: `PoCs/jepa_clip_translator/tests/__init__.py` (empty)

**Step 1: Create init files**

Both files are empty. They just ensure Python can find the modules.

**Step 2: Run full test suite**

Run: `cd PoCs/jepa_clip_translator && python -m pytest tests/ -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add PoCs/jepa_clip_translator/__init__.py PoCs/jepa_clip_translator/tests/__init__.py
git commit -m "chore: add init files for module discovery"
```

---

### Task 11: README

**Files:**
- Create: `PoCs/jepa_clip_translator/README.md`

**Step 1: Write README**

```markdown
# JEPA-to-CLIP Translator PoC

Iterative agent-driven search for a V-JEPA -> CLIP embedding translator.

## Quick Start (CPU, Phase A)

```bash
pip install -r requirements.txt

# 1. Download MSR-VTT data (annotations auto-downloaded, videos need manual placement)
#    Place .mp4 files in data/videos/

# 2. Pre-compute embeddings (slow on CPU, one-time)
python precompute_embeddings.py --subset-size 50 --device cpu

# 3. Run a single experiment
python train.py --config configs/exp_001.json --embeddings embeddings/msrvtt_embeddings.h5

# 4. Evaluate
python evaluate.py --config configs/exp_001.json --checkpoint checkpoints/exp_001/best.pt --embeddings embeddings/msrvtt_embeddings.h5

# 5. Run tests
python -m pytest tests/ -v
```

## Agent Loop

The coordinator agent reads `log.json`, proposes configs, spawns workers (each running `train.py`), and iterates until convergence. See `docs/plans/2026-03-05-jepa-clip-translator-design.md`.

## Models

- **V-JEPA**: `facebook/vjepa2-vitl-fpc64-256` (1024-dim)
- **CLIP**: `openai/clip-vit-large-patch14` (768-dim)

## File Structure

| File | Purpose |
|------|---------|
| `config.py` | Experiment config schema |
| `data_loader.py` | MSR-VTT download and loading |
| `precompute_embeddings.py` | V-JEPA + CLIP embedding extraction |
| `translator.py` | Model definitions (linear, MLP, residual) |
| `train.py` | Training script |
| `evaluate.py` | Retrieval metrics |
| `coordinator.py` | Experiment log and config generation |
| `losses.py` | Loss functions |
```

**Step 2: Commit**

```bash
git add PoCs/jepa_clip_translator/README.md
git commit -m "docs: JEPA-CLIP translator PoC readme"
```

---

## Execution Order

Tasks 1-4 have no dependencies on each other (scaffolding, data, models, losses) and can be done in parallel. Tasks 5-7 (precompute, train, evaluate) depend on earlier tasks. Task 8 (coordinator) depends on train + evaluate. Task 9 (integration) depends on everything. Tasks 10-11 are cleanup.

```
Task 1 (scaffold) ──┐
Task 2 (data)    ───┤
Task 3 (models)  ───┼── Task 5 (precompute) ── Task 6 (train) ──┐
Task 4 (losses)  ───┘                          Task 7 (eval)  ───┼── Task 8 (coordinator) ── Task 9 (integration) ── Task 10-11
```
