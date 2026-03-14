"""
Evaluation Harness — V-JEPA → CLIP Alignment

DO NOT MODIFY THIS FILE DURING EXPERIMENTATION.

This eval is self-sufficient:
- Discovers validation data by scanning for .npy files
- Discovers dimensions from the data itself
- Can run in --synthetic mode to test the interface with random data
- Validates the projector contract (input shape → output shape, no NaN)
- Reports structured metrics for the harness to parse

Projector Interface:
    The --projector argument points to a Python module that must define:

    def load(checkpoint_path: str = None) -> Any:
        '''Load/initialize the projector. Return any state object.'''

    def project(state: Any, vjepa_embeddings: np.ndarray) -> np.ndarray:
        '''
        Map V-JEPA embeddings into CLIP space.
        Input:  (N, D_vjepa) numpy array — any D_vjepa
        Output: (N, D_clip) numpy array — must match clip validation embeddings
        '''

Usage:
    # Normal eval (discovers data automatically)
    python eval/test_alignment.py --projector workspace/projector.py

    # With explicit checkpoint
    python eval/test_alignment.py --projector workspace/projector.py --checkpoint best.pt

    # Synthetic mode (test interface without real data)
    python eval/test_alignment.py --projector workspace/projector.py --synthetic

    # Specify data directory
    python eval/test_alignment.py --projector workspace/projector.py --data-dir data/val

    # Verbose (per-sample stats, shape info)
    python eval/test_alignment.py --projector workspace/projector.py --verbose
"""

import sys
import os
import argparse
import glob
import importlib.util
import json
import traceback
import numpy as np
from pathlib import Path
from typing import Any, Callable, Optional


# ============================================================================
# Data Discovery — find validation embeddings automatically
# ============================================================================

# Patterns that identify V-JEPA embedding files
VJEPA_PATTERNS = [
    "*vjepa*val*.npy", "*vjepa*test*.npy", "*vjepa*embed*.npy",
    "*v_jepa*val*.npy", "*v_jepa*embed*.npy",
    "*jepa*val*.npy", "*jepa*embed*.npy",
    "vjepa.npy", "vjepa_val.npy", "vjepa_embeddings.npy",
    "*source*val*.npy", "*source*embed*.npy",
    "*input*val*.npy", "*input*embed*.npy",
]

# Patterns that identify CLIP embedding files
CLIP_PATTERNS = [
    "*clip*val*.npy", "*clip*test*.npy", "*clip*embed*.npy",
    "clip.npy", "clip_val.npy", "clip_embeddings.npy",
    "*target*val*.npy", "*target*embed*.npy",
    "*output*val*.npy", "*output*embed*.npy",
]


def find_embedding_files(search_dirs: list[Path]) -> tuple[Optional[Path], Optional[Path]]:
    """
    Scan directories for V-JEPA and CLIP embedding .npy files.
    Returns (vjepa_path, clip_path) or (None, None) if not found.
    """
    vjepa_file = None
    clip_file = None

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Search recursively
        all_npy = list(search_dir.rglob("*.npy"))

        if not vjepa_file:
            for pattern in VJEPA_PATTERNS:
                matches = [f for f in all_npy if f.match(pattern)]
                if matches:
                    vjepa_file = matches[0]
                    break

        if not clip_file:
            for pattern in CLIP_PATTERNS:
                matches = [f for f in all_npy if f.match(pattern)]
                if matches:
                    clip_file = matches[0]
                    break

        if vjepa_file and clip_file:
            break

    return vjepa_file, clip_file


def describe_npy(path: Path) -> dict:
    """Load and describe a .npy file without loading fully into memory."""
    arr = np.load(str(path), mmap_mode='r')
    return {
        "path": str(path),
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "has_nan": bool(np.isnan(arr).any()) if np.issubdtype(arr.dtype, np.floating) else False,
        "has_inf": bool(np.isinf(arr).any()) if np.issubdtype(arr.dtype, np.floating) else False,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


# ============================================================================
# Projector Loading
# ============================================================================

def load_projector_module(projector_path: str) -> tuple[Callable, Callable]:
    """Dynamically import a projector module."""
    path = Path(projector_path)
    if not path.exists():
        raise FileNotFoundError(f"Projector not found: {projector_path}")

    spec = importlib.util.spec_from_file_location("projector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "load"):
        raise AttributeError(f"{projector_path} must define: load(checkpoint_path=None) -> Any")
    if not hasattr(module, "project"):
        raise AttributeError(
            f"{projector_path} must define: project(state, vjepa_embeddings: np.ndarray) -> np.ndarray"
        )

    return module.load, module.project


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(
    load_fn: Callable,
    project_fn: Callable,
    vjepa_embs: np.ndarray,
    clip_embs: np.ndarray,
    checkpoint_path: Optional[str] = None,
    max_samples: int = 0,
    verbose: bool = False,
) -> dict:
    """
    Run evaluation. Discovers all dimensions from the data.
    Returns metrics dict.
    """
    # Validate input shapes
    assert vjepa_embs.ndim == 2, f"V-JEPA embeddings must be 2D, got shape {vjepa_embs.shape}"
    assert clip_embs.ndim == 2, f"CLIP embeddings must be 2D, got shape {clip_embs.shape}"
    assert vjepa_embs.shape[0] == clip_embs.shape[0], (
        f"Sample count mismatch: V-JEPA has {vjepa_embs.shape[0]}, "
        f"CLIP has {clip_embs.shape[0]}"
    )

    n_samples = vjepa_embs.shape[0]
    d_vjepa = vjepa_embs.shape[1]
    d_clip = clip_embs.shape[1]

    if verbose:
        print(f"  V-JEPA: {n_samples} samples, {d_vjepa}d")
        print(f"  CLIP:   {n_samples} samples, {d_clip}d")

    # Subsample if requested
    if max_samples > 0 and n_samples > max_samples:
        idx = np.random.RandomState(42).choice(n_samples, max_samples, replace=False)
        vjepa_embs = vjepa_embs[idx]
        clip_embs = clip_embs[idx]
        n_samples = max_samples
        if verbose:
            print(f"  Subsampled to {n_samples}")

    # Load projector
    if verbose:
        print(f"\n  Loading projector...")

    state = load_fn(checkpoint_path)

    # Run projection
    if verbose:
        print(f"  Projecting {n_samples} samples from {d_vjepa}d → {d_clip}d...")

    projected = project_fn(state, vjepa_embs)

    # Validate output
    assert isinstance(projected, np.ndarray), (
        f"project() must return np.ndarray, got {type(projected)}"
    )
    assert projected.ndim == 2, (
        f"project() output must be 2D, got shape {projected.shape}"
    )
    assert projected.shape[0] == n_samples, (
        f"project() returned {projected.shape[0]} samples, expected {n_samples}"
    )
    assert projected.shape[1] == d_clip, (
        f"project() output dim is {projected.shape[1]}, expected {d_clip} (CLIP dim). "
        f"The projector must map into CLIP's embedding space."
    )

    # Check NaN/Inf
    has_nan = bool(np.any(np.isnan(projected)))
    has_inf = bool(np.any(np.isinf(projected)))
    stability = 0.0 if (has_nan or has_inf) else 1.0

    if has_nan or has_inf:
        print(f"  ⚠️  {'NaN' if has_nan else ''} {'Inf' if has_inf else ''} in projected embeddings!")
        return {
            "mean_cosine_similarity": 0.0,
            "retrieval_accuracy_at_5": 0.0,
            "training_stability": 0.0,
            "n_samples": n_samples,
            "d_vjepa": d_vjepa,
            "d_clip": d_clip,
        }

    # Cosine similarity
    proj_norm = projected / (np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8)
    clip_norm = clip_embs / (np.linalg.norm(clip_embs, axis=1, keepdims=True) + 1e-8)
    pair_cosine = np.sum(proj_norm * clip_norm, axis=1)
    mean_cosine = float(np.mean(pair_cosine))

    if verbose:
        print(f"\n  Cosine similarity:")
        print(f"    Mean:   {mean_cosine:.4f}")
        print(f"    Median: {float(np.median(pair_cosine)):.4f}")
        print(f"    Std:    {float(np.std(pair_cosine)):.4f}")
        print(f"    Min:    {float(np.min(pair_cosine)):.4f}")
        print(f"    Max:    {float(np.max(pair_cosine)):.4f}")

    # Retrieval accuracy
    pool = min(100, n_samples)
    sim_matrix = (proj_norm[:pool]) @ (clip_norm[:pool].T)

    correct_at_1 = 0
    correct_at_5 = 0
    for i in range(pool):
        ranked = np.argsort(sim_matrix[i])[::-1]
        if ranked[0] == i:
            correct_at_1 += 1
        if i in ranked[:5]:
            correct_at_5 += 1

    r_at_1 = correct_at_1 / pool
    r_at_5 = correct_at_5 / pool

    if verbose:
        print(f"\n  Retrieval (pool={pool}):")
        print(f"    R@1:  {r_at_1:.4f}")
        print(f"    R@5:  {r_at_5:.4f}")

    return {
        "mean_cosine_similarity": round(mean_cosine, 6),
        "retrieval_accuracy_at_5": round(r_at_5, 6),
        "retrieval_accuracy_at_1": round(r_at_1, 6),
        "training_stability": stability,
        "n_samples": n_samples,
        "d_vjepa": d_vjepa,
        "d_clip": d_clip,
    }


def check_pass(metrics: dict) -> bool:
    """Check against success criteria from goal.yaml."""
    if metrics.get("mean_cosine_similarity", 0) < 0.70:
        return False
    if metrics.get("training_stability", 0) < 1.0:
        return False
    return True


# ============================================================================
# Synthetic Data (for interface testing)
# ============================================================================

def make_synthetic(d_vjepa: int = 768, d_clip: int = 512, n: int = 200) -> tuple:
    """
    Generate synthetic paired embeddings for testing the eval pipeline.
    
    Creates data with a known linear relationship plus noise, so a perfect 
    projector would get ~0.5-0.7 cosine sim (proving metrics are meaningful
    even on fake data).
    """
    rng = np.random.RandomState(42)

    # Random "ground truth" projection
    W_true = rng.randn(d_vjepa, d_clip).astype(np.float32)
    W_true /= np.linalg.norm(W_true, axis=0, keepdims=True)

    # Generate V-JEPA embeddings
    vjepa = rng.randn(n, d_vjepa).astype(np.float32)

    # Generate CLIP embeddings as a noisy projection of V-JEPA
    clip = vjepa @ W_true + 0.3 * rng.randn(n, d_clip).astype(np.float32)

    return vjepa, clip


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V-JEPA → CLIP Alignment Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Discovery:
  The eval scans for .npy files matching common naming patterns.
  Recognized V-JEPA names: *vjepa*, *jepa*, *source*, *input*
  Recognized CLIP names:   *clip*, *target*, *output*
  
  Files should be 2D numpy arrays with shape (N, D).
  V-JEPA and CLIP files must have the same N (number of samples).

Synthetic Mode:
  Use --synthetic to test with random data. This verifies:
  - Your projector loads without errors
  - Input/output shapes are correct
  - No NaN/Inf in output
  - The eval pipeline works end to end
  
  Metrics on synthetic data are meaningless for the actual goal.
        """,
    )
    parser.add_argument("--projector", required=True,
                        help="Path to projector module (.py with load/project)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path for projector's load()")
    parser.add_argument("--data-dir", default=None,
                        help="Directory to scan for .npy files (default: scan experiment dir)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max validation samples (0 = use all)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (for interface testing)")
    parser.add_argument("--synthetic-d-vjepa", type=int, default=768,
                        help="V-JEPA dim for synthetic data")
    parser.add_argument("--synthetic-d-clip", type=int, default=512,
                        help="CLIP dim for synthetic data")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--discover-only", action="store_true",
                        help="Just find and describe data files, don't evaluate")
    args = parser.parse_args()

    # Find experiment root (this script lives in eval/)
    eval_dir = Path(__file__).parent
    experiment_dir = eval_dir.parent

    print(f"Eval: V-JEPA → CLIP Alignment")
    print(f"Projector: {args.projector}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print()

    # ====================================================================
    # Step 1: Get data (discover or synthesize)
    # ====================================================================

    if args.synthetic:
        print(f"🧪 Synthetic mode: {args.synthetic_d_vjepa}d → {args.synthetic_d_clip}d")
        print(f"   Metrics will be meaningless — this only tests the interface.\n")
        vjepa_embs, clip_embs = make_synthetic(
            d_vjepa=args.synthetic_d_vjepa,
            d_clip=args.synthetic_d_clip,
        )

    else:
        # Discover validation data
        search_dirs = []
        if args.data_dir:
            search_dirs.append(Path(args.data_dir))
        # Also search common locations relative to experiment
        search_dirs.extend([
            experiment_dir / "data" / "val",
            experiment_dir / "data",
            experiment_dir / "workspace" / "data",
            experiment_dir / "workspace",
            experiment_dir,
        ])

        vjepa_path, clip_path = find_embedding_files(search_dirs)

        if args.discover_only:
            print("Data discovery results:")
            if vjepa_path:
                info = describe_npy(vjepa_path)
                print(f"  V-JEPA: {info['path']}")
                print(f"    shape={info['shape']} dtype={info['dtype']} range=[{info['min']:.3f}, {info['max']:.3f}]")
            else:
                print(f"  V-JEPA: NOT FOUND")
                print(f"    Searched: {[str(d) for d in search_dirs]}")
                print(f"    Patterns: {VJEPA_PATTERNS[:4]}...")

            if clip_path:
                info = describe_npy(clip_path)
                print(f"  CLIP:   {info['path']}")
                print(f"    shape={info['shape']} dtype={info['dtype']} range=[{info['min']:.3f}, {info['max']:.3f}]")
            else:
                print(f"  CLIP:   NOT FOUND")
            sys.exit(0)

        if not vjepa_path or not clip_path:
            missing = []
            if not vjepa_path:
                missing.append("V-JEPA")
            if not clip_path:
                missing.append("CLIP")

            print(f"❌ Validation data not found: {', '.join(missing)}")
            print()
            print("The eval needs paired embedding files as .npy arrays.")
            print(f"Searched directories: {[str(d) for d in search_dirs if d.exists()]}")
            print()
            print("To generate validation data:")
            print("  1. Run V-JEPA encoder on validation videos → save as *vjepa*val*.npy")
            print("  2. Run CLIP encoder on same content → save as *clip*val*.npy")
            print("  3. Both must be shape (N, D) with matching N")
            print()
            print("Or use --synthetic to test the projector interface with random data.")
            print("Or use --discover-only to see what files were found.")
            sys.exit(2)

        print(f"Found V-JEPA embeddings: {vjepa_path}")
        print(f"Found CLIP embeddings:   {clip_path}")

        vjepa_embs = np.load(str(vjepa_path))
        clip_embs = np.load(str(clip_path))

        if args.verbose:
            for name, arr, path in [("V-JEPA", vjepa_embs, vjepa_path), ("CLIP", clip_embs, clip_path)]:
                info = describe_npy(path)
                print(f"  {name}: {info['shape']} {info['dtype']} "
                      f"range=[{info['min']:.3f}, {info['max']:.3f}] mean={info['mean']:.3f}")

    print()

    # ====================================================================
    # Step 2: Load and run projector
    # ====================================================================

    try:
        load_fn, project_fn = load_projector_module(args.projector)
    except (FileNotFoundError, AttributeError) as e:
        print(f"❌ {e}")
        sys.exit(2)

    try:
        metrics = evaluate(
            load_fn=load_fn,
            project_fn=project_fn,
            vjepa_embs=vjepa_embs,
            clip_embs=clip_embs,
            checkpoint_path=args.checkpoint,
            max_samples=args.max_samples,
            verbose=args.verbose,
        )
    except AssertionError as e:
        print(f"❌ Contract violation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Eval crashed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ====================================================================
    # Step 3: Report
    # ====================================================================

    print()
    for key, value in metrics.items():
        print(f"[METRIC] {key}={value}")

    if args.synthetic:
        print(f"\n⚠️  Synthetic data — metrics are not meaningful for the goal.")
        print(f"[EVAL] INTERFACE OK ✅" if metrics["training_stability"] == 1.0
              else "[EVAL] INTERFACE FAILED ❌")
    else:
        passed = check_pass(metrics)
        print(f"\n[EVAL] {'PASS ✅' if passed else 'FAIL ❌'}")

    if args.output_json:
        output = {**metrics, "synthetic": args.synthetic}
        if not args.synthetic:
            output["passed"] = check_pass(metrics)
        Path(args.output_json).write_text(json.dumps(output, indent=2))

    if args.synthetic:
        sys.exit(0)
    else:
        sys.exit(0 if check_pass(metrics) else 1)


if __name__ == "__main__":
    main()
