import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from losses import compute_loss, _infonce


def test_mse_loss_nonneg():
    pred, tgt = torch.randn(8, 768), torch.randn(8, 768)
    r = compute_loss(pred, tgt, tgt, [{"function": "mse", "target": "clip_image",
                                        "weight": 1.0, "temperature": 0.07, "label_smoothing": 0.0}])
    assert r["loss"].item() >= 0


def test_cosine_loss_range():
    pred, tgt = torch.randn(8, 768), torch.randn(8, 768)
    r = compute_loss(pred, tgt, tgt, [{"function": "cosine", "target": "clip_image",
                                        "weight": 1.0, "temperature": 0.07, "label_smoothing": 0.0}])
    assert 0.0 <= r["loss"].item() <= 2.0


def test_infonce_loss_nonneg():
    pred, tgt = torch.randn(8, 768), torch.randn(8, 768)
    r = compute_loss(pred, tgt, tgt, [{"function": "infonce", "target": "clip_image",
                                        "weight": 1.0, "temperature": 0.07, "label_smoothing": 0.0}])
    assert r["loss"].item() >= 0


def test_infonce_fn_masking():
    torch.manual_seed(0)
    pred, tgt = torch.randn(8, 64), torch.randn(8, 64)
    # items 0,4 are duplicates (same video); masked = less false negatives = easier task
    loss_masked = _infonce(pred, tgt, batch_indices=[0,1,2,3,0,5,6,7]).item()
    loss_plain  = _infonce(pred, tgt, batch_indices=list(range(8))).item()
    assert loss_masked <= loss_plain + 0.5


def test_infonce_no_batch_indices():
    pred, tgt = torch.randn(8, 768), torch.randn(8, 768)
    loss = _infonce(pred, tgt, batch_indices=None)
    assert loss.shape == ()


def test_compute_loss_accepts_batch_indices():
    pred, tgt = torch.randn(8, 768), torch.randn(8, 768)
    terms = [{"function": "infonce", "target": "clip_image", "weight": 1.0,
              "temperature": 0.07, "label_smoothing": 0.0}]
    r = compute_loss(pred, tgt, tgt, terms, batch_indices=list(range(8)))
    assert r["loss"].item() >= 0


def test_token_level_pred_pooled():
    pred = torch.randn(4, 8, 768)
    tgt  = torch.randn(4, 768)
    terms = [{"function": "mse", "target": "clip_image", "weight": 1.0,
              "temperature": 0.07, "label_smoothing": 0.0}]
    r = compute_loss(pred, tgt, tgt, terms)
    assert r["loss"].shape == ()
