import torch
import pytest
from evaluate import compute_retrieval_metrics


def test_perfect_retrieval():
    n = 10
    queries = torch.eye(n)
    targets = torch.eye(n)
    metrics = compute_retrieval_metrics(queries, targets)
    assert metrics["R@1"] == 1.0
    assert metrics["R@5"] == 1.0
    assert metrics["R@10"] == 1.0
    assert metrics["median_rank"] == 1.0


def test_random_retrieval_is_low():
    torch.manual_seed(42)
    queries = torch.randn(100, 768)
    targets = torch.randn(100, 768)
    metrics = compute_retrieval_metrics(queries, targets)
    assert metrics["R@1"] < 0.2


def test_custom_ks():
    n = 10
    queries = torch.eye(n)
    targets = torch.eye(n)
    metrics = compute_retrieval_metrics(queries, targets, ks=(1, 3))
    assert "R@1" in metrics
    assert "R@3" in metrics
    assert "R@5" not in metrics
