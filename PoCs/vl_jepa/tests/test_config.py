import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from config import LossTerm, ExperimentConfig


def test_infonce_is_valid_loss_function():
    term = LossTerm(function="infonce", target="clip_image", weight=1.0)
    assert term.function == "infonce"


def test_invalid_function_raises():
    with pytest.raises(ValueError, match="function"):
        LossTerm(function="unknown_loss", target="clip_image")


def test_existing_functions_still_valid():
    for fn in ("mse", "cosine", "contrastive"):
        term = LossTerm(function=fn, target="clip_image")
        assert term.function == fn


def test_experiment_config_roundtrip_with_infonce():
    cfg = ExperimentConfig(experiment_id="test_infonce")
    cfg.loss.terms = [{"function": "infonce", "target": "clip_image", "weight": 1.0,
                       "temperature": 0.07, "label_smoothing": 0.0}]
    d = cfg.to_dict()
    cfg2 = ExperimentConfig.from_dict(d)
    assert cfg2.loss.terms[0]["function"] == "infonce"
